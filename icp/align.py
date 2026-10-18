#!/usr/bin/env python3
"""Stage 2: align a site scan to a BIM reference cloud (from generate.py). Paired by filename stem:
data/scans/203.ply registers against data/bim/203.ply.

  python icp/align.py --list-pairs             # what can be aligned
  python icp/align.py --pair 203               # one room (no flag: pair: in config.yaml)
  python icp/align.py --all-pairs              # every room with both clouds
  python icp/align.py --method all             # bake-off: fpfh vs pca vs yaw
  python icp/align.py --method manual --pick   # click matching points in two windows
  python icp/align.py --source a.ply --target b.ply --show
  python icp/align.py --apply results/<run>/transform.npy --show
  python icp/align.py --pair 203 --inspect     # levelled scan overview PNG, to read off a crop_xy box

Pipeline: level the scan (floor normal -> +Z), drop locally-sparse outliers, optional crop_xy, keep a
0.6-1.9 m wall band, voxel + normals, footprint check, coarse (yaw sweep by default), ICP down a
distance ladder, score. Each run writes results/<stamp>_<room>_<method>[_tag]/: transform.npy/.txt
(4x4 scan -> BIM world, levelling folded in), report.json, config_used.yaml, source_aligned.ply,
overlay.png, residual.png; results/latest.txt points at the newest.

Read the footprint check first: if scan and BIM rectangles differ by > ~100 mm the residual is floored
by that difference however well ICP converges (real unmodelled recess, scan seeing through a door, or
wrong crop). Scores: fitness = share of scan points within score_threshold, rmse over those,
median/p90 over ALL points (the honest number). Low fitness + good median = unmatched geometry, fine.
Coarse methods: yaw (reliable), fpfh (weak on lattice-sampled BIM clouds), pca (fails on near-square
rooms), manual (Kabsch from >= 3 picked pairs; the fallback when yaw lands 90/180 deg out).
"""
import argparse, json, sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
import yaml  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
reg = o3d.pipelines.registration
ROOT = Path(__file__).resolve().parent
CLOUD_EXT = (".ply", ".pcd", ".xyz")
SRC_C, TGT_C = [0.90, 0.35, 0.10], [0.20, 0.45, 0.85]  # scan orange, BIM blue


# ---------------------------------------------------------------- prep
def load(path):
    pc = o3d.io.read_point_cloud(str(path))
    if not len(pc.points):
        raise ValueError(f"no points read from {path}")
    return pc


def cloud(X):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))


def moved(pc, M):
    a = o3d.geometry.PointCloud(pc); a.transform(M)
    return a


def downsample(pc, voxel, normal_radius=None):
    d = pc.voxel_down_sample(voxel) if voxel and voxel > 0 else o3d.geometry.PointCloud(pc)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius or voxel * 5, max_nn=30))
    d.orient_normals_consistent_tangent_plane(20)
    return d


def level(pc, dist=0.03, max_planes=6, min_nz=0.9):
    """Rotate the most horizontal of several RANSAC planes (largest plane is usually a wall) to +Z, floor to z=0.
    Seeded: segment_plane otherwise gives a different tilt every run. -> (cloud, 4x4, tilt_deg)."""
    o3d.utility.random.seed(0)
    rest, best = o3d.geometry.PointCloud(pc), None
    for _ in range(max_planes):
        if len(rest.points) < 500:
            break
        pl, inl = rest.segment_plane(dist, 3, 600)
        n = np.array(pl[:3]); n = n if n[2] > 0 else -n
        if n[2] > min_nz and (best is None or len(inl) > best[1]):
            best = (n, len(inl), float(np.median(np.asarray(rest.points)[inl][:, 2])))
        rest = rest.select_by_index(inl, invert=True)
    if best is None:
        return o3d.geometry.PointCloud(pc), np.eye(4), 0.0
    n, _, z0 = best
    axis = np.cross(n, [0, 0, 1.0]); s = np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) if s < 1e-9 else np.eye(3) + K + K @ K * ((1 - n[2]) / s ** 2)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = -R @ [0, 0, z0]
    return moved(pc, T), T, float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))


def wall_slice(pc, lo, hi):
    """Horizontal band above the cloud's own min z: floor/ceiling otherwise dominate and starve the yaw."""
    X = np.asarray(pc.points); z = X[:, 2] - X[:, 2].min()
    return cloud(X[(z > lo) & (z < hi)])


def in_box(X, b):
    return (X[:, 0] >= b[0]) & (X[:, 0] <= b[2]) & (X[:, 1] >= b[1]) & (X[:, 1] <= b[3])


def crop_xy(pc, box, exclude=None):
    """Keep XY inside box (minus an optional exclude box): only for a genuinely dense second structure."""
    X = np.asarray(pc.points); keep = in_box(X, box)
    if exclude:
        keep &= ~in_box(X, exclude)
    return cloud(X[keep])


# ---------------------------------------------------------------- coarse / refine
def icp(src, tgt, init, thr, iters, p2plane=False):
    est = reg.TransformationEstimationPointToPlane() if p2plane else reg.TransformationEstimationPointToPoint()
    return reg.registration_icp(src, tgt, thr, init, est, reg.ICPConvergenceCriteria(max_iteration=iters))


def rigid(R, cs, ct):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = ct - R @ cs
    return M


def best_of(src, tgt, Ms, thr, iters):
    """Candidate whose short point-to-point ICP reaches the highest fitness (first one wins ties)."""
    f = [icp(src, tgt, M, thr, iters).fitness for M in Ms]
    return Ms[int(np.argmax(f))]


def centroids(src, tgt):
    return np.asarray(src.points).mean(0), np.asarray(tgt.points).mean(0)


def coarse_fpfh(src, tgt, voxel, seed=None):
    if seed is not None:
        o3d.utility.random.seed(int(seed))
    feat = lambda p: reg.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    d = voxel * 1.5
    return reg.registration_ransac_based_on_feature_matching(
        src, tgt, feat(src), feat(tgt), True, d, reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(d)],
        reg.RANSACConvergenceCriteria(4000000, 0.999)).transformation


def coarse_pca(src, tgt):
    """Centroid + principal axes, best of the 4 sign flips."""
    def axes(p):
        X = np.asarray(p.points); X = X - X.mean(0)
        return np.linalg.eigh(X.T @ X)[1][:, ::-1]
    As, At = axes(src), axes(tgt)
    Ms = []
    for sx in (1, -1):
        for sy in (1, -1):
            Rs = As * [sx, sy, 1]; Rs[:, 2] = np.cross(Rs[:, 0], Rs[:, 1])
            Ms.append(rigid(At @ Rs.T, *centroids(src, tgt)))
    return best_of(src, tgt, Ms, 0.10, 60)


def coarse_yaw(src, tgt, step=2.0, thr=0.30):
    """Yaw sweep about Z with centroid translation (both clouds must be gravity-aligned)."""
    Ms = [rigid(np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]]), *centroids(src, tgt))
          for a in np.radians(np.arange(0, 360, step))]
    return best_of(src, tgt, Ms, thr, 40)


def kabsch(P, Q):
    """Rigid fit from >= 3 hand-picked correspondences (Nx3, same order)."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    if P.shape != Q.shape or len(P) < 3:
        raise ValueError("need >=3 matching point pairs")
    U, _, Vt = np.linalg.svd((P - P.mean(0)).T @ (Q - Q.mean(0)))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return rigid(R, P.mean(0), Q.mean(0))


def refine(src, tgt, M, ladder, iters, p2plane):
    """ICP down a descending distance ladder. Point-to-plane on a wall-only slice is rank deficient (no z
    constraint) and returns NaN / fitness 0: such steps are skipped."""
    M, hist = np.array(M, float), []
    for thr in ladder:
        r = icp(src, tgt, M, thr, iters, p2plane)
        if not np.all(np.isfinite(r.transformation)) or r.fitness == 0.0:
            hist.append((thr, 0.0, float("nan"))); continue
        M = r.transformation; hist.append((thr, float(r.fitness), float(r.inlier_rmse)))
    return M, hist


def score(src, tgt, M, thr):
    d = np.asarray(moved(src, M).compute_point_cloud_distance(tgt))
    ev = reg.evaluate_registration(src, tgt, thr, M)
    return {"fitness": float(ev.fitness), "inlier_rmse_mm": float(ev.inlier_rmse) * 1000,
            "median_mm": float(np.median(d)) * 1000, "p90_mm": float(np.percentile(d, 90)) * 1000}


def pose(M):
    return {"yaw_deg": float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
            "translation_m": [round(float(v), 4) for v in M[:3, 3]]}


# ---------------------------------------------------------------- footprint check
def min_rect(Q):
    """Minimum-area rectangle of Nx2 points -> ((short, long), R, center); R's columns = (short, long) axes."""
    h = Q[ConvexHull(Q).vertices]
    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]; e = e / np.linalg.norm(e)
        R = np.array([[e[0], -e[1]], [e[1], e[0]]])
        q = h @ R; lo, hi = q.min(0), q.max(0); w, dd = hi - lo
        if best is None or w * dd < best[0]:
            best = (w * dd, w, dd, R, R @ ((lo + hi) / 2))
    _, w, dd, R, c = best
    return (float(min(w, dd)), float(max(w, dd))), (R[:, ::-1] if w > dd else R), c


def trimmed_xy(pc, trim=0.02):
    """XY + keep-mask without the most centroid-distant 2 %: screens stray sparse returns only."""
    Q = np.asarray(pc.points)[:, :2]; d = np.linalg.norm(Q - Q.mean(0), axis=1)
    return Q, d < np.quantile(d, 1 - trim)


def diagnose(src, tgt):
    a, b = (min_rect(Q[k])[0] for Q, k in (trimmed_xy(src), trimmed_xy(tgt)))
    return {"source_footprint_m": [round(v, 3) for v in a], "target_footprint_m": [round(v, 3) for v in b],
            "delta_mm": [round((a[i] - b[i]) * 1000, 1) for i in (0, 1)]}


# ---------------------------------------------------------------- pictures
def footprint_png(src, tgt, path):
    """Sizes only (not registered yet): scan in its own min-rect axes vs the BIM room rectangle."""
    Qs, ks = trimmed_xy(src); (ss, sl), Rs, cs = min_rect(Qs[ks]); Q = (Qs - cs) @ Rs
    Qt, kt = trimmed_xy(tgt); (ts, tl), _, _ = min_rect(Qt[kt])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(*Q[~ks].T, s=5, c="lightsalmon", label="scan (trimmed)"); ax.scatter(*Q[ks].T, s=5, c="firebrick", label="scan")
    for (w, h), col, lab in (((ss, sl), "darkorange", "scan min-rect"), ((ts, tl), "seagreen", "bim room size")):
        ax.add_patch(plt.Rectangle((-w / 2, -h / 2), w, h, fill=False, edgecolor=col, linewidth=2, label=lab))
    ax.set_title("footprint check: scan vs BIM (size only -- not yet registered)")
    save(fig, ax, path, 120, legend=dict(markerscale=3, fontsize=8, loc="upper right"))


def overview_png(pc, path):
    X = np.asarray(pc.points)
    fig, ax = plt.subplots(figsize=(11, 11))
    sca = ax.scatter(X[:, 0], X[:, 1], s=1, c=X[:, 2], cmap="viridis")
    ax.set_xticks(np.arange(np.floor(X[:, 0].min()), np.ceil(X[:, 0].max()) + 1))
    ax.set_yticks(np.arange(np.floor(X[:, 1].min()), np.ceil(X[:, 1].max()) + 1))
    ax.tick_params(labelsize=7); ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.set_title("levelled scan, XY, 1 m grid -- read off crop_xy: [xmin, ymin, xmax, ymax]")
    plt.colorbar(sca, ax=ax, shrink=0.7, label="z (m, floor=0)")
    save(fig, ax, path, 130)


def overlay_png(src, tgt, M, path):
    A, B = np.asarray(moved(src, M).points), np.asarray(tgt.points)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (i, j, name) in zip(axs, [(0, 1, "plan XY"), (0, 2, "elevation XZ")]):
        ax.scatter(B[:, i], B[:, j], s=0.4, c=[TGT_C], label="target (BIM)")
        ax.scatter(A[:, i], A[:, j], s=0.4, c=[SRC_C], label="source (scan)")
        ax.set_title(name); ax.set_aspect("equal"); ax.legend(markerscale=12, fontsize=8)
    save(fig, None, path, 110)


def residual_png(src, tgt, M, path, clip_mm=200):
    a = moved(src, M); d = np.asarray(a.compute_point_cloud_distance(tgt)) * 1000; A = np.asarray(a.points)
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.colorbar(ax.scatter(A[:, 0], A[:, 1], s=0.8, c=np.clip(d, 0, clip_mm), cmap="turbo"), label="distance to target [mm]")
    ax.set_title("residual after ICP")
    save(fig, ax, path, 110)


def save(fig, ax, path, dpi, legend=None):
    if ax is not None:
        ax.set_aspect("equal")
        if legend:
            ax.legend(**legend)
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)


def show(src, tgt, M, title):
    a = moved(src, M); a.paint_uniform_color(SRC_C)
    b = o3d.geometry.PointCloud(tgt); b.paint_uniform_color(TGT_C)
    o3d.visualization.draw_geometries([a, b, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)], window_name=title)


def pick(pc, title):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title); vis.add_geometry(pc); vis.run(); vis.destroy_window()
    return np.asarray(pc.points)[vis.get_picked_points()]


# ---------------------------------------------------------------- pairing / jobs
def res(p):
    """Absolute, or relative to the current directory if it exists there, else relative to icp/."""
    p = Path(p)
    return p if p.is_absolute() or p.exists() else ROOT / p


def stems(folder):
    f = res(folder)
    return {x.stem: x for x in sorted(f.iterdir()) if x.is_file() and x.suffix.lower() in CLOUD_EXT} if f.is_dir() else {}


def jobs(cfg, a):
    """[(name, scan, bim)] for this invocation."""
    s, b = stems(cfg["scans_dir"]), stems(cfg["bim_dir"])
    if a.source or a.target:
        if not (a.source and a.target):
            raise SystemExit("--source and --target must be given together")
        return [(res(a.source).stem, res(a.source), res(a.target))]
    if a.all_pairs:
        if not set(s) & set(b):
            raise SystemExit(f"no matching stems in {res(cfg['scans_dir'])} and {res(cfg['bim_dir'])}")
        return [(k, s[k], b[k]) for k in sorted(set(s) & set(b))]
    w = a.pair or cfg.get("pair")
    if not w:
        raise SystemExit("nothing to align: pass --pair NAME, --all-pairs, or set pair: in config.yaml (try --list-pairs)")
    for d, kind, extra in ((s, "scan", ""), (b, "BIM cloud", f"\ngenerate it: python icp/generate.py --rooms {w}")):
        if w not in d:
            raise SystemExit(f"no {kind} named '{w}'\navailable: {', '.join(d) or '(none)'}{extra}")
    return [(w, s[w], b[w])]


def preprocess(p, source, target):
    src_raw, tgt_raw = load(source), load(target)
    src, tgt, L = src_raw, tgt_raw, np.eye(4)
    if p.get("level_source"):
        src, L, tilt = level(src); print(f"  source levelled, floor tilt was {tilt:.2f} deg")
    if p.get("level_target"):
        tgt, _, tilt = level(tgt); print(f"  target levelled, floor tilt was {tilt:.2f} deg")
    if p.get("outlier_removal"):
        n = len(src.points); src = src.remove_statistical_outlier(**p["outlier_removal"])[0]
        print(f"  outlier removal {p['outlier_removal']} -> {n} -> {len(src.points)} pts")
    if p.get("crop_xy"):
        n = len(src.points); src = crop_xy(src, p["crop_xy"], p.get("crop_exclude_xy"))
        print(f"  crop_xy {p['crop_xy']} -> {n} -> {len(src.points)} pts")
    if p.get("wall_slice"):
        src, tgt = wall_slice(src, *p["wall_slice"]), wall_slice(tgt, *p["wall_slice"])
        print(f"  wall slice {p['wall_slice']} m -> {len(src.points)} / {len(tgt.points)} pts")
    return src_raw, tgt_raw, downsample(src, p["voxel"], p.get("normal_radius")), downsample(tgt, p["voxel"], p.get("normal_radius")), L


def coarse(m, s, t, cfg, a):
    c = cfg["coarse"]
    if m == "fpfh":
        return coarse_fpfh(s, t, cfg["preprocess"]["voxel"], c.get("fpfh_seed"))
    if m == "pca":
        return coarse_pca(s, t)
    if m == "yaw":
        return coarse_yaw(s, t, c["yaw_step_deg"], c["yaw_thr"])
    if m == "manual":
        if a.pick:
            print("  pick the SAME features in both windows, in the same order")
            return kabsch(pick(s, "SOURCE: shift+click, then close"), pick(t, "TARGET: shift+click, then close"))
        if c.get("manual_pairs"):
            return kabsch(c["manual_pairs"]["source"], c["manual_pairs"]["target"])
        raise SystemExit("manual needs coarse.manual_pairs in config, or --pick")
    raise SystemExit(f"unknown coarse method: {m}")


def align_pair(name, source, target, cfg, a, base):
    cfg = {**cfg, "preprocess": {**cfg["preprocess"], **cfg.get("per_room", {}).get(name, {}).get("preprocess", {})}}
    chosen = a.method or cfg["coarse"]["method"]
    if a.out:
        out = res(a.out); out.mkdir(parents=True, exist_ok=True)
    else:
        stem = "_".join(filter(None, [datetime.now().strftime("%Y%m%d-%H%M%S"), name, chosen, a.tag]))
        out, n = base / stem, 1
        while out.exists():
            out, n = base / f"{stem}_{n}", n + 1
        out.mkdir(parents=True)
    print(f"\n=== {name}\n  scan   {source}\n  bim    {target}\n  out    {out}\n\nloading and preprocessing")
    src_raw, tgt_raw, s, t, L = preprocess(cfg["preprocess"], source, target)

    d = diagnose(s, t)
    print(f"\nfootprint check  source {d['source_footprint_m']} m   target {d['target_footprint_m']} m   delta {d['delta_mm']} mm")
    if max(map(abs, d["delta_mm"])) > 100:
        footprint_png(s, t, out / "footprint_debug.png")
        print("  WARNING: footprints disagree by more than 100 mm -- see footprint_debug.png. This caps achievable RMSE;"
              "\n  it can be real unmodelled geometry or a scan seeing through a doorway, not only a crop/file mismatch.")

    r = cfg["refine"]
    results = {}
    if a.apply:
        results["applied"] = (np.load(res(a.apply)) @ np.linalg.inv(L), [])  # saved one has levelling folded in
    for m in [] if a.apply else (["fpfh", "pca", "yaw"] if chosen == "all" else [chosen]):
        print(f"\ncoarse method: {m}")
        M0 = coarse(m, s, t, cfg, a)
        M, hist = refine(s, t, M0, r["ladder"], r["max_iterations"], r["point_to_plane"])
        if r["point_to_plane"] and (not np.all(np.isfinite(M)) or hist[-1][1] == 0.0):
            print("  point-to-plane collapsed (rank deficient), retrying point-to-point")
            M, hist = refine(s, t, M0, r["ladder"], r["max_iterations"], False)
        for thr, fit, rmse in hist:
            print(f"    thr {thr * 1000:5.0f} mm   fitness {fit:.3f}   rmse {rmse * 1000:6.1f} mm")
        results[m] = (M, [{"threshold_mm": h * 1000, "fitness": f, "rmse_mm": e * 1000} for h, f, e in hist])

    report = {"run": {"name": name, "timestamp": datetime.now().isoformat(timespec="seconds"), "directory": str(out),
                      "source": str(source), "target": str(target), "methods": list(results), "tag": a.tag},
              "footprint": d, "results": {}}
    print(f"\n{'method':10s} {'fitness':>8s} {'rmse_mm':>9s} {'median_mm':>10s} {'p90_mm':>8s} {'yaw_deg':>9s}")
    best = None
    for k, (M, ladder) in results.items():
        sc, pz = score(s, t, M, cfg["evaluate"]["score_threshold"]), pose(M @ L)
        report["results"][k] = {**sc, **pz, "ladder": ladder, "transform": (M @ L).tolist()}
        print(f"{k:10s} {sc['fitness']:8.3f} {sc['inlier_rmse_mm']:9.1f} {sc['median_mm']:10.1f} {sc['p90_mm']:8.1f} {pz['yaw_deg']:9.2f}")
        key = (-sc["median_mm"], sc["fitness"])  # overall agreement, not inlier count
        if best is None or key > best[0]:
            best = (key, k)
    bk = report["best"] = best[1]
    Mb = results[bk][0] @ L  # levelling folded in

    np.save(out / "transform.npy", Mb); np.savetxt(out / "transform.txt", Mb, fmt="%.9f")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "config_used.yaml").write_text(yaml.safe_dump({**cfg, "source": str(source), "target": str(target)}, sort_keys=False))
    o3d.io.write_point_cloud(str(out / "source_aligned.ply"), moved(load(source), Mb))
    if cfg["viewer"]["save_png"]:
        overlay_png(src_raw, tgt_raw, Mb, out / "overlay.png")
        residual_png(s, t, results[bk][0], out / "residual.png")
    print(f"\nbest: {bk}   -> {out}")
    if a.show or cfg["viewer"]["show_window"]:
        show(src_raw, tgt_raw, Mb, f"{name} {bk} alignment")
    return out, report["results"][bk], bk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--pair", help="filename stem present in both data/scans and data/bim")
    ap.add_argument("--all-pairs", action="store_true")
    ap.add_argument("--list-pairs", action="store_true")
    ap.add_argument("--source", help="explicit scan path, overrides pairing")
    ap.add_argument("--target", help="explicit BIM cloud path, overrides pairing")
    ap.add_argument("--method", help="fpfh | pca | yaw | manual | all")
    ap.add_argument("--tag", help="label appended to the run directory name")
    ap.add_argument("--pick", action="store_true", help="click correspondences interactively (manual)")
    ap.add_argument("--apply", help="skip registration: score/render a saved 4x4 .npy")
    ap.add_argument("--show", action="store_true", help="open the Open3D viewer afterwards")
    ap.add_argument("--out", help="write here instead of a new timestamped run directory")
    ap.add_argument("--inspect", action="store_true", help="write <pair>_scan_overview.png and exit")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))["align"]

    if a.list_pairs:
        s, b = stems(cfg["scans_dir"]), stems(cfg["bim_dir"])
        both = sorted(set(s) & set(b))
        print("alignable pairs:\n" + "".join(f"  {k:<20} {s[k].name}  <->  {b[k].name}\n" for k in both) if both
              else "no pairs found: no filename stem appears in both folders")
        if set(s) - set(b):
            print("scan with no BIM cloud: " + ", ".join(sorted(set(s) - set(b))) + "\n  generate it: python icp/generate.py --rooms <room>")
        if set(b) - set(s):
            print("BIM cloud with no scan: " + ", ".join(sorted(set(b) - set(s))))
        return 0 if both else 1

    if a.inspect:
        if not (a.pair or a.source):
            raise SystemExit("--inspect needs --pair NAME or --source PATH")
        name, src = (res(a.source).stem, res(a.source)) if a.source else (a.pair, stems(cfg["scans_dir"]).get(a.pair))
        if src is None:
            raise SystemExit(f"no scan named '{a.pair}' in {cfg['scans_dir']}")
        scan = load(src)
        if {**cfg["preprocess"], **cfg.get("per_room", {}).get(name, {}).get("preprocess", {})}.get("level_source"):
            scan, _, tilt = level(scan); print(f"levelled, floor tilt was {tilt:.2f} deg")
        out = res(cfg["out_dir"]) / f"{name}_scan_overview.png"; out.parent.mkdir(parents=True, exist_ok=True)
        overview_png(scan, out)
        print(f"wrote {out}\nread off [xmin, ymin, xmax, ymax] -> per_room.{name}.preprocess.crop_xy in config.yaml")
        return 0

    js = jobs(cfg, a)
    for _, sp, tp in js:
        for label, p in (("scan", sp), ("bim cloud", tp)):
            if not Path(p).is_file():
                raise SystemExit(f"{label} not found: {p}")
    if len(js) > 1 and (a.out or a.pick):
        raise SystemExit("--out / --pick work on a single pair only")
    base = res(cfg["out_dir"]); base.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, sp, tp in js:
        try:
            out, sc, m = align_pair(name, sp, tp, cfg, a, base)
        except Exception as exc:  # keep a batch going
            if len(js) == 1:
                raise
            print(f"\n{name} FAILED: {exc}"); summary.append((name, None, None, None)); continue
        summary.append((name, m, sc["median_mm"], out))
        (base / "latest.txt").write_text(str(out), encoding="utf-8")
    if len(js) > 1:
        print(f"\n{'pair':<20} {'method':<8} {'median_mm':>10}  run")
        for name, m, med, out in summary:
            print(f"{name:<20} {'FAILED':<8} {'-':>10}" if m is None else f"{name:<20} {m:<8} {med:10.1f}  {Path(out).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
