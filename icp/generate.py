#!/usr/bin/env python3
"""Stage 1: IFC -> LiDAR-comparable BIM reference cloud per room (data/bim/<room>.ply + .json).

  python icp/generate.py --list                 # rooms in the IFC
  python icp/generate.py --rooms 203 207        # (no flag: rooms: in config.yaml)
  python icp/generate.py --all --spacing-xy 0.03
  python icp/generate.py --rooms 203 --ceiling-mode fixed --clear-height 2.60

A Revit IfcSpace solid (export with "Export rooms and spaces in 3D views") is bounded by the inner wall
faces, i.e. what a LiDAR in the room sees. Its triangles are classified by normal (vertical = wall,
down = floor, up = ceiling, dropped by default) and grid-sampled on a lattice anchored at the world
origin, so coplanar triangles share one seamless grid. Walls are clipped at the detected ceiling (lowest
IfcCovering/IfcSlab over the room, else the space top). Door openings (IfcDoor bbox overlapping the
room footprint + 0.3 m, padded by --opening-margin) are carved out so the cloud isn't solid where a scan
sees through a doorway; windows only with --window-openings. Output stays in IFC world coordinates, so
an ICP transform maps straight into the BIM frame. The stem is the bare room number so align.py pairs it.
"""
import argparse, json, logging
from pathlib import Path

import ifcopenshell, ifcopenshell.geom, ifcopenshell.util.element as ue
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("generate")


def mesh(product, settings):
    """World-coordinate (verts, tris) of a product, or None (no shape / Revit's ~1e62 junk coordinates)."""
    try:
        shape = ifcopenshell.geom.create_shape(settings, product)  # keep a reference: a freed shape reads as empty
    except Exception as exc:  # noqa: BLE001
        log.debug("no shape for #%s: %s", product.id(), exc); return None
    v, f = np.asarray(shape.geometry.verts, float).reshape(-1, 3), np.asarray(shape.geometry.faces, int).reshape(-1, 3)
    return (v, f) if len(v) and len(f) and np.isfinite(v).all() and np.abs(v).max() <= 1e6 else None


def storey(p):
    c = ue.get_container(p) or ue.get_aggregate(p)  # spaces in this model are aggregated, not contained
    return str(getattr(c, "Name", "") or "")


def load_rooms(model, settings, wanted):
    """IfcSpaces matching Name (number) or LongName, case-insensitive; Revit phase duplicates merged."""
    wanted = {w.strip().lower() for w in wanted} if wanted else None
    rooms, seen, merged = [], set(), 0
    for sp in model.by_type("IfcSpace"):
        num, name = (sp.Name or "").strip(), (sp.LongName or "").strip()
        if wanted is not None and num.lower() not in wanted and name.lower() not in wanted:
            continue
        m = mesh(sp, settings)
        if m is None:
            log.warning("space '%s' (#%s) has no usable geometry", num, sp.id()); continue
        key = (len(m[0]), *np.round(m[0].min(0), 4), *np.round(m[0].max(0), 4))
        if key in seen:
            merged += 1; continue
        seen.add(key)
        rooms.append(dict(number=num, name=name, global_id=sp.GlobalId, storey=storey(sp), verts=m[0], tris=m[1]))
    return rooms, merged


def element_boxes(model, settings, cls):
    """[(class#id, bbox_lo, bbox_hi)] for every element of an IFC class (computed once per run)."""
    out = []
    for e in model.by_type(cls):
        m = mesh(e, settings)
        if m is not None:
            out.append((f"{cls}#{e.id()}", m[0].min(0), m[0].max(0)))
    return out


def ceiling_z(boxes, lo, hi, floor_z, top_z, min_clear=1.6, inset=0.15):
    """Lowest covering/slab overlapping the (inset) footprint and plausibly above the floor."""
    best = None
    for src, elo, ehi in boxes:
        if ehi[0] <= lo[0] + inset or elo[0] >= hi[0] - inset or ehi[1] <= lo[1] + inset or elo[1] >= hi[1] - inset:
            continue
        if floor_z + min_clear <= elo[2] <= top_z + 0.5 and (best is None or elo[2] < best[0]):
            best = (float(elo[2]), src)
    return best or (top_z, "space_solid_top")


def opening_boxes(boxes, lo, hi, floor_z, top_z, margin, fp=0.30):
    """Doors/windows on this room's boundary (they sit in the wall, outside the space solid), padded, to the floor."""
    out = []
    for _, elo, ehi in boxes:
        if ehi[0] < lo[0] - fp or elo[0] > hi[0] + fp or ehi[1] < lo[1] - fp or elo[1] > hi[1] + fp \
                or ehi[2] < floor_z - 0.3 or elo[2] > top_z + 0.3:
            continue
        blo = elo - margin; blo[2] = min(blo[2], floor_z)
        out.append((blo, ehi + margin))
    return out


def sample_triangle(tri, sa, sb, drop):
    """Lattice points (world-anchored, spacing sa/sb on the kept axes) inside one triangle."""
    keep = [i for i in range(3) if i != drop]
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    if abs(n[drop]) < 1e-12:
        return np.empty((0, 3))
    t2 = tri[:, keep]
    av = np.arange(np.ceil(t2[:, 0].min() / sa), np.floor(t2[:, 0].max() / sa) + 1) * sa
    bv = np.arange(np.ceil(t2[:, 1].min() / sb), np.floor(t2[:, 1].max() / sb) + 1) * sb
    if not av.size or not bv.size:
        return np.empty((0, 3))
    aa, bb = np.meshgrid(av, bv, indexing="ij")
    p = np.column_stack([aa.ravel(), bb.ravel()])
    v0, v1, v2 = t2[1] - t2[0], t2[2] - t2[0], p - t2[0]
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < 1e-15:
        return np.empty((0, 3))
    u, v = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den, (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
    p = p[(u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)]
    out = np.empty((len(p), 3)); out[:, keep] = p
    out[:, drop] = (np.dot(n, tri[0]) - p @ n[keep]) / n[drop]  # from the plane equation
    return out


def sample_room(r, ceil_z, sxy, sz, floor, walls, ceiling, holes):
    """Sample the room's surfaces -> (points, n_wall, n_floor)."""
    floor_z = float(r["verts"][:, 2].min())
    parts = {"wall": [], "floor": [], "ceil": []}
    for tri in r["verts"][r["tris"]]:
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n = n / ln
        if abs(n[2]) <= 0.05 and walls:  # vertical: drop the dominant horizontal axis
            d = 0 if abs(n[0]) >= abs(n[1]) else 1
            parts["wall"].append(sample_triangle(tri, *[sz if k == 2 else sxy for k in range(3) if k != d], d))
        elif n[2] < -0.05 and floor:
            parts["floor"].append(sample_triangle(tri, sxy, sxy, 2))
        elif n[2] > 0.05 and ceiling:
            parts["ceil"].append(sample_triangle(tri, sxy, sxy, 2))
    W, F, C = (np.vstack(parts[k]) if parts[k] else np.empty((0, 3)) for k in ("wall", "floor", "ceil"))
    for lo, hi in holes:
        W = W[~np.all((W >= lo) & (W <= hi), axis=1)]
    W = W[(W[:, 2] >= floor_z - 1e-6) & (W[:, 2] <= ceil_z + 1e-6)]  # open at the top
    C = C[np.abs(C[:, 2] - ceil_z) < 1e-6]
    pts, nw, nf = np.vstack([W, F, C]), len(W), len(F)
    _, idx = np.unique(np.round(pts / 1e-4).astype(np.int64), axis=0, return_index=True)
    if len(idx) != len(pts):  # duplicates on shared triangle edges; the wall/floor split is diagnostic only
        pts = pts[np.sort(idx)]
        nf = int(np.sum(np.abs(pts[:, 2] - floor_z) < 1e-6)) if floor else 0
        nw = len(pts) - nf
    return pts, nw, nf


def write_ply(path, pts, ascii_ply):
    head = (f"ply\nformat {'ascii' if ascii_ply else 'binary_little_endian'} 1.0\ncomment Generated by SMCODE pointcloudgenerator\n"
            f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    with open(path, "w" if ascii_ply else "wb", **({"newline": "\n"} if ascii_ply else {})) as fh:
        if ascii_ply:
            fh.write(head); np.savetxt(fh, pts, fmt="%.6f")
        else:
            fh.write(head.encode("ascii")); pts.astype("<f4").tofile(fh)


def slug(t):
    return "".join(c if c.isalnum() else "_" for c in t.strip() if c.isalnum() or c in " -_").strip("_")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--ifc", type=Path); p.add_argument("--out", type=Path)
    p.add_argument("--rooms", nargs="+", help="room numbers or names")
    p.add_argument("--all", action="store_true"); p.add_argument("--list", action="store_true")
    p.add_argument("--spacing-xy", type=float); p.add_argument("--spacing-z", type=float)
    p.add_argument("--ceiling-mode", choices=("auto", "space_top", "fixed"))
    p.add_argument("--clear-height", type=float, help="for --ceiling-mode fixed")
    for f in ("--no-floor", "--no-walls", "--include-ceiling", "--no-door-openings", "--window-openings", "--binary"):
        p.add_argument(f, action="store_true")
    p.add_argument("--opening-margin", type=float)
    p.add_argument("--name-by", choices=("number", "number_name", "name"))
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s")
    c = yaml.safe_load(a.config.read_text())["generate"]
    pick = lambda arg, key: arg if arg is not None else c[key]
    ifc = a.ifc or ROOT / c["ifc"]
    out = a.out or ROOT / c["out_dir"]
    if not Path(ifc).is_file():
        raise SystemExit(f"IFC not found: {ifc} (set generate.ifc in config.yaml or pass --ifc)")
    model = ifcopenshell.open(str(ifc))

    if a.list:
        rows = sorted({(storey(s), (s.Name or "").strip(), (s.LongName or "").strip()) for s in model.by_type("IfcSpace")})
        for st, num, name in rows:
            print(f"{num:<10} {name:<32} {st}")
        print(f"\n{len(model.by_type('IfcSpace'))} spaces, {len(rows)} unique")
        return 0

    wanted = None if a.all else (a.rooms or c.get("rooms"))
    if not a.all and not wanted:
        raise SystemExit("nothing to do: pass --rooms, --all, or set rooms: in config.yaml")
    settings = ifcopenshell.geom.settings(); settings.set("use-world-coords", True)
    rooms, merged = load_rooms(model, settings, wanted)
    if not rooms:
        raise SystemExit(f"no matching spaces in {Path(ifc).name} (try --list)")

    sxy, sz = pick(a.spacing_xy, "spacing_xy"), pick(a.spacing_z, "spacing_z")
    mode, clear = a.ceiling_mode or c["ceiling_mode"], pick(a.clear_height, "clear_height")
    floor, walls = not a.no_floor and c["floor"], not a.no_walls and c["walls"]
    ceiling, ascii_ply = a.include_ceiling or c["ceiling"], not a.binary and c["ascii_ply"]
    margin, name_by = pick(a.opening_margin, "opening_margin"), a.name_by or c["name_by"]
    cut = [k for k, on in (("IfcDoor", not a.no_door_openings and c["doors"]), ("IfcWindow", a.window_openings or c["windows"])) if on]
    ceil_boxes = element_boxes(model, settings, "IfcCovering") + element_boxes(model, settings, "IfcSlab") if mode == "auto" else []
    hole_boxes = [b for k in cut for b in element_boxes(model, settings, k)] if walls else []

    fails = 0
    for r in rooms:
        label, lo, hi = f"{r['number']} ({r['name']})", r["verts"].min(0), r["verts"].max(0)
        fz, tz, warn = float(lo[2]), float(hi[2]), []
        try:
            if mode == "fixed":
                if clear is None:
                    raise ValueError("ceiling_mode='fixed' requires clear_height")
                cz, csrc = fz + float(clear), f"fixed({clear})"
            elif mode == "space_top":
                cz, csrc = tz, "space_solid_top"
            else:
                cz, csrc = ceiling_z(ceil_boxes, lo, hi, fz, tz)
                if csrc == "space_solid_top":
                    warn.append("no ceiling element found overlapping the room; used the space solid top (storey height). "
                                "Set ceiling_mode='fixed' with a measured clear_height if the scan disagrees.")
            if cz <= fz:
                raise ValueError(f"ceiling_z {cz} is not above floor {fz}")
            holes = opening_boxes(hole_boxes, lo, hi, fz, tz, margin)
            if cut and walls and not holes:
                warn.append("no door/window openings found on this room's boundary; wall cloud will be solid where a scan would see through a doorway")
            pts, nw, nf = sample_room(r, cz, sxy, sz, floor, walls, ceiling, holes)
            if not len(pts):
                raise RuntimeError("sampling produced no points")
        except Exception as exc:  # noqa: BLE001
            fails += 1; log.error("%s failed: %s", label, exc); continue

        num, nm = slug(r["number"]) or r["global_id"], slug(r["name"])
        stem = c["prefix"] + {"number_name": f"{num}_{nm}", "name": nm}.get(name_by if nm else "number", num)
        Path(out).mkdir(parents=True, exist_ok=True)
        write_ply(Path(out) / f"{stem}.ply", pts, ascii_ply)
        plo, phi = pts.min(0), pts.max(0)
        rnd = lambda v: [round(float(x), 4) for x in v]
        (Path(out) / f"{stem}.json").write_text(json.dumps({
            "room_number": r["number"], "room_name": r["name"], "global_id": r["global_id"], "storey": r["storey"],
            "points": {"total": len(pts), "wall": nw, "floor": nf},
            "heights": {"floor_z": round(fz, 4), "ceiling_z": round(cz, 4), "clear_height": round(cz - fz, 4), "ceiling_source": csrc},
            "bbox_world": {"min": rnd(plo), "max": rnd(phi), "extent": rnd(phi - plo)}, "local_offset": rnd(-plo),
            "sampling": {"xy": sxy, "z": sz}, "duplicates_merged": merged, "openings_cut": len(holes), "warnings": warn}, indent=2))
        e = phi - plo
        log.info("%s -> %s.ply | %d pts (%d wall / %d floor) | %.3f x %.3f x %.3f m | ceiling %s | %d opening(s) cut",
                 label, stem, len(pts), nw, nf, *e, csrc, len(holes))
        for w in warn:
            log.warning("%s: %s", label, w)
    if merged:
        log.info("%d duplicate space(s) merged", merged)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
