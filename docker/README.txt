Set Host network adapter to 192.168.1.2 (see: https://github.com/unitreerobotics/unilidar_sdk2)

Check if the lidar can be reached: ping 192.169.1.62

Tested with windows as host!
do everything using wsl and ubuntu.

What it does:
- Start Lidar Driver
- Make Lidar available as ros2 topic
- Rviz2 opens to visualize pointcloud.

build:
docker build -t unilidar_sdk2_foxy .

run:
docker run -it --rm \
  -v "$(pwd)/lidar_data:/root/ros2_ws/lidar_data" \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  unilidar_sdk2_foxy
  
 launch:
 ros2 launch unitree_lidar_ros2 launch.py
 
 
 save data as ros2bag:
 ros2 bag record -s mcap -o ~/ros2_ws/lidar_data/my_lidar_bag10 /unilidar/cloud
 
 
 additional terminal:
 docker exec -it <container_id> bash
 
 
 docker id:
 docker ps -a
