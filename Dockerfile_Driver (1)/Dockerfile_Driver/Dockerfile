FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=foxy

SHELL ["/bin/bash", "-c"]

# 1. Install base utilities, configure locale, and add ROS 2 repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    ca-certificates \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# 2. Install build tools, PCL 1.10, and ROS 2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    libpcl-dev \
    libeigen3-dev \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-foxy-ros-base \
    ros-foxy-pcl-ros \
    ros-foxy-pcl-conversions \
    ros-foxy-sensor-msgs \
    ros-foxy-std-msgs \
    ros-foxy-rviz2 \
	ros-foxy-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

# 3. Initialize rosdep
RUN rosdep init && rosdep update

# 4. Clone unilidar_sdk2 repository
WORKDIR /root
RUN git clone https://github.com/unitreerobotics/unilidar_sdk2.git

# 5. Build unitree_lidar_sdk C++ library
WORKDIR /root/unilidar_sdk2/unitree_lidar_sdk
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

# 6. Build unitree_lidar_ros2 package inside ROS 2 workspace
WORKDIR /root/ros2_ws/src
RUN ln -s /root/unilidar_sdk2/unitree_lidar_ros2 .

WORKDIR /root/ros2_ws
RUN source /opt/ros/foxy/setup.bash && \
    colcon build --symlink-install

# 7. Auto-source environment on login
RUN echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> /root/.bashrc

WORKDIR /root/ros2_ws
CMD ["/bin/bash"]