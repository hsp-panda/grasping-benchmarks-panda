ARG from=ubuntu:bionic
FROM ${from}

# Libs to control Panda robot
# ===========================
ENV ROS_DISTRO=melodic

RUN apt-get update &&\
    apt install -y --no-install-recommends \
        ros-${ROS_DISTRO}-libfranka \
        ros-${ROS_DISTRO}-franka-ros \
        ros-${ROS_DISTRO}-catkin \
        python-catkin-tools \
        ros-${ROS_DISTRO}-moveit \
        &&\
    rm -rf /var/lib/apt/lists/*

ENV PANDA_CATKIN_WS=/panda_catkin_ws

RUN mkdir -p ${PANDA_CATKIN_WS}/src &&\
    cd ${PANDA_CATKIN_WS}/src &&\
    git clone -b ${ROS_DISTRO}-devel https://github.com/ros-planning/moveit_tutorials.git && \
    git clone -b ${ROS_DISTRO}-devel https://github.com/ros-planning/panda_moveit_config.git && \
    apt-get update &&\    
    rosdep install -y --from-paths . --ignore-src --rosdistro ${ROS_DISTRO} && \
    rm -rf /var/lib/apt/lists/* &&\
    cd ${PANDA_CATKIN_WS} && \
    catkin config --extend /opt/ros/${ROS_DISTRO} && \
    catkin build && \
    echo 'source ${PANDA_CATKIN_WS}/devel/setup.bash' >> /etc/bash.bashrc




