# syntax=docker/dockerfile:experimental

ARG from=ubuntu:bionic
FROM ${from}

# Change default shell to bash
SHELL ["/bin/bash", "-c"]

# Install packages
# ================
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        sudo \
        gosu \
        software-properties-common \
        git \
        wget \
        curl \
        nano \
        unzip \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        apt-transport-https \
        apt-utils \
        dbus-x11 \
        bash-completion \
        gdb \
        ssh \
        openssh-client \
        iputils-ping \
        gnupg \
        gnupg2 \
        xvfb \
        &&\
    rm -rf /var/lib/apt/lists/*

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ `lsb_release -cs` main" &&\
    add-apt-repository ppa:ubuntu-toolchain-r/test &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends \
        gcc-8 \
        g++-8 \
        cmake \
        cmake-curses-gui \
        build-essential \
        &&\
    rm -rf /var/lib/apt/lists/*

# Setup locales and timezone
ENV TZ=Europe/Rome
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update &&\
    apt-get install -y --no-install-recommends locales locales-all tzdata &&\
    rm -rf /var/lib/apt/lists/* &&\
    echo "${TZ}" | tee /etc/timezone &&\
    dpkg-reconfigure --frontend noninteractive tzdata


# Install python 3
# ================
ENV PYTHON_VERSION=3.6
RUN add-apt-repository ppa:deadsnakes/ppa &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        libpython${PYTHON_VERSION}-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        python3-pygments \
        python3-numpy \
        &&\
    rm -rf /var/lib/apt/lists/*


# Install ROS Desktop Full
# ========================

# https://github.com/osrf/docker_images/blob/master/ros/
ENV ROS_DISTRO=melodic

RUN apt-key adv --keyserver keyserver.ubuntu.com \
                --recv-keys F42ED6FBAB17C654 &&\
    echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -cs` main" \
        > /etc/apt/sources.list.d/ros-latest.list &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-desktop-full \
	ros-${ROS_DISTRO}-jsk-tools \
        python-rosdep \
        python-rosinstall \
        python-rosinstall-generator \
        python-vcstools \
        python-wstool \
        python-catkin-tools \
        python3-catkin-pkg-modules \
        python3-rospkg-modules \
        python3-empy \
        &&\
    rm -rf /var/lib/apt/lists/* &&\
    rosdep init &&\
    rosdep update &&\
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc


# Install superquadric-lib and its dependencies (VTK, Ipopt, SWIG)
# ===============================================================

ENV WS_DIR=/workspace
ENV CATKIN_WS_DIR=${WS_DIR}/catkin_ws
ENV WS_INSTALL=${WS_DIR}/install
ENV WS_SOURCES=${WS_DIR}/sources
ARG WS_BIN=${WS_SOURCES}/bin

ENV CC=gcc-8
ENV CXX=g++-8

# VTK
RUN mkdir -p ${WS_SOURCES} && cd ${WS_SOURCES} &&\
    git clone https://gitlab.kitware.com/vtk/vtk.git && \
    cd vtk && git checkout tags/v8.2.0 &&\
    mkdir -p build && cd build && \
    cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${WS_INSTALL} \
      -DVTK_PYTHON_VERSION=${PYTHON_VERSION} \
      -DVTK_WRAP_PYTHON=ON \
      .. &&\
    make -j8 && make install &&\
    rm -rf ${WS_SOURCES}/vtk

# SWIG
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ `lsb_release -cs` main" &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends \
        autotools-dev \
        automake \
        bison \
        libpcre3-dev &&\
    rm -rf /var/lib/apt/lists/* &&\
    cd /tmp/ &&\
    curl https://netcologne.dl.sourceforge.net/project/swig/swig/swig-4.0.1/swig-4.0.1.tar.gz | tar xvz &&\
    cd swig-4.0.1 &&\
    ./configure && \
    make -j4 &&\
    make install &&\
    rm -r /tmp/swig-4.0.1

# IPOPT
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        file \
        gfortran \
        &&\
    rm -rf /var/lib/apt/lists/* &&\
    mkdir -p ${WS_SOURCES}/ipopt_tmp && cd ${WS_SOURCES}/ipopt_tmp &&\
    curl -o coinbrew https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew &&\
    chmod +x coinbrew &&\
    ./coinbrew fetch Ipopt:releases/3.13.0 --no-prompt &&\
    ./coinbrew build Ipopt --prefix=${WS_INSTALL} --test --no-prompt &&\
    ./coinbrew install Ipopt --no-prompt && \
    rm -rf ${WS_SOURCES}/ipopt_tmp


ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${WS_INSTALL}/lib
ENV IPOPT_DIR=${WS_INSTALL}

# Create a new runtimeusers group
RUN groupadd -K GID_MIN=900 -K GID_MAX=1100 runtimeusers

# Install superquadric-lib
RUN mkdir ${WS_INSTALL}/include/coin/ &&\
    cp -r ${WS_INSTALL}/include/coin-or/* ${WS_INSTALL}/include/coin/ && \
    cd ${WS_SOURCES} &&\
    git clone https://github.com/robotology/superquadric-lib.git && \
    cd ${WS_SOURCES}/superquadric-lib &&\
    mkdir -p build && cd build &&\
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${WS_INSTALL} \
        -DENABLE_BINDINGS=ON \
        -DYARP_EXE=OFF \
        -DVTK_DIR=${WS_INSTALL}/lib/cmake/vtk-8.2 \
        .. &&\
    make -j8 && make install &&\
    find ${IIT_DIR} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+ &&\
    find ${IIT_INSTALL} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+

ENV PYTHONPATH=${PYTHONPATH}:${WS_INSTALL}/lib/superquadriclib/bindings


# Create catkin_ws with dependencies/ros packages for python3
# ===========================================================
RUN apt-get update &&\
    pip3 install pyyaml &&\
    sudo apt-get install -y libboost-all-dev &&\
    mkdir -p ${CATKIN_WS_DIR}/src && \
    cd ${CATKIN_WS_DIR} &&\
    wstool init &&\
    wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5 &&\
    wstool set -y src/vision_opencv --git https://github.com/ros-perception/vision_opencv -v 1.13.0 &&\
    wstool set -y src/ros_numpy --git https://github.com/eric-wieser/ros_numpy &&\
    wstool up &&\
    rosdep update && rosdep install --from-paths src --ignore-src -y -r &&\
    rm -rf /var/lib/apt/lists/* &&\
    catkin config --extend /opt/ros/${ROS_DISTRO} -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release && \
    catkin build  -j8 && \
    echo 'source ${CATKIN_WS_DIR}/devel/setup.bash' >> /etc/bash.bashrc &&\
    find ${CATKIN_WS_DIR} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+

# Install benchmark-grasping
# ==========================

#RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN cd ${CATKIN_WS_DIR}/src &&\
    git clone https://github.com/hsp-panda/panda_ros_common &&\
    git clone -b fc_grasp_demo https://github.com/hsp-panda/grasping-benchmarks-panda.git &&\
    pip3 install grasping-benchmarks-panda/ &&\
    cd ${CATKIN_WS_DIR} &&\
    catkin config --extend /opt/ros/${ROS_DISTRO} -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release -DGPD_ROS=OFF && \
    catkin build  -j8 && \
    find ${CATKIN_WS_DIR} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+


# =============
# FIX OWNERSHIP
# =============

RUN find ${WS_DIR} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+ &&\
    find ${WS_INSTALL} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+ &&\
    find ${CATKIN_WS_DIR} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+

# Nvidia runtime support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Image initialization
# ====================

ENV USERNAME=docker

# Include a custom bashrc
COPY bashrc /usr/etc/skel/bashrc-dev

COPY setup.sh /usr/sbin/setup.sh
COPY entrypoint.sh /usr/sbin/entrypoint.sh
RUN chmod 755 /usr/sbin/setup.sh
RUN chmod 755 /usr/sbin/entrypoint.sh
ENTRYPOINT ["/usr/sbin/entrypoint.sh"]
CMD ["bash"]
