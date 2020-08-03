ARG from=ubuntu:bionic
FROM ${from}

# Change default shell to bash
SHELL ["/bin/bash", "-c"]

# Install packages
# ================
RUN apt-get update &&\
    apt-get install -y --no-install-recommends --fix-missing\
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
        mesa-common-dev \
        apt-transport-https \
        apt-utils \
        dbus-x11 \
        bash-completion \
        gdb \
        ssh \
        iputils-ping \
        gnupg \
        gnupg2 \
        xvfb \
        mercurial \
        freeglut3-dev \
        libflann-dev \
        &&\
    rm -rf /var/lib/apt/lists/*

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ `lsb_release -cs` main" &&\
    add-apt-repository “deb http://security.ubuntu.com/ubuntu xenial-security main” &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends --fix-missing\
        gcc-8 \
        g++-8 \
        cmake \
        cmake-curses-gui \
        mc \
        lynx \
        libqhull* \
        pkg-config \
        libxmu-dev \
        libxi-dev \
        libboost-all-dev \
        libeigen3-dev \
        libusb-1.0-0-dev \
        libudev-dev \
        doxygen \
        graphviz \
        libgtest-dev \
        libpcap-dev \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libjpeg.dev \
        libtiff4.dev \
        libswscale-dev \
        libjasper-dev \
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


# Install python3
# ===============
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

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 \
                --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 &&\
    echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -cs` main" \
        > /etc/apt/sources.list.d/ros-latest.list &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends \
        # ros-${ROS_DISTRO}-desktop \
        ros-${ROS_DISTRO}-desktop-full \
        #ros-${ROS_DISTRO}-fake-localization \
        #ros-${ROS_DISTRO}-map-server \
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

# Set workspace directories
# =========================
ENV WS_DIR=/workspace
ENV CATKIN_WS_DIR=${WS_DIR}/catkin_ws
ENV WS_INSTALL=${WS_DIR}/install
ENV WS_SOURCES=${WS_DIR}/sources
ARG WS_BIN=${WS_SOURCES}/bin

# Create a new runtimeusers group
RUN groupadd -K GID_MIN=900 -K GID_MAX=1100 runtimeusers

# Install Eigen
# =============
RUN mkdir -p ${WS_SOURCES} &&\
    cd ${WS_SOURCES} &&\
    && git clone -b tags/3.2.0 https://github.com/eigenteam/eigen-git-mirror eigen \
    && cd eigen \
    && mkdir build \
    && cd build \
    && cmake  -D CMAKE_INSTALL_PREFIX=${WS_INSTALL} .. \
    && make -j8 \
    && make install \
    && find ${WS_SOURCES} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+ && \
    && find ${WS_INSTALL} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+

# Install VTK
# ============
RUN apt-get update &&\
    apt-get install -y --no-install-recommends\
    libvtk7-* &&\
    rm -rf /var/lib/apt/lists/*

# Install PCL
# ===========
RUN cd ${WS_SOURCES}\
    && wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.9.0.zip \
    && unzip pcl-1.9.0.zip \
    && cd pcl-pcl-1.9.0 \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=None -D BUILD_GPU=ON -D BUILD_apps=ON -D BUILD_examples=ON  -DCMAKE_INSTALL_PREFIX=${WS_INSTALL}.. \
    && make -j 8 \
    && make install \
    && find ${WS_SOURCES} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+ && \
    && find ${WS_INSTALL} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+

# Install Opencv
# ==============
RUN cd ${WS_SOURCES}\
    && wget https://github.com/opencv/opencv/archive/3.4.3.zip \
    && unzip 3.4.3.zip \
    && cd opencv-3.4.3 \
    && mkdir build \
    && cd build \
    && cmake -D WITH_OPENMP=ON -D ENABLE_PRECOMPILED_HEADERS=OFF  -DCMAKE_INSTALL_PREFIX=${WS_INSTALL} ..\
    && make -j 8 \
    && make install \
    && find ${WS_SOURCES} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+ && \
    && find ${WS_INSTALL} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+

# Install CAFFE
# =============
RUN apt-get update &&\
    apt-get install -y \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    libopencv-dev \
    libhdf5-serial-dev \
    protobuf-compiler \
    libboost-all-dev \
    libatlas-base-dev \
    liblmdb-dev \
    libturbojpeg \
    libturbojpeg0-dev \
    libgoogle-glog-dev &&\
    rm -rf /var/lib/apt/lists/* &&\
    cd ${WS_SOURCES}\
    && git clone https://github.com/fbottarel/caffe.git &&\
    cd ${WS_SOURCES}/caffe \
    && mkdir build \
    && cd build \
    && cmake -D BUILD_python=OFF \
        -D BUILD_python_layer=OFF \
        -D BLAS=Atlas \
        -D CUDA_rt_LIBRARY=/usr/local/cuda/lib64/libcudart.so \
        -D CMAKE_INSTALL_PREFIX=${WS_INSTALL} \
        ../ \
    && make -j 8 \
    && make install \
    && find ${WS_SOURCES} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+ && \
    && find ${WS_INSTALL} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+

# Install gpd
# ===========
RUN cd ${WS_SOURCES}\
    && git clone https://github.com/atenpas/gpd gpd \
    && cd gpd \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=${WS_INSTALL}\
        -D USE_CAFFE=ON \
        .. \
    && make -j8  \
    && find ${WS_SOURCES} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+ && \
    && find ${WS_INSTALL} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+

# Create catkin_ws with dependencies/ros packages for python3
# ===========================================================
RUN mkdir -p ${CATKIN_WS_DIR}/src && \
    cd ${CATKIN_WS_DIR} &&\
    wstool init &&\
    wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5 &&\
    wstool set -y src/vision_opencv --git https://github.com/ros-perception/vision_opencv -v 1.13.0 &&\
    wstool up &&\
    rosdep update && rosdep install --from-paths src --ignore-src -y -r &&\
    rm -rf /var/lib/apt/lists/* &&\
    catkin config --extend /opt/ros/${ROS_DISTRO} -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release && \
    catkin build  -j8 && \
    echo 'source ${CATKIN_WS_DIR}/devel/setup.bash' >> /etc/bash.bashrc &&\
    find ${CATKIN_WS_DIR} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+

# Install benchmark-grasping
# ==========================
RUN cd ${CATKIN_WS_DIR} &&\
    wstool set -y src/grasping-benchmarks-panda --git https://github.com/robotology/grasping-benchmarks-panda &&\
    wstool up &&\
    cd src/grasping-benchmarks-panda &&\
    pip3 install -e grasping_benchmarks &&\
    cd ${CATKIN_WS_DIR} &&\
    catkin config --extend /opt/ros/${ROS_DISTRO} -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release -DGPD_ROS=ON && \
    catkin build  -j8 && \
    find ${CATKIN_WS_DIR} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+

# =============
# FIX OWNERSHIP
# =============
RUN find ${WS_DIR} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+ &&\
    find ${WS_INSTALL} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+

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
