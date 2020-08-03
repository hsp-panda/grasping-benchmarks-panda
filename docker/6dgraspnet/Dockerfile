# per il momento
# assicurasi di avere python 2.7
# installare
# sudo apt-get install -y libspatialindex-dev per requirements.txt
# poi pyrender version richiesta nei requirements.txt si scarica versione imageio non compatibile con python2.7. In master di pyrender hanno risolto questo bug. ho provto quindi a usare pyrender master
# necessita di qt4
# sudo apt-get install -y qt4-default
# compile with CC=gcc-7 CXX=g++-7
# --------------------------------------------------------#
ARG from=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
FROM ${from}

# Create a new runtimeusers group (it might be already present)
RUN groupadd runtimeusers || true

ENV CC="gcc-8"
ENV CXX="g++-8"

RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        git \
        cmake \
        cmake-curses-gui \
        build-essential \
        gcc-8 \
        g++-8 \
        python-pip \
        python-setuptools \
        libpython-dev \
        python-dev \
        python-wheel \
        libspatialindex-dev \
        qt4-default \
        &&\
        rm -rf /var/lib/apt/lists/*

# Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs &&\
    git lfs install \
    &&\
    rm -rf /var/lib/apt/lists/*

# Setup locales and tzdata
RUN sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/g' /etc/locale.gen &&\
    locale-gen &&\
    update-locale LANG="en_US.UTF-8" &&\
    export DEBIAN_FRONTEND=noninteractive &&\
    export DEBCONF_NONINTERACTIVE_SEEN=true &&\
    rm -rf /etc/localtime &&\
    rm -rf /etc/timezone &&\
    echo 'tzdata tzdata/Areas select Europe' | debconf-set-selections &&\
    echo 'tzdata tzdata/Zones/Europe select Rome' | debconf-set-selections &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends tzdata &&\
    rm -rf /var/lib/apt/lists/*

# ===========================
# Setup the python virtualenv
# ===========================
ENV WS_DIR=/workspace
ENV VIRTUAL_ENV=${WS_DIR}/ve
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
RUN pip install virtualenv

# ===================
# Install 6D GraspNet
# ===================

# tensorflow 1.12
RUN mkdir ${WS_DIR} && virtualenv ${VIRTUAL_ENV} &&\
    wget https://storage.googleapis.com/tf-performance/tf_binary/tensorflow-1.12.0.a6d8ffa.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl && \
    ${VIRTUAL_ENV}/bin/pip install tensorflow-1.12.0.a6d8ffa.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl && \
    rm tensorflow-1.12.0.a6d8ffa.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl


RUN cd ${WS_DIR} &&\
    git clone https://github.com/NvLabs/6dof-graspnet &&\
    git clone https://github.com/charlesq34/pointnet2 &&\
    cd 6dof-graspnet && \
    # workaround needed to install imageio version compatible with python 2.7
    ${VIRTUAL_ENV}/bin/pip install vtk &&\
    sed -i '/pyrender/d' requirements.txt &&\
    sed -i '/mayavi/d' requirements.txt &&\
    echo -e "pyglet==1.4.10\npyrender==0.1.43\npyface==6.1.2\ntraitsui==6.1.2\nmayavi==4.7.0" >> requirements.txt &&\
    ${VIRTUAL_ENV}/bin/pip install -r requirements.txt && \
    find ${WS_DIR} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+ &&\
    find ${VIRTUAL_ENV} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+

# download pre-trained models
RUN mkdir ${WS_DIR}/6dof-graspnet/checkpoints && cd ${WS_DIR}/6dof-graspnet/checkpoints &&\
    ${VIRTUAL_ENV}/bin/pip install gdown &&\
    mkdir -p latent_size_2_ngpus_1_gan_1_confidence_weight_0.1_npoints_1024_num_grasps_per_object_256_train_evaluator_0_/tf_output && \
    cd latent_size_2_ngpus_1_gan_1_confidence_weight_0.1_npoints_1024_num_grasps_per_object_256_train_evaluator_0_ &&\
    gdown https://drive.google.com/uc?id=1PC-FDD9gyGwgtEkEhQl9NDwhZ4sqauao &&\
    gdown https://drive.google.com/uc?id=1VkPcv6mvpIPf6rfSxGoJJBGrIm3ES4Ch && \
    cd tf_output &&\
    gdown https://drive.google.com/uc?id=19GhsCIWx2jZ8MeF8clqBC3MFtlZtzrwd &&\
    gdown https://drive.google.com/uc?id=1MqCBN17pkly-d69brzYybLbBd3zfgu0t &&\
    gdown https://drive.google.com/uc?id=1uUYd1RoEgzsJE0FCesXFEVnsNMlUdJPU &&\
    gdown https://drive.google.com/uc?id=1_kZq_yw44OhMeVlFEVgwsTdsjJIHtkUj &&\
    mkdir -p ${WS_DIR}/6dof-graspnet/checkpoints/npoints_1024_train_evaluator_0_allowed_categories__ngpus_1_/tf_output && \
    cd  ${WS_DIR}/6dof-graspnet/checkpoints/npoints_1024_train_evaluator_0_allowed_categories__ngpus_1_ &&\
    gdown https://drive.google.com/uc?id=1nbP4eYL3jKUh7CAQATk3rV-BDoOZ3_Pa &&\
    cd tf_output &&\
    gdown https://drive.google.com/uc?id=1D4JmCSFgUv6TyIOHazgNt-YE18VoTHal &&\
    gdown https://drive.google.com/uc?id=1ULJbs6UWoIxZ9Q5p0vqgfIZ61TreU9nX &&\
    gdown https://drive.google.com/uc?id=1AAF5C3Aoy60KlWcMqKV1r35oZeU3vQT- &&\
    gdown https://drive.google.com/uc?id=1xfEXEdIjJPQ31u885oRCuHy-rNH1KjZx &&\
    gdown https://drive.google.com/uc?id=1OrRG1lGelECLaY7uJkgDyTp5b_yl05DV &&\
    gdown https://drive.google.com/uc?id=11KE6pblIOVGJ0mgeaLagkM3y6goJiFBO &&\
    mkdir -p ${WS_DIR}/6dof-graspnet/checkpoints/npoints_1024_train_evaluator_1_allowed_categories__ngpus_8_/tf_output && \
    cd  ${WS_DIR}/6dof-graspnet/checkpoints/npoints_1024_train_evaluator_1_allowed_categories__ngpus_8_ &&\
    gdown https://drive.google.com/uc?id=1I7W-6h4niL6F2gOKPXV4j6UdFWmNmdcF &&\
    cd tf_output &&\
    gdown https://drive.google.com/uc?id=1bLN37AgGmJCjZ9RSVtUXdaCOsi1EzpsM &&\
    gdown https://drive.google.com/uc?id=18xC4bl2fwy5wy3qgog_fdAvFQQtSY_-k &&\
    gdown https://drive.google.com/uc?id=1hqRMdrR5Tdx1RDOXDQFJmzbAHE1fa7Aq &&\
    gdown https://drive.google.com/uc?id=1bYZfDT_OM7lBr6yl77oiSfbn_9QO19XW &&\
    find ${WS_DIR} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+ &&\
    find ${VIRTUAL_ENV} -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+


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
