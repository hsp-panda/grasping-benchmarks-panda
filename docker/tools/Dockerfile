ARG from=ubuntu:bionic
FROM ${from}

# Install packages
# ================
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        wget \
        curl \
        unzip \
        apt-transport-https \
        &&\
    rm -rf /var/lib/apt/lists/*

# QtCreator
# =========
ARG QTCREATOR_VERSION=4.11.1
COPY QtCreatorSetup.js /tmp/QtCreatorSetup.js
COPY qtaccount.ini /root/.local/share/Qt/qtaccount.ini
RUN cd /tmp &&\
    wget http://download.qt.io/official_releases/qtcreator/${QTCREATOR_VERSION%.*}/${QTCREATOR_VERSION}/qt-creator-opensource-linux-x86_64-${QTCREATOR_VERSION}.run &&\
    chmod +x qt-creator-opensource-linux-x86_64-${QTCREATOR_VERSION}.run &&\
    ./qt-creator-opensource-linux-x86_64-${QTCREATOR_VERSION}.run --platform minimal --script QtCreatorSetup.js &&\
    rm /tmp/qt-creator-opensource-linux-x86_64-${QTCREATOR_VERSION}.run /tmp/QtCreatorSetup.js &&\
    ln -s /opt/qtcreator/bin/qtcreator.sh /usr/bin/qtcreator &&\
    rm /root/.local/share/Qt/qtaccount.ini

# Atom
# ====
ARG ATOM_VERSION=v1.40.1
ARG ATOM_PKG_TMP=/tmp/atom_packages.txt

RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        gconf2 \
        libxtst6 \
        gvfs-bin \
        xdg-utils \
        policykit-1 \
        &&\
    rm -rf /var/lib/apt/lists/* &&\
    curl -L https://github.com/atom/atom/releases/download/${ATOM_VERSION}/atom-amd64.deb > /tmp/atom.deb && \
    dpkg -i /tmp/atom.deb && \
    rm -f /tmp/atom.deb && \
    echo "tool-bar" >> ${ATOM_PKG_TMP} &&\
    echo "indent-detective" >> ${ATOM_PKG_TMP} &&\
    echo "latex-completions" >> ${ATOM_PKG_TMP} &&\
    echo "hyperclick" >> ${ATOM_PKG_TMP} &&\
    apm install --packages-file ${ATOM_PKG_TMP} &&\
    cp -r /root/.atom /opt/dotatom &&\
    find /opt/dotatom -not -group runtimeusers -exec chgrp runtimeusers {} \; -exec chmod g+rw {} \+


# Visual studio code
# ==================
RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg &&\
    install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/ &&\
    sh -c 'echo "deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list' &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends code &&\
    rm -rf /var/lib/apt/lists/*


ENV QT_X11_NO_MITSHM=1