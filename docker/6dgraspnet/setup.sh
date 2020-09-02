#!/bin/bash
set -e

# These variables can be overridden by docker environment variables
USER_UID=${USER_UID:-1000}
USER_GID=${USER_GID:-1000}
USERNAME=${USERNAME:-docker}
USER_HOME=${USER_HOME:-/home/$USERNAME}

# Get the uid from pwd if it is not owned by root
if [[ $USER_UID = "1000" && $(stat -c "%u" $(pwd)) -ne 0 ]] ; then
    USER_UID=$(stat -c "%u" $(pwd))
fi

# Get the gid from pwd if it is not owned by root
if [[ $USER_GID = "1000" && $(stat -c "%g" $(pwd)) -ne 0 ]] ; then
    USER_GID=$(stat -c "%g" $(pwd))
fi

create_user() {
    # If the home folder exists, set a flag.
    # Creating the user during container initialization often is anticipated
    # by the mount of a docker volume. In this case the home directory is already
    # present in the file system and adduser skips by default the copy of the
    # configuration files.
    HOME_FOLDER_EXISTS=0
    if [ -d $USER_HOME ] ; then HOME_FOLDER_EXISTS=1 ; fi

    # Create a group with USER_GID
    if ! getent group ${USERNAME} >/dev/null; then
        echo "Creating ${USERNAME} group"
        groupadd -f -g ${USER_GID} ${USERNAME} 2> /dev/null
    fi

    # Create a user with USER_UID
    if ! getent passwd ${USERNAME} >/dev/null; then
        echo "Creating ${USERNAME} user"
        adduser --quiet \
                --disabled-login \
                --home ${USER_HOME} \
                --uid ${USER_UID} \
                --gid ${USER_GID} \
                --gecos 'Workspace' \
                ${USERNAME}
    fi

    # The home must belong to the user
    chown ${USER_UID}:${USER_GID} ${USER_HOME}

    # If configuration files have not been copied, do it manually
    if [ ${HOME_FOLDER_EXISTS} -ne 0 ] ; then

        for file in .bashrc .bash_logout .profile ; do
            if [[ ! -f ${USER_HOME}/${file} ]] ; then
                install -m 644 -g ${USERNAME} -o ${USERNAME} /etc/skel/${file} ${USER_HOME}
            fi
        done
    fi
}

# Create the user if run -u is not passed
if [[ $(id -u) -eq 0 && $(id -g) -eq 0 ]] ; then
    echo "==> Creating the runtime user"
    create_user

    # Set a default root password
    echo "==> Setting the default root password"
    ROOT_PASSWORD="root"
    echo "root:${ROOT_PASSWORD}" | chpasswd

    # Set a default password
    echo "==> Setting the default user password"
    USER_PASSWORD=${USERNAME}
    echo "${USERNAME}:${USER_PASSWORD}" | chpasswd
    echo "${USERNAME}    ALL=(ALL:ALL) ALL" >> /etc/sudoers

    # Add the user to video group for HW acceleration (Intel GPUs)
    usermod -aG video ${USERNAME}

    # Assign the user to the runtimeusers group
    gpasswd -a ${USERNAME} runtimeusers
fi

# Setup the custom bashrc
echo "==> Including additional bashrc configurations"
cp /usr/etc/skel/bashrc-dev /home/$USERNAME/.bashrc-dev
chown ${USERNAME}:${USERNAME} /home/$USERNAME/.bashrc-dev
echo "source /home/$USERNAME/.bashrc-dev" >> /home/${USERNAME}/.bashrc
echo "source /home/$USERNAME/.bashrc-dev" >> /root/.bashrc
