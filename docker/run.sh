# ====================================
# Run command. Verify if the xauth file exists or if a container with the same
# name is already running.
# ====================================

if (( "$#" < 2 || "$#" > 3 ))
then
    echo "Illegal number of parameters. Usage: run.sh <username> <container-id> [image-name]"
    echo "Example: run.sh panda-user ros-container"
    exit 1
fi

# ====================================
# Specify some variables that are useful while setting up the container
# ====================================

USERNAME=${1:-${USER}}
CONTAINERNAME=${2:-"benchmark"}
IMAGENAME=${3:-"panda/ros:nvidia"}
XSOCK="/tmp/.X11-unix"
XAUTH="/tmp/.$CONTAINERNAME.xauth"
USER_UID=$UID
USER_GID=$UID

# ====================================
# With two arguments, the user wants an already existing container.
# With three, a new one must be spawned
# Create the proper container name according to whether it exists or not
# ====================================

if [ "$#" -eq 2 ]
then
    NEW_CONTAINER="false"
    if [ ! "$(docker ps -a | grep $CONTAINERNAME)" ]
    then
        echo "Container $CONTAINERNAME does not exist. No action performed."
        exit 1
    fi
else
    NEW_CONTAINER="true"
    if [ "$(docker ps -a | grep $CONTAINERNAME)" ]
    then
        TIMESTAMP=`date +"%y_%m_%d-%k_%M_%S"`
        echo "Container $CONTAINERNAME already exists. Adding timestamp $TIMESTAMP."
        CONTAINERNAME=$CONTAINERNAME-$TIMESTAMP
    fi
fi

echo "Running container $CONTAINERNAME as $USERNAME..."

# ====================================
# Create a Xauth file for each container if it does not already exist
# ====================================

if [ ! -f $XAUTH ]
then
    xauth nlist :0 | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
    chmod a+x $XAUTH
    echo "Created file Xauth file $XAUTH"
fi

# ====================================
# Spin up the container with the specified options (if it doesn't exist already)
#
# This should support both nvidia and intel graphics adapters
# A shared directory will be created in HOME/workspace/docker-shared-workspace/$CONTAINERNAME
#
# Add --network=host and --privileged if connecting to other ROS nodes
# Add --volume=<host-volume>:<mount-point> for sharing the host filesystem
# ====================================

if [ "$NEW_CONTAINER" == 'true' ]
then
    mkdir -p $HOME/workspace/docker-shared-workspace/$CONTAINERNAME
    docker run \
        -it \
        --name=$CONTAINERNAME \
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        -e USER_UID=$USER_UID \
        -e USER_GID=$USER_GID \
        -e USERNAME=$USERNAME \
        -e XAUTHORITY=$XAUTH \
        --volume=$XSOCK:$XSOCK:rw \
        --volume=$XAUTH:$XAUTH:rw \
        --device /dev/dri \
        --gpus=all \
        --network=host \
        --privileged \
        $IMAGENAME \
        bash
else
    docker start $CONTAINERNAME > /dev/null
    docker exec -it -u $USERNAME $CONTAINERNAME bash
fi