# ====================================
# Run command. Verify if the xauth file exists or if a container with the same
# name is already running.
# ====================================

if [ "$#" -lt 2 ]; then
    echo "Illegal number of parameters. Usage: run.sh <username> <container-id> [image-name]"
    echo "Example: run.sh fbottarel ros-container"
    exit 1
fi

USERNAME=${1:-"erampone"}
CONTAINERNAME=${2:-"ros_sq"}
IMAGENAME=${3:-"eleramp/superquadric_tools_panda:latest"}
XSOCK="/tmp/.X11-unix"
XAUTH="/tmp/.$CONTAINERNAME.xauth"

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
# Alternative way to create the xauth file
# ====================================

# if [ ! -f $XAUTH ]
# then
#     xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
#     if [ ! -z "$xauth_list" ]
#     then
#         echo $xauth_list | xauth -f $XAUTH nmerge -
#     else
#         touch $XAUTH
#     fi
#     chmod a+r $XAUTH
# fi

# ====================================
# Add --network=host and --privileged if connecting to other ROS nodes
# Add --volume=<host-volume>:<mount-point> for sharing the host filesystem
# ====================================

if [ ! "$(docker ps -a | grep $CONTAINERNAME)" ]
then
    docker run \
        -it \
        --name=$CONTAINERNAME \
        -e DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        -e USER_UID=1000 \
        -e USER_GID=1000 \
        -e USERNAME=$USERNAME \
        -e XAUTHORITY=$XAUTH \
        --volume=$XSOCK:$XSOCK:rw \
        --volume=$XAUTH:$XAUTH:rw \
        --device /dev/dri \
        --gpus=all \
        --volume=$HOME/workspace:/home/$USERNAME/workspace \
        $IMAGENAME
else
    docker start $CONTAINERNAME > /dev/null
    docker exec -it -u $1 $2 bash
fi

# ====================================
# Once the script is running:
# ====================================
# docker exec -it -u <user> ros-container bash
