#!/bin/bash
set -eu -o pipefail

if [ ! -x "$(which setup.sh)" ] ; then
    echo "==> File setup.sh not found."
    exit 1
fi

# Initialize the container
echo "==> Configuring 6DoFGraspNet benchmarking image"
setup.sh

# We need to compile additional tensorflow operators in this one
echo "==> Compiling custom TensorFlow operators"
CURR_DIR=$PWD
cd /workspace/sources/6dof-graspnet
sh compile_pointnet_tfops.sh
cd $CURR_DIR
echo "==> Custom TensorFlow operators ready"

echo "==> 6DoFGraspNet benchmarking container ready"

# If a CMD is passed, execute it
echo "== execute command $@"
gosu $USERNAME "$@"
