#!/bin/bash
set -eu -o pipefail

if [ ! -x "$(which setup.sh)" ] ; then
    echo "==> File setup.sh not found."
    exit 1
fi

# Initialize the container
echo "==> Configuring 6DoFGraspNet benchmarking image"
setup.sh
echo "==> 6DoFGraspNet benchmarking container ready"

# If a CMD is passed, execute it
echo "== execute command $@"
gosu $USERNAME "$@"