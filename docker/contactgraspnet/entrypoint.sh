#!/bin/bash
set -eu -o pipefail

if [ ! -x "$(which setup.sh)" ] ; then
    echo "==> File setup.sh not found."
    exit 1
fi

# Initialize the container
echo "==> Configuring Contact-GraspNet benchmarking image"
setup.sh

echo "==> Contact-GraspNet benchmarking container ready"

# If a CMD is passed, execute it
echo "== execute command $@"
gosu $USERNAME "$@"
