#!/bin/bash
set -eu -o pipefail

# Move Atom packages to the user's home
# This command should work even if ~/.atom is mounted as volume from the host,
# and it should comply the presence of an existing ~/.atom/packages/ folder
COPY_ATOM_PACKAGES=${COPY_ATOM_PACKAGES:-0}
if [[ ${COPY_ATOM_PACKAGES} -eq 1 && -d "/root/.atom" ]] ; then
	echo "==> Setting up Atom packages into $USERNAME's home ..."
	if [ -d "/home/$USERNAME/.atom_packages_from_root" ] ; then
		rm -r "/home/$USERNAME/.atom_packages_from_root"
	fi
	mv /root/.atom /home/$USERNAME/.atom_packages_from_root
	chown -R $USERNAME:$USERNAME /home/$USERNAME/.atom_packages_from_root
	declare -a ATOM_PACKAGES
	ATOM_PACKAGES=($(find /home/$USERNAME/.atom_packages_from_root/packages -mindepth 1 -maxdepth 1 -type d))
	for package in "${ATOM_PACKAGES[@]}" ; do
		if [ ! -e /home/$USERNAME/.atom/packages/"$(basename $package)" ] ; then
			cd $package
			su -c "apm link" $USERNAME
		fi
	done
	cd /
	echo "... Done"
fi



