#!/usr/bin/env bash
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /opt/rosplat-msgs/setup.bash

rosplat_runtime_dir="${XDG_RUNTIME_DIR:-/tmp/rosplat-runtime-${UID}}"
mkdir -p "${rosplat_runtime_dir}"
chmod 0700 "${rosplat_runtime_dir}"
export XDG_RUNTIME_DIR="${rosplat_runtime_dir}"

exec "$@"
