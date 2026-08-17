#!/usr/bin/env bash
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /opt/rosplat-msgs/setup.bash

exec "$@"
