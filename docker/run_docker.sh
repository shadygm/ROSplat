#!/bin/bash

# Script to build and run rosplat with host UID/GID and hostname matching, without using a .env file.

usage() {
    echo "Usage: $0 [-h] [-b] [-n] [-u] [-c]"
    echo " -h    Show this help message"
    echo " -b    Build with cache"
    echo " -n    Build without cache"
    echo " -u    Start docker-compose and enter container"
    echo " -c    Stop and clean up"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

BUILD=false
BUILD_ARGS=""
COMPOSEUP=false

while getopts "hbnu" opt; do
    case ${opt} in
        h )
            usage
            exit 0
            ;;
        b )
            BUILD=true
            ;;
        n )
            BUILD=true
            BUILD_ARGS="--no-cache"
            ;;
        u )
            COMPOSEUP=true
            ;;
        c )
            docker compose down --remove-orphans
            exit 0
            ;;
        * )
            usage
            exit 1
            ;;
    esac
done

# Export host user info as environment variables
export USER_UID=$(id -u)
export USER_GID=$(id -g)
export USERNAME=$(id -un)
export USER_PASSWORD=${USERNAME}  # Using username as password
export HOSTNAME=$(hostname)
export HOME=$HOME
export DISPLAY=$DISPLAY
export XAUTHORITY=$XAUTHORITY
export SSH_AUTH_SOCK=$SSH_AUTH_SOCK
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0

if [ "$BUILD" = true ]; then
    echo "Building docker image..."
    docker compose build $BUILD_ARGS
fi

if [ "$COMPOSEUP" = true ]; then
    echo "Starting docker container..."
    docker compose up -d
    docker compose exec rosplat bash
fi
