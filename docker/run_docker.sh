#!/bin/bash

usage() {
    echo "Usage: $0 [-h] [-b] [-n] [-u] [-c] [-j | -f]"
    echo " -h    Show this help message"
    echo " -b    Build with cache"
    echo " -n    Build without cache"
    echo " -u    Start docker-compose and enter container"
    echo " -c    Stop and clean up"
    echo " -j    Use ROS 2 Jazzy (default)"
    echo " -f    Use ROS 2 Foxy"
    echo " -l    Use ROS 2 Humble"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

# Default to Jazzy
DOCKERFILE="docker/Dockerfile.jazzy"
BUILD=false
BUILD_ARGS=""
COMPOSEUP=false

while getopts "hbnucjfl" opt; do
    case ${opt} in
        h ) usage; exit 0 ;;
        b ) BUILD=true ;;
        n ) BUILD=true; BUILD_ARGS="--no-cache" ;;
        u ) COMPOSEUP=true ;;
        c ) docker compose down --remove-orphans; exit 0 ;;
        j ) DOCKERFILE="docker/Dockerfile.jazzy" ;;
        f ) DOCKERFILE="docker/Dockerfile.foxy" ;;
        l ) DOCKERFILE="docker/Dockerfile.humble" ;;
        * ) usage; exit 1 ;;
    esac
done

export USER_UID=$(id -u)
export USER_GID=$(id -g)
export USERNAME=$(id -un)
export USER_PASSWORD=${USERNAME}
export HOSTNAME=$(hostname)
export HOME=$HOME
export DISPLAY=$DISPLAY
export XAUTHORITY=$XAUTHORITY
export SSH_AUTH_SOCK=$SSH_AUTH_SOCK
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
export DOCKERFILE  # Needed by docker-compose

if [ "$BUILD" = true ]; then
    echo -e "Building docker image with \033[0;31m$DOCKERFILE\033[0m..."
    DOCKER_BUILDKIT=1 docker compose build $BUILD_ARGS
    if [ "$?" -ne 0 ]; then
        echo "Docker build failed!"
        exit 1
    fi
fi

if [ "$COMPOSEUP" = true ]; then
    echo "Starting docker container..."
    mkdir -p ../data
    docker compose up -d
    if [ "$?" -ne 0 ]; then
        echo "Docker compose up failed!"
        exit 1
    fi
    docker compose exec rosplat bash
fi
