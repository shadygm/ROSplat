#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
    echo "Usage: $0 [-h] [-b | -n] [-s | -u | -r] [-c]"
    echo " -h    Show this help message"
    echo " -b    Build with cache"
    echo " -n    Build without cache"
    echo " -s    Start or reuse the container in the background"
    echo " -u    Start docker-compose and enter container"
    echo " -r    Start docker-compose and run ROSplat"
    echo " -c    Stop and clean up"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

BUILD=false
BUILD_ARGS=()
START_ONLY=false
OPEN_SHELL=false
RUN_APP=false
CLEAN=false

while getopts "hbnsurc" opt; do
    case ${opt} in
        h ) usage; exit 0 ;;
        b ) BUILD=true ;;
        n ) BUILD=true; BUILD_ARGS+=(--no-cache) ;;
        s ) START_ONLY=true ;;
        u ) OPEN_SHELL=true ;;
        r ) RUN_APP=true ;;
        c ) CLEAN=true ;;
        * ) usage; exit 1 ;;
    esac
done

action_count=0
[[ "${START_ONLY}" == true ]] && ((action_count += 1))
[[ "${OPEN_SHELL}" == true ]] && ((action_count += 1))
[[ "${RUN_APP}" == true ]] && ((action_count += 1))
if ((action_count > 1)); then
    echo "Choose only one of -s, -u, or -r." >&2
    exit 2
fi

export USER_UID="$(id -u)"
export USER_GID="$(id -g)"
export DISPLAY="${DISPLAY:-:0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

if [[ "${CLEAN}" == true ]]; then
    docker compose down --remove-orphans
fi

if [[ "${BUILD}" == true ]]; then
    echo "Building ROSplat with ROS 2 Lyrical and Spirula Vulkan..."
    DOCKER_BUILDKIT=1 docker compose build "${BUILD_ARGS[@]}"
fi

if [[ "${START_ONLY}" == true || "${OPEN_SHELL}" == true || "${RUN_APP}" == true ]]; then
    echo "Starting or reusing the ROSplat container..."
    mkdir -p ../data
    docker compose up -d
fi

if [[ "${START_ONLY}" == true ]]; then
    echo "ROSplat container is running in the background."
fi

if [[ "${OPEN_SHELL}" == true ]]; then
    docker compose exec rosplat rosplat-entrypoint bash
fi

if [[ "${RUN_APP}" == true ]]; then
    docker compose exec rosplat rosplat-entrypoint python3 -m rosplat.main
fi
