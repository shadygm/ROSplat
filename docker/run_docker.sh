#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "Usage: $0 [-h] [-b | -n] [-u | -r] [-c]"
    echo " -h    Show this help message"
    echo " -b    Build with cache"
    echo " -n    Build without cache"
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
OPEN_SHELL=false
RUN_APP=false
CLEAN=false

while getopts "hbnurc" opt; do
    case ${opt} in
        h ) usage; exit 0 ;;
        b ) BUILD=true ;;
        n ) BUILD=true; BUILD_ARGS+=(--no-cache) ;;
        u ) OPEN_SHELL=true ;;
        r ) RUN_APP=true ;;
        c ) CLEAN=true ;;
        * ) usage; exit 1 ;;
    esac
done

if [[ "${OPEN_SHELL}" == true && "${RUN_APP}" == true ]]; then
    echo "Choose either -u or -r, not both." >&2
    exit 2
fi

export USER_UID="$(id -u)"
export USER_GID="$(id -g)"
export DISPLAY="${DISPLAY:-:0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

detect_cuda_architectures() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi

    nvidia-smi \
        --query-gpu=compute_cap \
        --format=csv,noheader,nounits 2>/dev/null \
        | sed 's/[[:space:]]//g' \
        | awk '/^[0-9]+\.[0-9]+$/' \
        | sort -Vu \
        | paste -sd ';' -
}

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="$(detect_cuda_architectures)"
fi

if [[ -n "${TORCH_CUDA_ARCH_LIST}" ]]; then
    echo "Using CUDA compute architecture(s): ${TORCH_CUDA_ARCH_LIST}"
elif [[ "${OPEN_SHELL}" == true || "${RUN_APP}" == true ]]; then
    echo "Could not detect a supported NVIDIA GPU compute capability." >&2
    echo "Set TORCH_CUDA_ARCH_LIST explicitly to override detection." >&2
    exit 3
fi

if [[ "${CLEAN}" == true ]]; then
    docker compose down --remove-orphans
fi

if [[ "${BUILD}" == true ]]; then
    echo "Building ROSplat with ROS 2 Lyrical and CUDA 13.3..."
    DOCKER_BUILDKIT=1 docker compose build "${BUILD_ARGS[@]}"
fi

if [[ "${OPEN_SHELL}" == true || "${RUN_APP}" == true ]]; then
    echo "Starting docker container..."
    mkdir -p ../data
    docker compose up -d
fi

if [[ "${OPEN_SHELL}" == true ]]; then
    docker compose exec rosplat rosplat-entrypoint bash
fi

if [[ "${RUN_APP}" == true ]]; then
    docker compose exec rosplat rosplat-entrypoint python3 -m rosplat.main
fi
