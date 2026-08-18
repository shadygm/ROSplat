#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath -- "${SCRIPT_DIR}/..")"

usage() {
    echo "Usage: $0 PLY_PATH [PUBLISHER_OPTIONS...]"
    echo
    echo "Examples:"
    echo "  $0 data/horse.ply"
    echo "  $0 data/horse.ply --batch-size 10000 --rate 10"
    echo "  $0 data/horse.ply --topic /my_gaussians"
}

if (($# == 0)) || [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ply_input="$1"
shift
if ! ply_path="$(realpath -e -- "${ply_input}")"; then
    echo "PLY file does not exist: ${ply_input}" >&2
    exit 2
fi
if [[ ! -f "${ply_path}" || "${ply_path,,}" != *.ply ]]; then
    echo "Expected a .ply file: ${ply_input}" >&2
    exit 2
fi
case "${ply_path}" in
    "${REPO_ROOT}"/*) ;;
    *)
        echo "PLY file must be inside the ROSplat repository: ${REPO_ROOT}" >&2
        exit 2
        ;;
esac

container_path="/workspace/${ply_path#"${REPO_ROOT}/"}"
"${SCRIPT_DIR}/run_docker.sh" -s

cd "${SCRIPT_DIR}"
echo "Publishing ${container_path}; press Ctrl+C to stop the publisher."
exec docker compose exec rosplat rosplat-entrypoint \
    python3 -m misc.generate_gaussian_bag \
    --ply-path "${container_path}" "$@"
