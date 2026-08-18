#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if (($#)); then
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        echo "Usage: $0"
        echo "Start or reuse the ROSplat container and launch the ImGui UI."
        exit 0
    fi
    echo "Usage: $0" >&2
    exit 2
fi

# run_docker keeps the Compose service alive after the foreground UI exits.
exec "${SCRIPT_DIR}/run_docker.sh" -r
