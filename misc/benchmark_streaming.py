#!/usr/bin/env python3
"""Benchmark the existing ROS GaussianArray streaming path phase by phase."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from gaussian_interface.msg import GaussianArray
from rclpy.serialization import deserialize_message, serialize_message

from misc.generate_gaussian_bag import make_gaussian_batch
from rosplat.config.world_settings import WorldSettings
from rosplat.core.gaussian_representation import (
    GaussianData,
    combine_gaussians,
    from_ply,
)
from rosplat.render.camera import Camera
from rosplat.render.renderer import SpirulaRenderer


def _slice(data: GaussianData, count: int) -> GaussianData:
    return GaussianData(
        data.xyz[:count],
        data.rot[:count],
        data.scale[:count],
        data.opacity[:count],
        data.sh[:count],
    )


def _elapsed(callable_):
    started = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - started


def _mib(byte_count: int | float) -> float:
    return byte_count / (1024 * 1024)


def run_benchmark(
    ply_path: Path,
    *,
    limit: int,
    batch_size: int,
    width: int,
    height: int,
    render_samples: int,
) -> dict:
    source, ply_seconds = _elapsed(lambda: from_ply(ply_path))
    count = min(limit, len(source)) if limit else len(source)
    source = _slice(source, count)
    bytes_per_splat = (11 + source.sh_dim) * 4

    camera = Camera(height, width)
    renderer, renderer_init_seconds = _elapsed(
        lambda: SpirulaRenderer(
            width,
            height,
            SimpleNamespace(world_camera=camera),
        )
    )

    timings = {
        "message_build": 0.0,
        "cdr_serialize": 0.0,
        "cdr_deserialize": 0.0,
        "message_to_numpy": 0.0,
        "spirula_append": 0.0,
        "legacy_host_concat": 0.0,
    }
    wire_bytes = 0
    legacy_scene = None
    legacy_cumulative_copy_bytes = 0
    geometric_growth_copy_bytes = 0
    previous_capacity = renderer.capacity

    try:
        for batch_number, start in enumerate(range(0, count, batch_size)):
            end = min(start + batch_size, count)
            message, elapsed = _elapsed(
                lambda start=start, end=end: make_gaussian_batch(
                    source,
                    start,
                    end,
                    refresh=start == 0,
                )
            )
            timings["message_build"] += elapsed

            payload, elapsed = _elapsed(lambda: serialize_message(message))
            timings["cdr_serialize"] += elapsed
            wire_bytes += len(payload)

            received, elapsed = _elapsed(
                lambda: deserialize_message(payload, GaussianArray)
            )
            timings["cdr_deserialize"] += elapsed

            batch, elapsed = _elapsed(
                lambda: WorldSettings.convert_gaussian_array(received)
            )
            timings["message_to_numpy"] += elapsed
            if batch is None:
                raise RuntimeError("benchmark generated an empty batch")

            old_count = renderer.splat_count
            _, elapsed = _elapsed(
                lambda: renderer.append_gaussians(
                    batch,
                    refresh=batch_number == 0,
                )
            )
            timings["spirula_append"] += elapsed
            current_capacity = renderer.capacity
            if current_capacity != previous_capacity and old_count:
                geometric_growth_copy_bytes += old_count * bytes_per_splat
            previous_capacity = current_capacity

            legacy_scene, elapsed = _elapsed(
                lambda: combine_gaussians([legacy_scene, batch])
            )
            timings["legacy_host_concat"] += elapsed
            if batch_number:
                legacy_cumulative_copy_bytes += len(legacy_scene) * bytes_per_splat

        rgba, first_render_seconds = _elapsed(renderer.render_rgba8)
        dirty_render_seconds = []
        for _ in range(render_samples):
            renderer.update_camera_pose()
            _, elapsed = _elapsed(renderer.render_rgba8)
            dirty_render_seconds.append(elapsed)
        raw_bytes = sum(
            array.nbytes
            for array in (
                source.xyz,
                source.rot,
                source.scale,
                source.opacity,
                source.sh,
            )
        )
        processing_seconds = sum(
            timings[name]
            for name in (
                "message_build",
                "cdr_serialize",
                "cdr_deserialize",
                "message_to_numpy",
                "spirula_append",
            )
        )
        result = {
            "input": {
                "path": str(ply_path),
                "splats": count,
                "batch_size": batch_size,
                "batches": (count + batch_size - 1) // batch_size,
                "sh_floats_per_splat": source.sh_dim,
                "raw_soa_mib": _mib(raw_bytes),
                "cdr_wire_mib": _mib(wire_bytes),
            },
            "seconds": {
                "ply_load": ply_seconds,
                "renderer_init": renderer_init_seconds,
                **timings,
                "stream_processing_sum": processing_seconds,
                "first_render": first_render_seconds,
                "dirty_render_mean": statistics.fmean(dirty_render_seconds)
                if dirty_render_seconds
                else None,
                "dirty_render_median": statistics.median(dirty_render_seconds)
                if dirty_render_seconds
                else None,
                "dirty_render_min": min(dirty_render_seconds)
                if dirty_render_seconds
                else None,
                "dirty_render_max": max(dirty_render_seconds)
                if dirty_render_seconds
                else None,
            },
            "throughput_splats_per_second": count / processing_seconds,
            "copies": {
                "legacy_host_concat_mib": _mib(legacy_cumulative_copy_bytes),
                "legacy_host_plus_cuda_projection_mib": _mib(
                    legacy_cumulative_copy_bytes * 2
                ),
                "new_geometric_device_growth_mib": _mib(
                    geometric_growth_copy_bytes
                ),
            },
            "renderer": {
                "splats": renderer.splat_count,
                "capacity": renderer.capacity,
                "render_shape": list(rgba.shape),
                "nonblack_rgb_pixels": int(np.count_nonzero(np.any(rgba[..., :3], axis=2))),
            },
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
        }
        return result
    finally:
        renderer.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="Maximum splats to process; 0 means the complete input.",
    )
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument(
        "--render-samples",
        type=int,
        default=5,
        help="Number of additional dirty-camera frames to time after the first render.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.render_samples < 0:
        parser.error("--render-samples cannot be negative")

    result = run_benchmark(
        args.ply_path,
        limit=args.limit,
        batch_size=args.batch_size,
        width=args.width,
        height=args.height,
        render_samples=args.render_samples,
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
