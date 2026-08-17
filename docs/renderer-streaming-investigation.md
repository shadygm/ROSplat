# Renderer correctness and streaming performance investigation

## Outcome

The OpenGL and gsplat/CUDA paths have been replaced by one renderer based on
Spirula Studio's Vulkan backend. Dear ImGui is presented by wgpu-py's Vulkan
backend, so ROSplat no longer uses OpenGL for either splats or UI presentation.

The old OpenGL mismatch was not one isolated shader constant. Its default
ordering state was undefined, its ordering became stale as soon as the camera
moved, model transforms were applied inconsistently, and a global center-depth
sort could not reproduce gsplat's per-tile rasterization in difficult overlap
cases. Using the same tiled projection/sort/rasterization design removes that
class of divergence.

Streaming can still be improved substantially. The native Vulkan append is no
longer the main ingestion bottleneck: Python construction of nested ROS
messages and CDR deserialization now dominate. SOG is therefore not the next
change recommended for live ROS updates. A packed, versioned, struct-of-arrays
ROS chunk should come first. SOG or Streamed SOG remains a strong later option
for distributing static, precomputed scenes.

## Why the OpenGL output differed

The removed implementation can be inspected immediately before commit
`31b56bc` with, for example:

```bash
git show 31b56bc^:rosplat/render/renderer/OpenGLRenderer.py
git show 31b56bc^:rosplat/render/renderer/shaders/gau_vert.glsl
git show 31b56bc^:rosplat/config/world_settings.py
```

The concrete findings were:

| Finding | Rendering consequence |
| --- | --- |
| The vertex shader always indexed `gi[gl_InstanceID]`, but the index SSBO began unallocated and `auto_sort` defaulted to `False`. | Before the first manual sort, the shader read an unbound/undefined ordering buffer. This alone could produce missing, duplicated, or otherwise incorrect splats. |
| Sorting was optional and occurred only when scene state was uploaded. Camera pose changes updated a matrix but did not re-sort. | Alpha blending used an order belonging to an earlier view. The error became view-dependent and was most visible on overlapping transparent splats. |
| The model matrix transformed Gaussian centers, while covariance projection used the original quaternion/scale and only the view matrix. | Rotating or scaling the model moved centers without applying the same transform to the ellipses. |
| The SH direction used the original position (`g_pos - cam_pos`) rather than the model-transformed position. | View-dependent color was inconsistent with transformed geometry. |
| The implementation globally sorted Gaussian centers, then used conventional source-over blending. | A single global order cannot exactly reproduce a tiled renderer's pixel-local ordering for large, intersecting anisotropic splats. |
| GPU sorters returned the complete order to CPU and uploaded it to OpenGL again. | Every sort introduced an O(N log N) sort plus a full GPU-to-CPU-to-GPU index transfer. |
| A depth attachment existed, but the transparency pass did not explicitly own all depth-test/depth-write state. | Behavior could depend on state inherited from other UI/render work. This was a risk, although not required to explain the confirmed mismatch above. |

The replacement uses Spirula's projection, tile intersection, radix sort, and
rasterization on every dirty render. It also uses explicit OpenCV camera axes
and derives focal length from the vertical FOV and viewport height, including
non-square viewports.

## Current data path

```text
GaussianArray callback (ROS executor thread)
  -> nested message fields copied to one NumPy SoA batch
  -> batch queued without rebuilding the complete host scene
  -> GUI thread drains all available batches once per frame
  -> activated values converted to Spirula log-scale/logit layout
  -> synchronous H2D append into geometrically grown Vulkan buffers
  -> Spirula project / tile-sort / rasterize
  -> RGB float readback and RGBA8 conversion
  -> reusable WGPU Vulkan texture upload
  -> Dear ImGui Vulkan presentation
```

The renderer and ImGui currently own separate Vulkan devices/contexts. This is
Vulkan-only, but it is not zero-copy: each dirty frame reads Spirula's float
RGB image to host memory and uploads RGBA8 to the WGPU texture. At 1280x720,
that is about 10.55 MiB of RGB float readback plus 3.52 MiB of RGBA8 upload per
dirty frame.

## Measured performance

Measurements were made on 2026-08-17 using the checked-in 1,000,000-splat
`data/horse.ply` (SH degree 1), batches of 1,000, an NVIDIA GeForce RTX 3070
Laptop GPU with driver 610.57.04, and the Ubuntu 26.04 / ROS 2 Lyrical
container. The benchmark exercises the actual ROS message classes and CDR
serializer, the same conversion used by `WorldSettings`, and the real native
Vulkan bridge.

Reproduce the phase benchmark with:

```bash
docker exec -w /workspace rosplat rosplat-entrypoint \
  python3 -m misc.benchmark_streaming \
  --ply-path data/horse.ply --limit 0 --batch-size 1000 \
  --width 1280 --height 720 --render-samples 5 --json
```

### One-million-splat result

| Phase | Time | Share of new end-to-end processing sum |
| --- | ---: | ---: |
| Publisher: build nested `GaussianArray` objects | 10.537 s | 57.5% |
| Publisher: CDR serialization | 1.173 s | 6.4% |
| Receiver: CDR deserialization | 4.328 s | 23.6% |
| Receiver: nested message to NumPy SoA | 1.294 s | 7.1% |
| NumPy activation conversion and Spirula Vulkan append | 1.003 s | 5.5% |
| **Processing sum** | **18.335 s** | **100%** |

Additional results:

- Throughput excluding the intentional publisher rate limit: 54,541 splats/s.
- Activated float SoA size: 87.74 MiB.
- Serialized CDR wire size: 91.56 MiB.
- Steady dirty render at 1280x720 and 1M splats: 37.06 ms mean,
  37.10 ms median over five frames (about 27 FPS before WGPU upload/UI work).
- Benchmark peak RSS: 532.6 MiB. This includes the source PLY arrays and the
  deliberately retained legacy-concatenation baseline, so it is not an idle
  application-memory measurement.

The real default workflow was also exercised with the PLY publisher at 30 Hz,
ROS discovery/subscription, the complete WGPU/ImGui window, GUI-thread queue
draining, native appends, and final GPU count validation:

```bash
# Terminal 1
python3 misc/generate_gaussian_bag.py \
  --ply-path data/horse.ply --batch-size 1000 --rate 30

# Terminal 2
python3 -m docker.streaming_ui_smoke_test \
  --expected-splats 1000000 --timeout 60
```

It completed with both CPU and GPU counts at 1,000,000 in **38.050 s**. The
rate limit alone requires at least 33.333 s, so the complete application added
about 4.72 s while continuing to render the growing scene.

### Removed quadratic copying

The previous host path concatenated the complete accumulated scene for every
received batch. The CUDA path then used five `torch.cat` calls to rebuild the
complete device scene when cached batches were written through. With 1M SH1
splats in 1,000-splat batches, the benchmark measured/reconstructed:

| Copy behavior | Cumulative data copied |
| --- | ---: |
| Old host full-scene concatenation | 42.88 GiB |
| Old host plus one CUDA write-through per batch | 85.77 GiB |
| New geometric device-buffer growth | 91.42 MiB |

The old host baseline itself took 6.120 s on this machine, excluding the many
temporary allocations and the corresponding CUDA concatenation. The new path
keeps batches separate on CPU and uses capacity doubling on the device.

## Remaining bottlenecks and recommended order

1. **Replace nested per-splat ROS objects with a packed chunk.** Add a versioned
   `GaussianChunk` message containing count, SH degree, refresh flags, and one
   packed `uint8[]` payload (or bounded flat SoA arrays). Decode with NumPy
   views. Keep `GaussianArray` as a compatibility input during migration.
   This directly targets the 57.5% construction and 23.6% deserialization
   costs and avoids one dynamic `float32[]` allocation per splat.
2. **Use larger chunks and throttle preview renders while loading.** The current
   1,000-at-30-Hz policy imposes a 33.3 s floor for 1M splats and can request a
   new render for every 0.1% of the scene. Chunks of roughly 8K-32K, combined
   with a 10-15 FPS loading preview and an immediate final render, should be
   benchmarked for the latency/throughput tradeoff.
3. **Bound reliable QoS queues.** The subscriber currently requests reliable
   delivery with depth 1,000. Append-only geometry cannot safely drop arbitrary
   chunks, but a much smaller reliable queue plus producer backpressure avoids
   retaining a very large number of nested messages when rendering falls
   behind.
4. **Remove the cross-context frame copy if moving-camera FPS is critical.**
   Share a Vulkan device/image and synchronize it directly, or add explicit
   external-memory/semaphore interop between Spirula and WGPU. This is a larger
   integration than the message fix and does not improve static frames, which
   are already cached.
5. **Then profile Spirula kernels for the target scenes.** Native append is only
   5.5% of measured ingestion work. Optimizing it before the ROS representation
   would have low return.

## SOG and Streamed SOG assessment

[SOG (Spatially Ordered Gaussians)](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/)
is a lossy, quantized runtime format. PlayCanvas reports typical files around
15-20x smaller than equivalent PLY and stores co-located Gaussian attributes
in lossless WebP property images. [Streamed SOG](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/streamed-sog/)
adds a spatial tree, multiple LODs, and standard SOG datasets as chunks so a
camera can progressively select parts of very large scenes. The open-source
[SplatTransform](https://github.com/playcanvas/splat-transform) tool reads and
writes both formats.

Those properties make SOG a good fit when:

- a complete static scene can be encoded offline;
- lossy quantization is acceptable;
- network/storage size matters more than live editability; and
- camera-selected spatial LOD is desired for tens of millions of splats.

It is not a direct substitute for ROSplat's current live append stream:

- encoding uses global/spatial ordering, codebooks, quantization, and (for
  Streamed SOG) precomputed LOD chunks;
- a chronological batch of newly reconstructed Gaussians is not automatically
  a valid camera-selectable LOD tree;
- updates would require independent re-encoded chunks or rebuilding affected
  spatial chunks; and
- the pinned Spirula Studio revision has no SOG reader, so ROSplat would need a
  decoder, a chunk/LOD manager, and float staging before the current renderer,
  or new shaders that consume the quantized representation directly.

Recommendation: implement the packed ROS `GaussianChunk` first for live data.
Add SOG later as a separate static-scene transport: publish a versioned scene
URI/hash and let clients fetch SOG/Streamed SOG chunks over file or HTTP range
requests, rather than carrying many WebP assets inside DDS messages. This
preserves ROS for discovery/control while using SOG for the workload it was
designed to solve.

## Validation summary

- Native bridge CTest: 1/1 passed.
- Python tests: 20 passed, 4 subtests passed.
- Vulkan-only Docker image built from Ubuntu 26.04 without CUDA or PyTorch.
- Native renderer smoke: 4 splats, 64x64 RGBA, non-black output.
- Full UI smoke: WGPU forced and reported Vulkan; normal shutdown passed.
- Full streamed UI smoke: 1M published and present in device scene in 38.050 s.
