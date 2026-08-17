#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  define ROSPLAT_API __declspec(dllexport)
#else
#  define ROSPLAT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RosplatSpirulaDeviceInfo {
    char name[256];
    uint64_t vram_bytes;
    int usable;
} RosplatSpirulaDeviceInfo;

ROSPLAT_API const char* rosplat_spirula_last_error(void);
ROSPLAT_API int rosplat_spirula_device_count(void);
ROSPLAT_API int rosplat_spirula_device_info(
    int index,
    RosplatSpirulaDeviceInfo* info
);
ROSPLAT_API int rosplat_spirula_select_device(int index);

/*
 * Reset the active scene and reserve device storage. `sh_degree` is 0..4 and
 * `initial_capacity` may be zero. Scale and opacity inputs use Spirula's raw
 * training representation: log(scale) and logit(opacity).
 */
ROSPLAT_API int rosplat_spirula_reset_scene(
    int sh_degree,
    int64_t initial_capacity
);

/* Append one contiguous struct-of-arrays batch to the device-resident scene. */
ROSPLAT_API int rosplat_spirula_append(
    int64_t count,
    const float* means_xyz,
    const float* quaternions_wxyz,
    const float* log_scales_xyz,
    const float* opacity_logits,
    const float* features_dc_rgb,
    const float* features_sh_rgb
);

/* OpenCV camera convention: +Z forward, +Y down, row-major world-to-camera. */
ROSPLAT_API int rosplat_spirula_set_camera(
    int width,
    int height,
    const float* view_matrix_4x4,
    const float* intrinsics_fx_fy_cx_cy
);

/* Render top-to-bottom RGBA8 into caller-owned storage. */
ROSPLAT_API int rosplat_spirula_render_rgba8(
    uint8_t* output_rgba,
    size_t output_bytes
);

ROSPLAT_API int64_t rosplat_spirula_splat_count(void);
ROSPLAT_API int64_t rosplat_spirula_capacity(void);
ROSPLAT_API void rosplat_spirula_shutdown(void);

#ifdef __cplusplus
}
#endif
