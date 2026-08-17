#include "spirula_bridge.h"

#include "backend/api/BackendRuntime.h"
#include "backend/api/BackendTypes.h"
#include "core/Tensor.h"
#include "engine/Engine.h"
#include "engine/EngineState.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

template <typename Fn>
int guard(Fn&& fn) noexcept {
    try {
        fn();
        g_last_error.clear();
        return 1;
    } catch (const std::exception& error) {
        g_last_error = error.what();
    } catch (...) {
        g_last_error = "unknown native exception";
    }
    return 0;
}

TorchTensorView view_1d(void* pointer, int64_t count, int channels) {
    return TorchTensorView{
        reinterpret_cast<uint64_t>(pointer),
        sizeof(float),
        {count, channels},
    };
}

TorchTensorView view_2d(void* pointer, int64_t rows, int64_t columns, int channels) {
    return TorchTensorView{
        reinterpret_cast<uint64_t>(pointer),
        sizeof(float),
        {rows, columns, channels},
    };
}

struct DeviceScene {
    float3* means = nullptr;
    float4* quats = nullptr;
    float3* scales = nullptr;
    float* opacities = nullptr;
    float3* features_dc = nullptr;
    float3* features_sh = nullptr;
    int64_t count = 0;
    int64_t capacity = 0;
    int sh_degree = 0;
    int num_sh = 0;

    void release() {
        backend::device_free(means);
        backend::device_free(quats);
        backend::device_free(scales);
        backend::device_free(opacities);
        backend::device_free(features_dc);
        backend::device_free(features_sh);
        means = nullptr;
        quats = nullptr;
        scales = nullptr;
        opacities = nullptr;
        features_dc = nullptr;
        features_sh = nullptr;
        count = 0;
        capacity = 0;
        sh_degree = 0;
        num_sh = 0;
    }
};

struct HostRender {
    float3* rgb = nullptr;
    size_t pixels = 0;

    void resize(size_t requested_pixels) {
        if (requested_pixels <= pixels) return;
        backend::host_free_pinned(rgb);
        rgb = static_cast<float3*>(
            backend::host_malloc_pinned(requested_pixels * sizeof(float3)));
        if (!rgb)
            throw std::runtime_error("failed to allocate pinned render readback buffers");
        pixels = requested_pixels;
    }

    void release() {
        backend::host_free_pinned(rgb);
        rgb = nullptr;
        pixels = 0;
    }
};

class Renderer {
public:
    std::mutex mutex;
    DeviceScene scene;
    HostRender host;
    int width = 0;
    int height = 0;
    bool camera_ready = false;

    void reset(int sh_degree, int64_t initial_capacity) {
        if (sh_degree < 0 || sh_degree > 4)
            throw std::invalid_argument("SH degree must be in [0, 4]");
        if (initial_capacity < 0)
            throw std::invalid_argument("initial capacity cannot be negative");

        engine_reset();
        scene.release();
        scene.sh_degree = sh_degree;
        scene.num_sh = (sh_degree + 1) * (sh_degree + 1) - 1;
        reserve(initial_capacity);
        bind_scene();
    }

    template <typename T>
    static T* grow_buffer(T* old_pointer, int64_t old_count, int64_t new_count) {
        if (new_count == 0) return nullptr;
        T* next = static_cast<T*>(
            backend::device_malloc_checked(static_cast<size_t>(new_count) * sizeof(T),
                                           "ROSplat streaming scene"));
        if (old_pointer && old_count > 0) {
            backend::memcpy_sync(next, old_pointer,
                                 static_cast<size_t>(old_count) * sizeof(T),
                                 backend::MemcpyKind::DeviceToDevice);
        }
        backend::device_free(old_pointer);
        return next;
    }

    void reserve(int64_t requested) {
        if (requested <= scene.capacity) return;
        int64_t next = std::max<int64_t>(requested, std::max<int64_t>(4096, scene.capacity * 2));
        scene.means = grow_buffer(scene.means, scene.count, next);
        scene.quats = grow_buffer(scene.quats, scene.count, next);
        scene.scales = grow_buffer(scene.scales, scene.count, next);
        scene.opacities = grow_buffer(scene.opacities, scene.count, next);
        scene.features_dc = grow_buffer(scene.features_dc, scene.count, next);
        if (scene.num_sh > 0) {
            scene.features_sh = grow_buffer(
                scene.features_sh,
                scene.count * static_cast<int64_t>(scene.num_sh),
                next * static_cast<int64_t>(scene.num_sh));
        }
        scene.capacity = next;
        bind_scene();
    }

    void bind_scene() {
        auto& state = engine();
        state.cur_num_splats = scene.count;
        state.max_num_splats = scene.capacity;
        state.num_sh = scene.num_sh;
        state.sh_degree = scene.sh_degree;

        state.world.means = DeviceVector<float3>(
            view_1d(scene.means, scene.capacity, 3));
        state.world.quats = DeviceVector<float4>(
            view_1d(scene.quats, scene.capacity, 4));
        state.world.scales = DeviceVector<float3>(
            view_1d(scene.scales, scene.capacity, 3));
        state.world.opacities = DeviceVector<float>(
            view_1d(scene.opacities, scene.capacity, 1));
        state.world.features_dc = DeviceVector<float3>(
            view_1d(scene.features_dc, scene.capacity, 3));
        if (scene.num_sh > 0) {
            state.world.features_sh = DeviceTensor2D<float3>(
                view_2d(scene.features_sh, scene.capacity, scene.num_sh, 3));
        } else {
            state.world.features_sh = DeviceTensor2D<float3>();
        }
        state.world.initialized = scene.capacity > 0;
    }

    void append(int64_t batch_count,
                const float* means,
                const float* quats,
                const float* scales,
                const float* opacities,
                const float* dc,
                const float* sh) {
        if (batch_count < 0)
            throw std::invalid_argument("append count cannot be negative");
        if (batch_count == 0) return;
        if (!means || !quats || !scales || !opacities || !dc)
            throw std::invalid_argument("append received a null required array");
        if (scene.num_sh > 0 && !sh)
            throw std::invalid_argument("append received null higher-order SH data");
        if (scene.count > std::numeric_limits<int64_t>::max() - batch_count)
            throw std::overflow_error("splat count overflow");

        const int64_t offset = scene.count;
        reserve(offset + batch_count);
        backend::memcpy_sync(scene.means + offset, means,
                             static_cast<size_t>(batch_count) * sizeof(float3),
                             backend::MemcpyKind::HostToDevice);
        backend::memcpy_sync(scene.quats + offset, quats,
                             static_cast<size_t>(batch_count) * sizeof(float4),
                             backend::MemcpyKind::HostToDevice);
        backend::memcpy_sync(scene.scales + offset, scales,
                             static_cast<size_t>(batch_count) * sizeof(float3),
                             backend::MemcpyKind::HostToDevice);
        backend::memcpy_sync(scene.opacities + offset, opacities,
                             static_cast<size_t>(batch_count) * sizeof(float),
                             backend::MemcpyKind::HostToDevice);
        backend::memcpy_sync(scene.features_dc + offset, dc,
                             static_cast<size_t>(batch_count) * sizeof(float3),
                             backend::MemcpyKind::HostToDevice);
        if (scene.num_sh > 0) {
            backend::memcpy_sync(
                scene.features_sh + offset * scene.num_sh, sh,
                static_cast<size_t>(batch_count) * scene.num_sh * sizeof(float3),
                backend::MemcpyKind::HostToDevice);
        }
        scene.count += batch_count;
        bind_scene();
    }

    void set_camera(int requested_width,
                    int requested_height,
                    const float* view_matrix,
                    const float* intrinsics) {
        if (requested_width <= 0 || requested_height <= 0)
            throw std::invalid_argument("render dimensions must be positive");
        if (!view_matrix || !intrinsics)
            throw std::invalid_argument("camera arrays cannot be null");

        float dist_coeffs[8] = {};
        set_camera_params(
            requested_width,
            requested_height,
            "PINHOLE",
            "NONE",
            TorchTensorView{reinterpret_cast<uint64_t>(view_matrix), sizeof(float), {1, 4, 4}},
            TorchTensorView{reinterpret_cast<uint64_t>(intrinsics), sizeof(float), {1, 4}},
            TorchTensorView{reinterpret_cast<uint64_t>(dist_coeffs), sizeof(float), {1, 8}});
        width = requested_width;
        height = requested_height;
        camera_ready = true;
    }

    void render(uint8_t* output, size_t output_bytes) {
        if (!output) throw std::invalid_argument("render output cannot be null");
        if (!camera_ready) throw std::runtime_error("camera has not been configured");
        const size_t pixel_count = static_cast<size_t>(width) * height;
        if (output_bytes < pixel_count * 4)
            throw std::invalid_argument("render output buffer is too small");
        if (scene.count == 0) {
            for (size_t index = 0; index < pixel_count; ++index) {
                output[index * 4 + 0] = 0;
                output[index * 4 + 1] = 0;
                output[index * 4 + 2] = 0;
                output[index * 4 + 3] = 255;
            }
            return;
        }

        forward_3dgs("3dgs", scene.sh_degree, true);
        auto& state = engine();
        auto& rgb = std::get<0>(state.fwd.renders);
        if (!rgb.data_ptr())
            throw std::runtime_error("Spirula produced no render buffers");

        host.resize(pixel_count);
        backend::memcpy_sync(host.rgb, rgb.data_ptr(), pixel_count * sizeof(float3),
                             backend::MemcpyKind::DeviceToHost);

        for (size_t index = 0; index < pixel_count; ++index) {
            const float3 color = host.rgb[index];
            output[index * 4 + 0] = to_u8(color.x);
            output[index * 4 + 1] = to_u8(color.y);
            output[index * 4 + 2] = to_u8(color.z);
            // The render is already composited over black. Keeping the ImGui
            // texture opaque avoids multiplying edge colors by alpha twice.
            output[index * 4 + 3] = 255;
        }
    }

    void shutdown() {
        engine_reset();
        scene.release();
        host.release();
        width = 0;
        height = 0;
        camera_ready = false;
    }

private:
    static uint8_t to_u8(float value) {
        value = std::clamp(value, 0.0f, 1.0f);
        return static_cast<uint8_t>(std::lround(value * 255.0f));
    }
};

Renderer& renderer() {
    static Renderer instance;
    return instance;
}

}  // namespace

extern "C" {

const char* rosplat_spirula_last_error(void) {
    return g_last_error.c_str();
}

int rosplat_spirula_device_count(void) {
    try {
        g_last_error.clear();
        return backend::device_count();
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return -1;
    }
}

int rosplat_spirula_device_info(int index, RosplatSpirulaDeviceInfo* info) {
    return guard([&] {
        if (!info) throw std::invalid_argument("device info output cannot be null");
        const backend::DeviceInfo source = backend::device_info(index);
        std::memset(info, 0, sizeof(*info));
        std::strncpy(info->name, source.name, sizeof(info->name) - 1);
        info->vram_bytes = source.vram_bytes;
        info->usable = source.usable ? 1 : 0;
    });
}

int rosplat_spirula_select_device(int index) {
    return guard([&] {
        if (!backend::device_select(index))
            throw std::runtime_error("Vulkan device selection failed");
    });
}

int rosplat_spirula_reset_scene(int sh_degree, int64_t initial_capacity) {
    Renderer& instance = renderer();
    std::lock_guard<std::mutex> lock(instance.mutex);
    return guard([&] { instance.reset(sh_degree, initial_capacity); });
}

int rosplat_spirula_append(int64_t count,
                           const float* means_xyz,
                           const float* quaternions_wxyz,
                           const float* log_scales_xyz,
                           const float* opacity_logits,
                           const float* features_dc_rgb,
                           const float* features_sh_rgb) {
    Renderer& instance = renderer();
    std::lock_guard<std::mutex> lock(instance.mutex);
    return guard([&] {
        instance.append(count, means_xyz, quaternions_wxyz, log_scales_xyz,
                        opacity_logits, features_dc_rgb, features_sh_rgb);
    });
}

int rosplat_spirula_set_camera(int width,
                               int height,
                               const float* view_matrix_4x4,
                               const float* intrinsics_fx_fy_cx_cy) {
    Renderer& instance = renderer();
    std::lock_guard<std::mutex> lock(instance.mutex);
    return guard([&] {
        instance.set_camera(width, height, view_matrix_4x4,
                            intrinsics_fx_fy_cx_cy);
    });
}

int rosplat_spirula_render_rgba8(uint8_t* output_rgba, size_t output_bytes) {
    Renderer& instance = renderer();
    std::lock_guard<std::mutex> lock(instance.mutex);
    return guard([&] { instance.render(output_rgba, output_bytes); });
}

int64_t rosplat_spirula_splat_count(void) {
    Renderer& instance = renderer();
    std::lock_guard<std::mutex> lock(instance.mutex);
    return instance.scene.count;
}

int64_t rosplat_spirula_capacity(void) {
    Renderer& instance = renderer();
    std::lock_guard<std::mutex> lock(instance.mutex);
    return instance.scene.capacity;
}

void rosplat_spirula_shutdown(void) {
    Renderer& instance = renderer();
    std::lock_guard<std::mutex> lock(instance.mutex);
    (void)guard([&] { instance.shutdown(); });
}

}  // extern "C"
