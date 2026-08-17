#include "spirula_bridge.h"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    const int devices = rosplat_spirula_device_count();
    if (devices < 0) {
        std::cerr << rosplat_spirula_last_error() << '\n';
        return 1;
    }
    if (devices == 0) {
        std::cout << "SKIP: no usable Vulkan devices\n";
        return 0;
    }
    if (!rosplat_spirula_select_device(0) ||
        !rosplat_spirula_reset_scene(0, 1)) {
        std::cerr << rosplat_spirula_last_error() << '\n';
        return 1;
    }

    const float means[] = {0.0f, 0.0f, 3.0f};
    const float quats[] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float scales[] = {-1.5f, -1.5f, -1.5f};
    const float opacities[] = {4.0f};
    const float dc[] = {1.77245385f, -1.77245385f, -1.77245385f};
    if (!rosplat_spirula_append(1, means, quats, scales, opacities, dc, nullptr)) {
        std::cerr << rosplat_spirula_last_error() << '\n';
        return 1;
    }

    const float view[] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const float intrinsics[] = {32.0f, 32.0f, 32.0f, 32.0f};
    if (!rosplat_spirula_set_camera(64, 64, view, intrinsics)) {
        std::cerr << rosplat_spirula_last_error() << '\n';
        return 1;
    }

    std::vector<uint8_t> output(64 * 64 * 4);
    if (!rosplat_spirula_render_rgba8(output.data(), output.size())) {
        std::cerr << rosplat_spirula_last_error() << '\n';
        return 1;
    }
    if (output[(32 * 64 + 32) * 4 + 0] == 0) {
        std::cerr << "center pixel remained black\n";
        return 1;
    }
    rosplat_spirula_shutdown();
    return 0;
}
