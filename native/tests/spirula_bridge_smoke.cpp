#include "spirula_bridge.h"

#include <algorithm>
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
        !rosplat_spirula_reset_scene(1, 1)) {
        std::cerr << rosplat_spirula_last_error() << '\n';
        return 1;
    }

    const float means[] = {0.0f, 0.0f, 3.0f};
    const float quats[] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float scales[] = {-1.5f, -1.5f, -1.5f};
    const float opacities[] = {4.0f};
    const float dc[] = {0.0f, 0.0f, 0.0f};
    const float sh[] = {
        0.75f, 0.25f, -0.5f,
        0.75f, 0.25f, -0.5f,
        0.75f, 0.25f, -0.5f,
    };
    if (!rosplat_spirula_append(1, means, quats, scales, opacities, dc, sh)) {
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
    const std::vector<uint8_t> full_sh_output = output;

    if (!rosplat_spirula_set_render_options(
            ROSPLAT_SPIRULA_OUTPUT_COLOR, 0) ||
        !rosplat_spirula_render_rgba8(output.data(), output.size())) {
        std::cerr << rosplat_spirula_last_error() << '\n';
        return 1;
    }
    if (std::equal(output.begin(), output.end(), full_sh_output.begin())) {
        std::cerr << "SH degree selection did not change the color render\n";
        return 1;
    }

    if (!rosplat_spirula_set_render_options(
            ROSPLAT_SPIRULA_OUTPUT_DEPTH, -1) ||
        !rosplat_spirula_render_rgba8(output.data(), output.size())) {
        std::cerr << rosplat_spirula_last_error() << '\n';
        return 1;
    }
    if (output[(32 * 64 + 32) * 4 + 0] == 0) {
        std::cerr << "center depth pixel remained black\n";
        return 1;
    }

    if (!rosplat_spirula_set_render_options(
            ROSPLAT_SPIRULA_OUTPUT_OPACITY, 0) ||
        !rosplat_spirula_render_rgba8(output.data(), output.size())) {
        std::cerr << rosplat_spirula_last_error() << '\n';
        return 1;
    }
    if (output[(32 * 64 + 32) * 4 + 0] == 0) {
        std::cerr << "center opacity pixel remained black\n";
        return 1;
    }
    rosplat_spirula_shutdown();
    return 0;
}
