# Third-party source dependencies

ROSplat keeps source dependencies with their upstream license texts under
`external/`. The authoritative license for each dependency is the file in its
own source tree.

| Dependency | Revision | License | Source |
| --- | --- | --- | --- |
| Spirula Studio | `8509253d7247f3b6ffad4cdae241b5e94410c3fe` | GPL-3.0 | <https://github.com/harry7557558/spirula-studio> |

ROSplat-specific adapter code is not copied into the Spirula Studio submodule,
and no Spirula Studio files are relicensed.

## Vulkan UI runtime packages

These packages are installed from their published Python distributions. Their
license files remain in each installed distribution; they are not copied into
ROSplat source files.

| Dependency | Version | License | Purpose |
| --- | --- | --- | --- |
| imgui-bundle | `1.92.801` | MIT | Dear ImGui Python bindings and widgets |
| wgpu-py | `0.32.0` | BSD-2-Clause | WebGPU renderer using the Vulkan backend |
| rendercanvas | `2.7.2` | BSD-2-Clause | Window and presentation event loop |
| glfw (Python wrapper) | `2.10.2` | MIT | rendercanvas desktop window backend |
