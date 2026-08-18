ROSplat
=======

_The Online ROS2-Based Gaussian Splatting-Enabled Visualizer_

[Shady Gmira](https://www.linkedin.com/in/shady-gmira-ba678121a/)

![Project Image](https://github.com/shadygm/ROSplat/blob/main/assets/images/image.png) ![Demo Animation](https://github.com/shadygm/ROSplat/raw/main/assets/gifs/output.gif)

Overview
--------

ROSplat is the first online ROS2-based visualizer that leverages Gaussian splatting to render complex 3D scenes. It is designed to efficiently visualize millions of Gaussians by using custom ROS2 messages and GPU-accelerated sorting and rendering techniques. ROSplat also supports data loading from PLY files.

Features
--------

*   **Real-Time Visualization:** Render millions of Gaussian "splats" in real time.
*   **ROS2 Integration:** Built on ROS2 for online data exchange of Gaussians, Images, and IMU data.
*   **Custom Gaussian Messages:** Uses custom message types (_SingleGaussian_ and _GaussianArray_) to encapsulate properties such as position, rotation, scale, opacity, and spherical harmonics.
*   **Portable Vulkan Rendering:** Uses Spirula Studio's Vulkan projection,
    tile sorting, and Gaussian rasterization through a small native bridge.
*   **Vulkan ImGui UI:** The application and all dynamic textures use
    Dear ImGui through wgpu-py's Vulkan backend; there is no CUDA, Torch,
    gsplat, or OpenGL renderer path.

Setup
-----

The supported container targets **Ubuntu 26.04 LTS**, **ROS 2 Lyrical**, and a
Vulkan 1.2-capable GPU. The renderer requires Vulkan buffer device addresses
and timeline semaphores, matching Spirula Studio's backend requirements.

### Dependencies

*   Docker with the Compose plugin
*   A Vulkan 1.2-capable GPU and current graphics driver
*   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    when using an NVIDIA GPU through Docker

The Docker image builds the pinned Spirula Studio source, its Slang shaders,
the ROSplat native bridge, ROS, the Vulkan UI dependencies, and the generated
`gaussian_interface` messages. CUDA and PyTorch are not required.

### Docker-Based Setup

Docker is the supported setup path. On Arch Linux, install the NVIDIA Container Toolkit once:

    sudo pacman -S --needed nvidia-container-toolkit

The Compose service requests every available GPU through the toolkit's CDI device. Confirm the CDI devices are visible with:

    nvidia-ctk cdi list

Then build and run ROSplat:

    ./docker/run_docker.sh -b    # Build rosplat:lyrical-vulkan once
    ./docker/launch_ui.sh        # Start/reuse the container and launch ROSplat

For an interactive development shell or cleanup:

    ./docker/run_docker.sh -u
    ./docker/run_docker.sh -c

Run the unit-test suite inside the development container with:

    docker compose -f docker/docker-compose.yml exec rosplat rosplat-entrypoint python3 -m unittest discover -s tests

The Compose configuration enables the graphics, display, and utility driver
capabilities required for Vulkan and X11 presentation. `wgpu-py` is forced to
Vulkan and ROSplat rejects a different backend at startup.

The renderer-correctness findings, measured one-million-splat streaming
breakdown, and SOG recommendation are documented in
[docs/renderer-streaming-investigation.md](docs/renderer-streaming-investigation.md).

### Native build without Docker

Initialize the pinned renderer dependency and build the bridge:

    git submodule update --init --recursive
    cmake -S . -B build-native -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build-native --target rosplat_spirula

The first configure downloads Spirula's pinned Slang compiler and the first
build compiles its SPIR-V shader set. ROSplat finds
`build-native/librosplat_spirula.so` automatically. Override the path with
`ROSPLAT_SPIRULA_LIBRARY` when installing elsewhere.

### Third-party licensing

Spirula Studio stays pinned and unmodified in `external/spirula-studio`, with
its upstream history and GPL-3.0 license intact. ROSplat-owned bridge code is
under `native/`. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the
exact revision and license boundary. ROSplat itself is GPL-3.0, so the combined
distribution is license-compatible without relicensing upstream files.

Building the Gaussian Messages
------------------------------

ROSplat defines two custom ROS2 messages to handle Gaussian data, located in the `gaussian_interface/msg` folder.

> **Note:** The Gaussian messages are based on the [original Gaussian Splatting implementation by graphdeco-inria](https://github.com/graphdeco-inria/gaussian-splatting).

### Message Definitions

#### 1\. SingleGaussian.msg

    # Gaussian.msg
    
    # 3D Pose
    float32[3] xyz
    
    # Rotation as Quaternion
    float32[4] rotation
    
    # Scale along each axis
    float32[3] scale
    
    # Opacity
    # Quantized as a number between 0-255 instead of 0-1
    uint8 opacity
    
    # Spherical Harmonics
    float32[] spherical_harmonics

#### 2\. GaussianArray.msg

    gaussian_interface/SingleGaussian[] gaussians

### Building the Messages Without Docker

**a) Build your workspace using colcon:**

    colcon build --packages-select gaussian_interface

**b) Source your workspace:**

    . install/setup.bash

> **Important:** Depending on your shell, you might need to adjust these commands.

Usage
-----

Inside the Docker container the messages are already built and sourced. For a manual installation, build the messages first and then launch the visualizer from the project's root directory:

    python3 -m rosplat.main

### Testing Gaussian Visualization

The development container is a reusable background service. Build it once
after cloning or whenever Docker/native dependencies change:

```bash
./docker/run_docker.sh -b
```

Normal Python source edits are bind-mounted into `/workspace` immediately and
do not require rebuilding or recreating the container. Use the two host-side
helpers from the repository root:

```bash
# Terminal 1: start/reuse the container and run the Vulkan ImGui app
./docker/launch_ui.sh

# Terminal 2: stream a repository PLY using the same container
./docker/publish_ply.sh data/horse.ply
```

The publisher accepts the existing options after the PLY path:

```bash
./docker/publish_ply.sh data/horse.ply --batch-size 10000 --rate 10
./docker/publish_ply.sh data/horse.ply --topic /my_gaussians
```

The default topic is `/gaussian_test`. In ROSplat, select it in **ROS
Settings** and click **Add**. The publisher waits for that subscription before
sending anything, so either terminal can be started first.

Closing the UI or publisher only stops that foreground process; the container
continues running and is reused by the next command. Check or stop it explicitly:

```bash
docker compose -f docker/docker-compose.yml ps
./docker/run_docker.sh -c
```

Use `-c` only when you intentionally want to stop the development service.

Contributions
-------------

Contributions and feedback are welcome!

Acknowledgments
---------------

I'm glad to have worked on such a challenging topic and grateful for the invaluable advice and support I received throughout this project.

Special thanks to [Qihao Yuan](https://scholar.google.com/citations?user=14GwKcMAAAAJ&hl=en) and [Kailai Li](https://kailaili.github.io/) for their guidance and encouragement as well as the constructive feedback that helped shape this work.

This project was additionally influenced by [limacv](https://github.com/limacv) 's implementation of the [GaussianSplattingViewer](https://github.com/limacv/GaussianSplattingViewer) repository.

Contact
-------

For questions or further information, please email: **shady.gmira\[at\]gmail.com**
