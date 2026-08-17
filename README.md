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
*   **CUDA and OpenGL Rendering:** Supports GPU-accelerated rendering using CUDA and OpenGL.

Setup
-----

The supported container targets **Ubuntu 26.04 LTS**, **ROS 2 Lyrical**, and **CUDA 13.3**. ROSplat can fall back to its OpenGL renderer, but the CUDA renderer requires an NVIDIA GPU.

### Dependencies

*   Docker with the Compose plugin
*   An NVIDIA driver compatible with CUDA 13
*   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

The Docker image contains ROS, CUDA, PyTorch, gsplat, the GUI dependencies, and the generated `gaussian_interface` messages. A separate host CUDA toolkit is not required.

### Docker-Based Setup

Docker is the supported setup path. On Arch Linux, install the NVIDIA Container Toolkit once:

    sudo pacman -S --needed nvidia-container-toolkit

The Compose service requests every available GPU through the toolkit's CDI device. Confirm the CDI devices are visible with:

    nvidia-ctk cdi list

Then build and run ROSplat:

    cd docker
    ./run_docker.sh -b    # Build rosplat:lyrical-cuda13.3
    ./run_docker.sh -r    # Start the container and launch ROSplat

For an interactive development shell or cleanup:

    ./run_docker.sh -u
    ./run_docker.sh -c

After starting the container, verify ROS imports and a real gsplat CUDA kernel with:

    docker compose exec rosplat rosplat-entrypoint python3 docker/smoke_test.py

The launch script queries all installed GPUs with `nvidia-smi` and passes their compute capabilities to gsplat through `TORCH_CUDA_ARCH_LIST`. Duplicate architectures are removed automatically. You can override detection when needed:

    TORCH_CUDA_ARCH_LIST=8.9 ./run_docker.sh -r

Refer to [NVIDIA's CUDA GPU support matrix](https://developer.nvidia.com/cuda-gpus) for the correct compute capability.

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

To test visualizing Gaussians over ROS2 messages:

1.  Place your `PLY` file under the `data/` directory.
2.  Start the container with `./run_docker.sh -u` in two terminals:

*   **Terminal 1:** Run the visualizer:

    ```bash
        python3 -m rosplat.main
    ```

*   **Terminal 2:** Publish Gaussian data:

    ```bash
        python3 misc/generate_gaussian_bag.py --ply-path data/your_file.ply
    ```

The `generate_gaussian_bag.py` script waits for a subscriber, then publishes the scene once in bounded batches on `/gaussian_test`. In ROSplat, select that topic in **ROS Settings** and click **Add** to begin the stream.

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
