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

*   **Real-Time Visualization:** Render millions of Gaussian “splats” in real time.
*   **ROS2 Integration:** Built on ROS2 for online data exchange of Gaussians, Images, and IMU data.
*   **Custom Gaussian Messages:** Uses custom message types (_SingleGaussian_ and _GaussianArray_) to encapsulate properties such as position, rotation, scale, opacity, and spherical harmonics.
*   **CUDA and OpenGL Rendering:** Supports GPU-accelerated rendering using CUDA and OpenGL.

Setup
-----

This project was developed and tested on **Ubuntu 24.04 LTS** using **ROS2 Jazzy**. Please note: **Performance degrades significantly without an NVIDIA graphics card.**

### Dependencies

*   **Mandatory:** ROS2 (tested on ROS2 Jazzy)
*   **Optional (for GPU-based Sorting):**
    *   `cupy` (ensure compatibility with your CUDA version)
    *   `torch` (if using PyTorch for GPU sorting)
    *   `gsplat` (for CUDA-based rendering)

To install the optional GPU-based libraries individually:

    pip install cupy-cuda12x  # Install Cupy (replace 12x with your CUDA version)

    pip install torch         # Install PyTorch

    pip install git+https://github.com/nerfstudio-project/gsplat.git # Install gsplat for CUDA-based rendering
The program will automatically prioritize sorting methods in the following order: **1) Torch → 2) Cupy → 3) CPU**

To install all dependencies at once:

    pip install -r requirements.txt        # For GPU acceleration


    pip install -r requirements-no-gpu.txt  # Without GPU acceleration

### Docker-Based Setup

Alternatively, you can set up the project using Docker. A setup script is available under the `docker` directory.

Before running Docker, ensure you have installed:

    sudo apt-get install -y nvidia-container-toolkit

This enables GPU communication between the host and the container. If you come across any other issues, follow the instructions under [the official NVIDIA guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Then, to build and run the Docker container:

    cd docker
    ./run_docker.sh -h    # Display help and usage instructions
    ./run_docker.sh -bu   # Build the Docker image and launch the container with docker-compose

**Important:** Ensure the host machine's CUDA version matches the version specified in the Dockerfile. If you are using a CUDA version other than 12.6, update the Dockerfile accordingly.

> **Note:** If you require CUDA-based rendering, you must install the `gsplat` library manually within the container after it is launched.

#### Accessing ROSplat Inside the Container

After starting the Docker container, navigate to the project directory inside the container:

    cd projects/ROSplat

Building the Gaussian Messages
------------------------------

ROSplat defines two custom ROS2 messages to handle Gaussian data, located in the `gaussian_interface/msg` folder.

> **Note:** The Gaussian messages are based on the [original Gaussian Splatting implementation by graphdeco-inria](https://github.com/graphdeco-inria/gaussian-splatting).

### Message Definitions

#### 1\. SingleGaussian.msg

    geometry_msgs/msg/Point xyz
    geometry_msgs/msg/Quaternion rotation
    float32 opacity
    geometry_msgs/msg/Vector3 scale
    float32[] spherical_harmonics

#### 2\. GaussianArray.msg

    gaussian_interface/SingleGaussian[] gaussians

### Building the Messages

**a) Build your workspace using colcon:**

    colcon build --packages-select gaussian_interface

**b) Source your workspace:**

    . install/setup.bash

> **Important:** Depending on your shell, you might need to adjust these commands.

Usage
-----

Once the Gaussian messages are built, you can launch the visualizer from the project's root directory:

    cd src
    python3 main.py

### Testing Gaussian Visualization

To test visualizing Gaussians over ROS2 messages:

1.  Place your `PLY` file under the `data/` directory.
2.  Open **two terminals inside the Docker container**:

*   **Terminal 1:** Run the visualizer:

    cd projects/ROSplat/src
    python3 main.py

*   **Terminal 2:** Publish Gaussian data:

    cd projects/ROSplat/misc
    python3 generate_gaussian_bag.py --ply_path ../data/your_file.ply

The `generate_gaussian_bag.py` script will continuously publish batches of Gaussian messages to the `/gaussian_test` topic, which the visualizer will display in real time whenever you are subscribed to it.

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