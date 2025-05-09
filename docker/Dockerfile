FROM nvidia/cuda:12.6.0-devel-ubuntu24.04
LABEL maintainer="Shady Gmira <shady.gmira@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

# Base packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    gnupg2 \
    lsb-release \
    locales \
    python3 \
    python3-pip \
    python3-full \
    sudo \
    x11-apps \
    zenity \
    openssh-client \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Locale
RUN locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

ENV ROS_DISTRO=jazzy

# ROS 2 setup
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | \
    gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-desktop \
    python3-colcon-common-extensions \
    ros-dev-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --break-system-packages --ignore-installed -r requirements.txt

# Change this env variable to your own system requirements
ENV TORCH_CUDA_ARCH_LIST="8.6" 
RUN pip install --break-system-packages git+https://github.com/nerfstudio-project/gsplat.git

# OpenGL & NVIDIA driver integration (Ubuntu 24.04, Driver 550)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd-dev \
    libgl1 \
    libglx0 \
    libegl1 \
    libx11-6 \
    libglfw3-dev \
    mesa-utils

ARG USER_UID
ARG USER_GID
ARG USERNAME

RUN set -eux; \
    # Remove user with same UID if exists
    EXISTING_USER_BY_UID=$(getent passwd "${USER_UID}" | cut -d: -f1 || true); \
    if [ -n "$EXISTING_USER_BY_UID" ]; then \
    userdel -r "$EXISTING_USER_BY_UID" || true; \
    fi; \
    \
    # Remove user with same name if exists
    if id "${USERNAME}" >/dev/null 2>&1; then \
    userdel -r "${USERNAME}" || true; \
    fi; \
    \
    # Remove group with same GID if exists
    if getent group "${USER_GID}" >/dev/null; then \
    groupdel "$(getent group "${USER_GID}" | cut -d: -f1)"; \
    fi; \
    \
    # Remove group with same name if exists
    if getent group "${USERNAME}" >/dev/null; then \
    groupdel "${USERNAME}"; \
    fi; \
    \
    # Now create group and user cleanly
    groupadd --gid "${USER_GID}" "${USERNAME}"; \
    useradd -m -u "${USER_UID}" -g "${USER_GID}" "${USERNAME}"; \
    usermod -aG sudo "${USERNAME}"; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers



# Setup bashrc for ROS 2
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USERNAME}/.bashrc && \
    chown ${USER_UID}:${USER_GID} /home/${USERNAME}/.bashrc

# Set environment
ENV USER=${USERNAME}
ENV HOME=/home/${USERNAME}
WORKDIR /home/${USERNAME}
