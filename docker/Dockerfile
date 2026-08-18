# syntax=docker/dockerfile:1

FROM ubuntu:26.04

LABEL org.opencontainers.image.title="ROSplat"
LABEL org.opencontainers.image.description="ROSplat with ROS 2 Lyrical and Spirula Vulkan rendering"
LABEL org.opencontainers.image.source="https://github.com/shadygm/ROSplat"

ARG ROS_APT_SOURCE_VERSION=1.2.0
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=lyrical \
    VIRTUAL_ENV=/opt/rosplat-venv \
    PATH=/opt/rosplat-venv/bin:${PATH} \
    HOME=/home/rosplat

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        locales \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && locale-gen en_US.UTF-8 \
    && curl -fsSL \
        -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.resolute_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb \
    && rm /tmp/ros2-apt-source.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 \
        libgl1 \
        libglvnd-dev \
        libglfw3 \
        libglx0 \
        libvulkan-dev \
        libx11-6 \
        libx11-xcb1 \
        libxcursor1 \
        libxext6 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        mesa-utils \
        python3-colcon-common-extensions \
        python3-numpy \
        ros-lyrical-geometry-msgs \
        ros-lyrical-rmw-fastrtps-cpp \
        ros-lyrical-ros-base \
        ros-lyrical-rosidl-default-generators \
        ros-lyrical-sensor-msgs \
        zenity \
        vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv --system-site-packages "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade pip "setuptools>=77,<80" wheel

COPY CMakeLists.txt /tmp/rosplat-native/CMakeLists.txt
COPY native /tmp/rosplat-native/native
COPY external/spirula-studio /tmp/rosplat-native/external/spirula-studio

RUN cmake -S /tmp/rosplat-native -B /tmp/rosplat-native-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/rosplat-native \
        -DROSPLAT_BUILD_TESTS=OFF \
    && cmake --build /tmp/rosplat-native-build --target rosplat_spirula \
        --parallel "$(nproc)" \
    && cmake --install /tmp/rosplat-native-build \
    && rm -rf /tmp/rosplat-native /tmp/rosplat-native-build

ENV ROSPLAT_SPIRULA_LIBRARY=/opt/rosplat-native/lib/rosplat/librosplat_spirula.so

COPY requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY gaussian_interface /tmp/rosplat_ws/src/gaussian_interface

RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && colcon --log-base /tmp/rosplat_ws/log build \
        --base-paths /tmp/rosplat_ws/src \
        --build-base /tmp/rosplat_ws/build \
        --install-base /opt/rosplat-msgs \
        --merge-install \
        --packages-select gaussian_interface \
    && rm -rf /tmp/rosplat_ws

COPY docker/entrypoint.sh /usr/local/bin/rosplat-entrypoint

RUN chmod 0755 /usr/local/bin/rosplat-entrypoint \
    && set -eux; \
       group_name="$(getent group "${USER_GID}" | cut -d: -f1 || true)"; \
       if [[ -z "${group_name}" ]]; then \
           group_name=rosplat; \
           groupadd --gid "${USER_GID}" "${group_name}"; \
       fi; \
       user_name="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)"; \
       if [[ -z "${user_name}" ]]; then \
           useradd --create-home --home-dir /home/rosplat \
               --uid "${USER_UID}" --gid "${group_name}" rosplat; \
       else \
           usermod --home /home/rosplat --gid "${group_name}" "${user_name}"; \
       fi; \
       mkdir -p /workspace /home/rosplat; \
       chown -R "${USER_UID}:${USER_GID}" /workspace /home/rosplat

USER ${USER_UID}:${USER_GID}
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/rosplat-entrypoint"]
CMD ["sleep", "infinity"]
