"""Lightweight conversion helpers for ``sensor_msgs/msg/Image`` messages."""

from typing import Protocol

import numpy as np


class ImageMessage(Protocol):
    height: int
    width: int
    encoding: str
    is_bigendian: int
    step: int
    data: bytes


_ENCODINGS = {
    "rgb8": (np.dtype("u1"), 3),
    "bgr8": (np.dtype("u1"), 3),
    "rgba8": (np.dtype("u1"), 4),
    "bgra8": (np.dtype("u1"), 4),
    "mono8": (np.dtype("u1"), 1),
    "8uc1": (np.dtype("u1"), 1),
    "mono16": (np.dtype("u2"), 1),
    "16uc1": (np.dtype("u2"), 1),
}


def image_to_rgb8(message: ImageMessage) -> np.ndarray:
    """Convert a common ROS image encoding to a contiguous RGB8 array.

    The converter deliberately supports the encodings ROSplat displays instead
    of pulling in the full OpenCV and cv_bridge dependency stack.
    """
    encoding = message.encoding.lower()
    try:
        dtype, channels = _ENCODINGS[encoding]
    except KeyError as error:
        supported = ", ".join(sorted(_ENCODINGS))
        raise ValueError(
            f"Unsupported image encoding {message.encoding!r}; supported: {supported}"
        ) from error

    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    bytes_per_pixel = dtype.itemsize * channels
    packed_row_size = width * bytes_per_pixel
    if step < packed_row_size:
        raise ValueError(
            f"Image step {step} is smaller than the packed row size {packed_row_size}"
        )

    raw = np.frombuffer(message.data, dtype=np.uint8)
    required_size = height * step
    if raw.size < required_size:
        raise ValueError(
            f"Image data has {raw.size} bytes, but {required_size} are required"
        )

    packed = np.ascontiguousarray(
        raw[:required_size].reshape(height, step)[:, :packed_row_size]
    )
    if dtype.itemsize > 1:
        byte_order = ">" if bool(message.is_bigendian) else "<"
        dtype = dtype.newbyteorder(byte_order)

    pixels = packed.view(dtype).reshape(height, width, channels)

    if encoding == "bgr8":
        rgb = pixels[:, :, ::-1]
    elif encoding == "bgra8":
        rgb = pixels[:, :, [2, 1, 0]]
    elif encoding == "rgba8":
        rgb = pixels[:, :, :3]
    elif channels == 1:
        grayscale = pixels[:, :, 0]
        if dtype.itemsize > 1:
            grayscale = (
                grayscale.astype(np.uint32) * 255 // np.iinfo(np.uint16).max
            ).astype(np.uint8)
        rgb = np.repeat(grayscale[:, :, None], 3, axis=2)
    else:
        rgb = pixels

    return np.ascontiguousarray(rgb, dtype=np.uint8)
