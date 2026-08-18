import unittest
from types import SimpleNamespace

import numpy as np

from rosplat.ros.image_conversion import image_to_rgb8


def image_message(*, width, height, encoding, step, data, is_bigendian=0):
    return SimpleNamespace(
        width=width,
        height=height,
        encoding=encoding,
        step=step,
        data=data,
        is_bigendian=is_bigendian,
    )


class ImageConversionTest(unittest.TestCase):
    def test_converts_bgr_and_ignores_row_padding(self):
        message = image_message(
            width=2,
            height=1,
            encoding="bgr8",
            step=8,
            data=bytes([30, 20, 10, 60, 50, 40, 255, 255]),
        )

        converted = image_to_rgb8(message)

        np.testing.assert_array_equal(
            converted, np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        )
        self.assertTrue(converted.flags.c_contiguous)

    def test_drops_rgba_alpha_channel(self):
        message = image_message(
            width=1,
            height=1,
            encoding="rgba8",
            step=4,
            data=bytes([1, 2, 3, 4]),
        )

        np.testing.assert_array_equal(
            image_to_rgb8(message), np.array([[[1, 2, 3]]], dtype=np.uint8)
        )

    def test_scales_big_endian_mono16(self):
        message = image_message(
            width=2,
            height=1,
            encoding="mono16",
            step=4,
            data=bytes([0, 0, 255, 255]),
            is_bigendian=1,
        )

        np.testing.assert_array_equal(
            image_to_rgb8(message),
            np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8),
        )

    def test_rejects_an_unsupported_encoding(self):
        message = image_message(
            width=1,
            height=1,
            encoding="yuv422",
            step=2,
            data=bytes(2),
        )

        with self.assertRaisesRegex(ValueError, "Unsupported image encoding"):
            image_to_rgb8(message)

    def test_rejects_truncated_data(self):
        message = image_message(
            width=2,
            height=2,
            encoding="rgb8",
            step=6,
            data=bytes(11),
        )

        with self.assertRaisesRegex(ValueError, "12 are required"):
            image_to_rgb8(message)


if __name__ == "__main__":
    unittest.main()
