import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from rosplat.core.gaussian_representation import _infer_sh_degree, from_ply


EXTRA_FEATURES_BY_DEGREE = {0: 0, 1: 9, 2: 24, 3: 45}


def write_gaussian_ply(path: Path, degree: int) -> None:
    fields = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("opacity", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    fields.extend((f"f_rest_{index}", "f4") for index in range(EXTRA_FEATURES_BY_DEGREE[degree]))
    fields.extend((f"scale_{index}", "f4") for index in range(3))
    fields.extend((f"rot_{index}", "f4") for index in range(4))

    vertices = np.zeros(2, dtype=fields)
    vertices["z"] = [1.0, 2.0]
    vertices["rot_0"] = 1.0
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


class GaussianRepresentationTest(unittest.TestCase):
    def test_loads_every_supported_sh_degree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for degree in range(4):
                with self.subTest(degree=degree):
                    path = Path(temp_dir) / f"degree_{degree}.ply"
                    write_gaussian_ply(path, degree)

                    gaussians = from_ply(path)

                    self.assertEqual(gaussians.xyz.shape, (2, 3))
                    self.assertEqual(gaussians.sh.shape, (2, 3 * (degree + 1) ** 2))

    def test_can_cap_the_loaded_degree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "degree_3.ply"
            write_gaussian_ply(path, 3)

            gaussians = from_ply(path, max_sh_degree=1)

            self.assertEqual(gaussians.sh.shape, (2, 12))

    def test_rejects_a_non_sh_coefficient_count(self):
        with self.assertRaisesRegex(ValueError, "0, 9, 24, or 45"):
            _infer_sh_degree(6)


if __name__ == "__main__":
    unittest.main()
