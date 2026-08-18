import unittest

from misc.generate_gaussian_bag import make_gaussian_batch
from rosplat.core.gaussian_representation import naive_gaussian


class GaussianPublisherTest(unittest.TestCase):
    def test_serializes_a_bounded_refresh_batch(self):
        source = naive_gaussian()

        message = make_gaussian_batch(source, 1, 3, refresh=True)

        self.assertTrue(message.refresh)
        self.assertEqual(len(message.gaussians), 2)
        self.assertEqual(list(message.gaussians[0].xyz), source.xyz[1].tolist())
        self.assertEqual(
            list(message.gaussians[1].spherical_harmonics), source.sh[2].tolist()
        )
        self.assertEqual(message.gaussians[0].opacity, 255)


if __name__ == "__main__":
    unittest.main()
