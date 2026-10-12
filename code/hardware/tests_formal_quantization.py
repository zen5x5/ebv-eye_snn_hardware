"""Focused checks for the formal multi-subject fixed-point evaluation."""

import unittest

import numpy as np
import torch

from .evaluate_formal_quantization import (
    ExactFixedPointReference, errors, normalized_to_pixels, project_grid,
)
from .evaluate_pursuit_fixed_point import FixedLayer, RuntimeCounters


class FormalQuantizationTests(unittest.TestCase):
    def setUp(self):
        self.reference = ExactFixedPointReference.__new__(ExactFixedPointReference)
        self.reference.accumulator_minimum = -(2**23)
        self.reference.accumulator_maximum = 2**23 - 1
        self.reference.spike_maximum_integer = 255
        self.reference.alpha_rounding = "floor"

    def layer(self, weights, constant, pool=False):
        count = len(weights)
        return FixedLayer("check", torch.tensor(weights, dtype=torch.float32),
                          torch.tensor(constant, dtype=torch.float32),
                          torch.full((count,), 10), torch.full((count,), 10),
                          torch.zeros(count, dtype=torch.int64), pool)

    def test_convolution_matches_int64_accumulation_and_pool(self):
        rng = np.random.default_rng(2021)
        x = rng.integers(0, 175, (2, 3, 6, 8), dtype=np.int64)
        w = rng.integers(-32768, 32768, (4, 3, 3, 3), dtype=np.int64)
        b = np.array([-1024, 4096, 0, 8192], dtype=np.int64)
        expected = np.zeros((2, 4, 6, 8), dtype=np.int64)
        padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        for y in range(6):
            for z in range(8):
                for channel in range(4):
                    expected[:, channel, y, z] = (
                        padded[:, :, y:y+3, z:z+3] * w[channel]
                    ).sum(axis=(1, 2, 3)) + b[channel]
        expected = expected.clip(-(2**23), 2**23-1)
        expected = expected.reshape(2, 4, 3, 2, 4, 2).max(axis=(3, 5))
        actual = self.reference._linear_or_convolution(
            torch.from_numpy(x), self.layer(w, b, True), RuntimeCounters.empty()
        )
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_linear_matches_int64(self):
        x = np.array([[174, 17, 255], [0, 250, 32]], dtype=np.int64)
        w = np.array([[32767, -32768, 532], [-23000, 30000, 2]], dtype=np.int64)
        expected = (x @ w.T + np.array([17, -11])).clip(-(2**23), 2**23-1)
        actual = self.reference._linear_or_convolution(
            torch.from_numpy(x), self.layer(w, [17, -11]), RuntimeCounters.empty()
        )
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_lif_negative_leak_and_subtractive_reset(self):
        previous = torch.tensor([[-3., 513.]])
        current = torch.tensor([[0., 2300.]])
        layer = self.layer([[0.], [0.]], [0., 0.])
        spikes, membrane = self.reference._lif(current, previous, layer, RuntimeCounters.empty())
        np.testing.assert_array_equal(spikes.numpy(), [[0., 2.]])
        np.testing.assert_array_equal(membrane.numpy(), [[-3., 636.]])

    def test_grid_ties_and_zero_error(self):
        points = np.array([[189., 231.], [189.01, 231.01], [-100., 3000.]])
        np.testing.assert_array_equal(project_grid(points), [[150., 150.], [228., 312.], [150., 1770.]])
        np.testing.assert_array_equal(normalized_to_pixels(np.array([[0., 0.]])), [[540., 960.]])
        angle, pixel = errors(points, points)
        np.testing.assert_array_equal(angle, np.zeros(3))
        np.testing.assert_array_equal(pixel, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
