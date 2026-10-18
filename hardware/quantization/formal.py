"""Evaluate frozen 16-bit quantization on the formal L40S-trained models.

This reuses the existing BN/threshold folding and fixed-point arithmetic. CPU
float64 convolution holds exact integer sums (the worst-case bound is below
2**53), avoiding float32's limited integer range for newly trained checkpoints.
No training, calibration on test labels, or checkpoint modification occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from .fixed_point import (
    FixedPointPursuitReference,
    RuntimeCounters,
    load_model,
    output_channel_view,
    sha256_file,
)
from .metrics import (
    MM_PER_PX,
    SCREEN_CENTER_X_PX,
    SCREEN_CENTER_Y_PX,
    VIEW_DISTANCE_MM,
)


CONTRACT = dict(
    weight_bits=16, readout_weight_bits=16, constant_bits=24,
    accumulator_bits=24, spike_bits=8, scale_mode="per_output_channel",
    scale_policy="no_clipping", maximum_fraction_bits=15,
    state_guard_bits=0, alpha_rounding="floor",
)
GRID_ROWS = np.arange(150, 931, 78, dtype=np.float64)
GRID_COLS = np.arange(150, 1771, 162, dtype=np.float64)


class ExactFixedPointReference(FixedPointPursuitReference):
    """Keep the existing arithmetic, with exact double-precision dot products."""

    def _linear_or_convolution(self, source_spikes, layer, counters):
        source = source_spikes.to(torch.float64)
        weight = layer.weight_integer.to(torch.float64)
        bound = float(source.abs().max()) * float(
            weight.abs().reshape(weight.shape[0], -1).sum(dim=1).max()
        )
        if bound >= 2**53:
            raise ArithmeticError(f"{layer.name}: exceeds exact integer range")
        if weight.ndim == 4:
            current = F.conv2d(source, weight, padding=1)
        else:
            current = F.linear(source, weight)
        scale = torch.pow(2.0, layer.product_left_shift.to(torch.float64))
        current = current * output_channel_view(scale, current.ndim)
        current = current + output_channel_view(
            layer.constant_integer.to(torch.float64), current.ndim
        )
        current = self._saturate_accumulator(current, layer.name, counters)
        return F.max_pool2d(current, 2) if layer.pool else current

    def _lif(self, current_integer, previous_membrane_integer, layer, counters):
        spikes, membrane = super()._lif(
            current_integer, previous_membrane_integer, layer, counters
        )
        if membrane.min() < -32768 or membrane.max() > 32767:
            raise ArithmeticError(f"{layer.name}: membrane exceeds signed 16 bits")
        return spikes, membrane


def normalized_to_pixels(prediction):
    xy = np.asarray(prediction, dtype=np.float64)
    return np.column_stack((
        xy[:, 1] * SCREEN_CENTER_Y_PX + SCREEN_CENTER_Y_PX,
        xy[:, 0] * SCREEN_CENTER_X_PX + SCREEN_CENTER_X_PX,
    ))


def project_grid(prediction_px):
    return np.column_stack((
        GRID_ROWS[np.abs(prediction_px[:, :1] - GRID_ROWS).argmin(axis=1)],
        GRID_COLS[np.abs(prediction_px[:, 1:] - GRID_COLS).argmin(axis=1)],
    ))


def errors(prediction_px, labels):
    def rays(points):
        return np.column_stack((
            (points[:, 1] - SCREEN_CENTER_X_PX) * MM_PER_PX,
            (points[:, 0] - SCREEN_CENTER_Y_PX) * MM_PER_PX,
            np.full(len(points), VIEW_DISTANCE_MM),
        ))
    a, b = rays(prediction_px), rays(labels)
    angles = np.rad2deg(np.arctan2(
        np.linalg.norm(np.cross(a, b), axis=1), (a * b).sum(axis=1)
    ))
    pixels = np.linalg.norm(prediction_px - labels, axis=1)
    return angles, pixels


def means(pair):
    return dict(mean_angular_error_deg=float(pair[0].mean()),
                mean_pixel_error=float(pair[1].mean()))
