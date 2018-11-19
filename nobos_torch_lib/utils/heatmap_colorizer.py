from typing import List

import numpy as np


def colorize_heatmaps(self, heatmaps: np.ndarray, color: List[int], channels: int = 3):
    color_array = np.array([[[color]]])
    clipped = np.clip(heatmaps, 0, 1)
    clipped = np.squeeze(clipped, axis=0)
    clipped = clipped[:, :, :, None] * np.ones(channels, dtype=int)[None, None, None, :]
    color_map = clipped * color_array
    return color_map


def colorize_heatmaps_by_scheme(self, heatmaps: np.ndarray, color_scheme: np.ndarray, channels: int = 3):
    clipped = np.clip(heatmaps, 0, 1)
    clipped = clipped[:, :, :, :, None] * np.ones(channels, dtype=int)[None, None, None, None, :]
    color_map = np.matmul(clipped, color_scheme)
    return color_map