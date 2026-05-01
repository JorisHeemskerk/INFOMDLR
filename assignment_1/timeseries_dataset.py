"""
DISCLAIMER: 
This code was previously part of Joris Heemskerk's & Bas de Blok's prior
work for the Computer Vision course, and is being re-used here.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple
import scipy.io


class TimeseriesDataset(Dataset):
    """
    A PyTorch Dataset for sliding-window time series data.
    """

    def __init__(
        self,
        source: str,
        window_size: int,
        horizon: int = 1,
        stride: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        self._data = np.array(scipy.io.loadmat(source)["Xtrain"])
        
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        self.window_size = window_size
        
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        self.horizon = horizon

        self.stride = stride
        self.dtype = dtype

        total_needed = window_size + self.horizon
        self._indices = list(range(0, len(self._data) - total_needed + 1, self.stride))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self._indices[idx]
        end   = start + self.window_size
        x = torch.tensor(self._data[start:end], dtype=self.dtype)
        y = torch.tensor(self._data[end:end + self.horizon], dtype=self.dtype)
        return x, y
