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
        stride: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        self._data = np.array(scipy.io.loadmat(source)["Xtrain"])
        
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        self.window_size = window_size

        self.stride = stride
        self.dtype = dtype

        total_needed = window_size + 1
        self._indices = list(
            range(0, len(self._data) - total_needed + 1, self.stride)
        )

        # Set these via the fit_normalisation()
        self.mean = None
        self.std = None

    def fit_normalisation(self, indices: list[int]) -> None:
        """
        Compute mean and std from the raw data windows of the given 
        indices (should be training indices only). Set self.mean and
        self.std.

        :param indices: Indices in the partition used to normalise.
        :type indices: list[int]
        """
        train_starts = [self._indices[i] for i in indices]
        train_data = np.concatenate([
            self._data[s : s + self.window_size + 1]
            for s in train_starts
        ])
        self.mean = float(train_data.mean())
        self.std  = float(train_data.std())
        if self.std == 0:
            raise ValueError("Standard deviation is zero; cannot normalise.")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self._indices[idx]
        end   = start + self.window_size
        x = self._data[start:end]
        y = self._data[end]

        # Apply normalisation if stats are available
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        return (
            torch.tensor(x, dtype=self.dtype),
            torch.tensor(y, dtype=self.dtype),
        )