import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Union


LABEL_MAP: dict[str, int] = {
    "rest":               0,
    "task_motor":         1,
    "task_story_math":    2,
    "task_working_memory": 3,
}

class MEGDataset(Dataset):
    """
    A PyTorch Dataset for MEG brain-decoding data stored as HDF5 files.

    Each .h5 file contains one recording matrix of shape
    (n_sensors=248, n_timesteps=35624).  This dataset builds a 
    sliding-window index over the time dimension of every loaded file, 
    so that each sample is a fixed-length multi-channel window paired 
    with the task-type label inferred from the filename.

    File naming convention expected on disk:
        <taskType>_<subjectID>_<chunk>.h5
    where taskType is one of: 
        [rest, task_motor, task_story_math, task_working_memory]
    """

    def __init__(
        self,
        data_dirs: Union[str, list[str]],
        window_size: int,
        stride: int,
        downsample_factor: int=1,
        dtype: torch.dtype=torch.float32,
        lazy: bool=False,
    )-> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}.")
        if downsample_factor < 1:
            raise ValueError(
                f"downsample_factor must be >= 1, got {downsample_factor}."
            )

        self.window_size = window_size
        self.stride = stride
        self.downsample_factor = downsample_factor
        self.dtype = dtype
        self.lazy = lazy

        # Normalisation statistics – set via fit_normalisation().
        self.mean: Optional[np.ndarray] = None  # shape (n_sensors, 1)
        self.std: Optional[np.ndarray] = None  # shape (n_sensors, 1)

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self._file_paths: list[str] = []
        self._file_labels: list[int] = []

        for directory in data_dirs:
            for fname in sorted(os.listdir(directory)):
                if not fname.endswith(".h5"):
                    continue
                label = self._parse_label(fname)
                self._file_paths.append(os.path.join(directory, fname))
                self._file_labels.append(label)

        if not self._file_paths:
            raise FileNotFoundError(
                f"No .h5 files found in: {data_dirs}"
            )

        # normal loading: read all matrices into RAM.
        # Lazy loading: only store file paths, open on demand.
        self._matrices: list[Optional[np.ndarray]] = []

        if not lazy:
            for path in self._file_paths:
                matrix = self._load_file(path)
                self._matrices.append(matrix)
        else:
            probe = self._load_file(self._file_paths[0])
            self._matrices = [None] * len(self._file_paths)
            # cache first file
            self._matrices[0] = probe

        self._index: list[tuple[int, int]] = []

        for file_idx, path in enumerate(self._file_paths):
            if self._matrices[file_idx] is not None:
                T = self._matrices[file_idx].shape[1]
            else:
                # Lazy: open just for the shape.
                with h5py.File(path, "r") as f:
                    name = self._get_dataset_name(path)
                    T = f[name].shape[1]
                T = T // downsample_factor

            n_windows = max(0, (T - window_size) // stride + 1)
            for t in range(0, n_windows * stride, stride):
                self._index.append((file_idx, t))

    @staticmethod
    def _get_dataset_name(filepath: str)-> str:
        """
        Reproduce the naming convention:
        strip the directory, then drop the trailing '_<chunk>.h5' part.
        E.g.  'rest_105923_1.h5'  ->  'rest_105923'
        """
        fname = os.path.basename(filepath) # 'rest_105923_1.h5'
        no_ext = fname[: fname.rfind(".")] # 'rest_105923_1'
        return "_".join(no_ext.split("_")[:-1]) # 'rest_105923'

    @staticmethod
    def _parse_label(filename: str)-> int:
        """
        Map a filename to an integer class label using LABEL_MAP.
        """
        for task in sorted(LABEL_MAP, key=len, reverse=True):
            if filename.startswith(task):
                return LABEL_MAP[task]
        raise ValueError(
            f"Cannot determine task label from filename: '{filename}'. "
            f"Expected one of {list(LABEL_MAP.keys())}."
        )

    def _load_file(self, path: str)-> np.ndarray:
        """
        Load one .h5 file and return a float32 array of shape
        (n_sensors, T_downsampled).
        """
        name = self._get_dataset_name(path)
        with h5py.File(path, "r") as f:
            matrix = f[name][()].astype(np.float32)
        if self.downsample_factor > 1:
            matrix = matrix[:, ::self.downsample_factor]
        return matrix

    def _get_matrix(self, file_idx: int)-> np.ndarray:
        """Return the matrix for file_idx, loading lazily if necessary."""
        if self._matrices[file_idx] is None:
            self._matrices[file_idx] = self._load_file(
                self._file_paths[file_idx]
            )
        return self._matrices[file_idx]

    # NOTE: this one is better but does not fit in ram.
    # def fit_normalisation(self, indices: list[int]) -> None:
    #     """
    #     Compute per-channel (per-sensor) mean and standard deviation from
    #     the windows identified by *indices* (which should be training
    #     indices only).  Sets `self.mean` and `self.std`, both shaped
    #     `(n_sensors, 1)`.

    #     :param indices: Indices in the partition used to normalise.
    #     :type indices: list[int]
    #     """
    #     n_sensors = self._get_matrix(0).shape[0]
    #     accumulator = []

    #     for i in indices:
    #         file_idx, t_start = self._index[i]
    #         matrix = self._get_matrix(file_idx)
    #         window = matrix[:, t_start : t_start + self.window_size]
    #         accumulator.append(window)

    #     stacked = np.concatenate(accumulator, axis=1)
    #     self.mean = stacked.mean(axis=1, keepdims=True).astype(np.float32)
    #     self.std = stacked.std(axis=1,  keepdims=True).astype(np.float32)

    #     zero_std = np.where(self.std == 0)[0]
    #     if zero_std.size > 0:
    #         raise ValueError(
    #             f"Standard deviation is zero for sensors: {zero_std.tolist()}."
    #             " Cannot normalise."
    #         )

    def fit_normalisation(self, indices: list[int]) -> None:
        """
        Compute per-channel (per-sensor) mean and standard deviation from
        the windows identified by *indices* (which should be training
        indices only).  Sets `self.mean` and `self.std`, both shaped
        `(n_sensors, 1)`.

        :param indices: Indices in the partition used to normalise.
        :type indices: list[int]
        """
        n_sensors = self._get_matrix(0).shape[0]

        # Accumulate per-channel sum and sum-of-squares in float64 for precision.
        running_sum = np.zeros((n_sensors, 1), dtype=np.float64)
        running_ssq = np.zeros((n_sensors, 1), dtype=np.float64)
        running_n   = 0

        for i in indices:
            file_idx, t_start = self._index[i]
            window = self._get_matrix(file_idx)[:, t_start : t_start + self.window_size]
            running_sum += window.sum(axis=1, keepdims=True)
            running_ssq += (window ** 2).sum(axis=1, keepdims=True)
            running_n   += self.window_size

        mean = running_sum / running_n
        var  = running_ssq / running_n - mean ** 2
        std  = np.sqrt(np.maximum(var, 0.0))  # clamp prevents sqrt of tiny negatives from float arithmetic

        zero_std = np.where(std == 0)[0]
        if zero_std.size > 0:
            raise ValueError(f"Zero std for sensors: {zero_std.tolist()}")

        self.mean = mean.astype(np.float32)
        self.std  = std.astype(np.float32)

    def get_n_sensors(self)-> int:
        """
        Get the number of MEG sensor channels (rows) in each window.
        """
        return self._get_matrix(0).shape[0]

    def __len__(self)-> int:
        return len(self._index)

    def __getitem__(
        self, idx: int
    )-> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a (window, label) pair.

        window : torch.Tensor of shape (n_sensors, window_size)
            The raw (or normalised) multi-channel time-series window.
        label : torch.Tensor scalar (long)
            Integer class label in {0, 1, 2, 3}.
        """
        file_idx, t_start = self._index[idx]
        matrix = self._get_matrix(file_idx)
        window = matrix[:, t_start : t_start + self.window_size].copy()

        if self.mean is not None and self.std is not None:
            window = (window - self.mean) / self.std

        label = self._file_labels[file_idx]

        return (
            torch.tensor(window, dtype=self.dtype),
            torch.tensor(label, dtype=torch.long),
        )

    def __repr__(self)-> str:
        return (
            f"MEGDataset("
            f"n_files={len(self._file_paths)}, "
            f"n_windows={len(self)}, "
            f"window_size={self.window_size}, "
            f"stride={self.stride}, "
            f"downsample_factor={self.downsample_factor}, "
            f"lazy={self.lazy})"
        )