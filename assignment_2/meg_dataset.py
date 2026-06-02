import os
import h5py
import torch
import numpy as np
import logging

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
        ommited_sensors: list[int],
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

        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None 

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
        
        self._sensor_mask = np.ones(
            self._peek_n_sensors(self._file_paths[0]),
            dtype=bool
        )
        self._sensor_mask[ommited_sensors] = False

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
        fname = os.path.basename(filepath)
        no_ext = fname[:fname.rfind(".")]
        return "_".join(no_ext.split("_")[:-1])

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

    @staticmethod
    def _get_chunk_number(path: str) -> int:
        """Extract the trailing chunk number from a filename."""
        fname = os.path.basename(path)
        no_ext = fname[:fname.rfind(".")]
        return int(no_ext.split("_")[-1])
    
    @staticmethod
    def _peek_n_sensors(path: str) -> int:
        """Read sensor count"""
        name = MEGDataset._get_dataset_name(path)
        with h5py.File(path, "r") as f:
            return f[name].shape[0]

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
        return matrix[self._sensor_mask]

    def _get_matrix(self, file_idx: int)-> np.ndarray:
        """
        Return the matrix for file_idx, loading lazily if necessary.
        """
        if self._matrices[file_idx] is None:
            self._matrices[file_idx] = self._load_file(
                self._file_paths[file_idx]
            )
        return self._matrices[file_idx]

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

        running_sum = np.zeros((n_sensors, 1), dtype=np.float64)
        running_ssq = np.zeros((n_sensors, 1), dtype=np.float64)
        running_n = 0

        for i in indices:
            file_idx, t_start = self._index[i]
            window = self._get_matrix(
                file_idx
            )[:, t_start : t_start + self.window_size]
            running_sum += window.sum(axis=1, keepdims=True)
            running_ssq += (window ** 2).sum(axis=1, keepdims=True)
            running_n += self.window_size

        mean = running_sum / running_n
        var = running_ssq / running_n - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0))

        zero_std = np.where(std == 0)[0]
        if zero_std.size > 0:
            raise ValueError(f"Zero std for sensors: {zero_std.tolist()}")

        self.mean = mean.astype(np.float32)
        self.std  = std.astype(np.float32)

    def get_fold_indices(
        self, 
        current_k: int, 
        total_k: int,
        logger: logging.Logger | None
    ) -> tuple[list[int], list[int]]:
        """
        Return train and validation window indices for one fold of
        file-level k-fold cross validation.

        Files are distributed across folds as evenly as possible.  All
        windows that belong to the files assigned to *current_k* become
        the validation set; all other windows become the training set.

        :param current_k: Zero-based index of the current fold 
            (0 … total_k-1).
        :type current_k: int
        :param total_k: Total number of folds.
        :type total_k: int
        :param logger: Logger to log to.
        :type logger: logging.Logger | None
        :return: (train_indices, val_indices) indices into self._index.
        :rtype: tuple[list[int], list[int]]
        """
        if not (0 <= current_k < total_k):
            raise ValueError(
                f"current_k must be in [0, total_k), "
                f"got current_k={current_k}, total_k={total_k}."
            )
        n_files = len(self._file_paths)
        if total_k > n_files:
            raise ValueError(
                f"total_k ({total_k}) cannot exceed the number of files "
                f"({n_files})."
            )

        file_to_fold: list[int] = [
            (self._get_chunk_number(p) - 1) % total_k
            for p in self._file_paths
        ]
        val_files: set[int] = {
            i for i, fold in enumerate(file_to_fold) if fold == current_k
        }

        if logger is not None:
            for i, path in enumerate(self._file_paths):
                fname = os.path.basename(path)
                if i in val_files:
                    logger.debug(
                        f"Fold {current_k + 1}/{total_k} - validation: {fname}"
                    )
                else:
                    logger.debug(
                        f"Fold {current_k + 1}/{total_k} - train:      {fname}"
                    )

        train_idx: list[int] = []
        val_idx: list[int] = []
        for win_idx, (file_idx, _) in enumerate(self._index):
            if file_idx in val_files:
                val_idx.append(win_idx)
            else:
                train_idx.append(win_idx)

        return train_idx, val_idx

    def get_person_fold_indices(
        self,
        current_k: int,
        total_k: int,
        logger: logging.Logger | None,
    ) -> tuple[list[int], list[int]]:
        """
        Return train and validation window indices for one fold of
        person-level k-fold cross validation.

        Unique subjects (inferred from the subject-ID segment of each
        filename) are distributed across folds as evenly as possible.
        All windows that belong to files of the subjects assigned to
        `current_k` become the validation set. All other windows become
        the training set.

        :param current_k: Zero-based index of the current fold.
        :type current_k: int
        :param total_k: Total number of folds.
        :type total_k: int
        :param logger: Logger to log to.
        :type logger: logging.Logger | None
        :return: (train_indices, val_indices) indices into self._index.
        :rtype: tuple[list[int], list[int]]
        """
        if not (0 <= current_k < total_k):
            raise ValueError(
                f"current_k must be in [0, total_k), "
                f"got current_k={current_k}, total_k={total_k}."
            )

        def _get_subject_id(path: str) -> str:
            return self._get_dataset_name(path).split("_")[-1]

        subjects: list[str] = sorted(
            {_get_subject_id(p) for p in self._file_paths}
        )
        n_subjects = len(subjects)

        if total_k != n_subjects:
            raise NotImplementedError(
                f"total_k ({total_k}) must be equal to the number of unique "
                f"subjects ({n_subjects})."
            )

        subject_to_fold: dict[str, int] = {
            s: idx % total_k for idx, s in enumerate(subjects)
        }
        val_subjects: set[str] = {
            s for s, fold in subject_to_fold.items() if fold == current_k
        }

        if logger is not None:
            for path in self._file_paths:
                fname = os.path.basename(path)
                subject = _get_subject_id(path)
                if subject in val_subjects:
                    logger.debug(
                        f"Fold {current_k + 1}/{total_k} - validation: {fname}"
                    )
                else:
                    logger.debug(
                        f"Fold {current_k + 1}/{total_k} - train:      {fname}"
                    )

        train_idx: list[int] = []
        val_idx: list[int] = []
        for win_idx, (file_idx, _) in enumerate(self._index):
            subject = _get_subject_id(self._file_paths[file_idx])
            if subject in val_subjects:
                val_idx.append(win_idx)
            else:
                train_idx.append(win_idx)

        return train_idx, val_idx

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
