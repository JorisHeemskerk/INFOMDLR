import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union


class TimeseriesDataset(Dataset):
    """
    A PyTorch Dataset for sliding-window time series data.

    Supports loading from CSV/Parquet files or directly from a NumPy array /
    Pandas DataFrame.  The window length (and optional forecast horizon) can be
    changed at any time via `set_window()`.

    Args:
        source:       Path to a CSV/Parquet file, a DataFrame, or a NumPy array.
        window_size:  Number of time steps in each input window.
        horizon:      Number of future steps to use as the prediction target.
                      Set to 0 to return only the input window (no target).
        target_col:   Column name(s) to use as the prediction target.
                      If None, the whole window is returned as both input and target.
        feature_cols: Subset of columns to use as input features.
                      If None, all columns are used.
        stride:       Step size between consecutive windows (default 1).
        normalize:    If True, apply z-score normalization using stats computed
                      from the training split.
        train_ratio:  Fraction of data used for fitting the normalizer.
        dtype:        PyTorch dtype for the returned tensors (default float32).
    """

    def __init__(
        self,
        source: str,
        window_size: int,
        horizon: int = 1,
        stride: int = 1,
        normalize: bool,
        train_ratio: float = 0.8,
        dtype: torch.dtype = torch.float32,
    ):
        self.dtype = dtype
        self.horizon = horizon
        self.stride = stride
        self.normalize = normalize

        # ── Load data ──────────────────────────────────────────────────────────
        self._data_raw = self._load(source)

        # ── Select columns ─────────────────────────────────────────────────────
        if isinstance(self._data_raw, pd.DataFrame):
            if feature_cols is not None:
                self._features = self._data_raw[feature_cols].values.astype(np.float32)
            else:
                self._features = self._data_raw.select_dtypes(include=[np.number]).values.astype(np.float32)

            if target_col is not None:
                cols = [target_col] if isinstance(target_col, str) else target_col
                self._targets = self._data_raw[cols].values.astype(np.float32)
            else:
                self._targets = self._features  # same array → no copy
        else:
            # NumPy path
            arr = self._data_raw.astype(np.float32)
            self._features = arr
            self._targets = arr

        # ── Optional normalisation (fit on train split only) ───────────────────
        if normalize:
            n_train = int(len(self._features) * train_ratio)
            self._mean = self._features[:n_train].mean(axis=0)
            self._std  = self._features[:n_train].std(axis=0) + 1e-8
            self._features = (self._features - self._mean) / self._std
            if self._targets is not self._features:
                t_mean = self._targets[:n_train].mean(axis=0)
                t_std  = self._targets[:n_train].std(axis=0) + 1e-8
                self._targets = (self._targets - t_mean) / t_std

        # ── Set the window (also builds the index list) ────────────────────────
        self.set_window(window_size)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_window(self, window_size: int, stride: Optional[int] = None) -> None:
        """
        Change the window size (and optionally the stride) on-the-fly.

        This rebuilds the internal index without reloading or re-normalising
        the underlying data.

        Args:
            window_size: New number of time steps per sample.
            stride:      New stride between windows.  If None, keeps current value.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be ≥ 1, got {window_size}.")
        self.window_size = window_size
        if stride is not None:
            self.stride = stride

        total_needed = window_size + self.horizon
        n = len(self._features)
        if total_needed > n:
            raise ValueError(
                f"window_size ({window_size}) + horizon ({self.horizon}) = "
                f"{total_needed} exceeds data length ({n})."
            )

        # Build the list of valid start indices
        self._indices = list(range(0, n - total_needed + 1, self.stride))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self._indices[idx]
        end   = start + self.window_size

        x = torch.tensor(self._features[start:end], dtype=self.dtype)

        if self.horizon > 0:
            y = torch.tensor(self._targets[end:end + self.horizon], dtype=self.dtype)
        else:
            y = x.clone()          # unsupervised / auto-encoder mode

        return x, y

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        """Number of input feature dimensions."""
        return self._features.shape[1] if self._features.ndim > 1 else 1

    @property
    def n_targets(self) -> int:
        """Number of target dimensions."""
        return self._targets.shape[1] if self._targets.ndim > 1 else 1

    @property
    def shape(self) -> Tuple[int, int, int]:
        """(n_samples, window_size, n_features) — useful for model initialisation."""
        return (len(self), self.window_size, self.n_features)

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _load(source: Union[str, Path, pd.DataFrame, np.ndarray]):
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix in {".parquet", ".pq"}:
                return pd.read_parquet(path)
            elif path.suffix in {".csv", ".tsv", ".txt"}:
                sep = "\t" if path.suffix in {".tsv", ".txt"} else ","
                return pd.read_csv(path, sep=sep)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
        elif isinstance(source, pd.DataFrame):
            return source
        elif isinstance(source, np.ndarray):
            return source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    def __repr__(self) -> str:
        return (
            f"TimeSeriesDataset("
            f"samples={len(self)}, "
            f"window_size={self.window_size}, "
            f"horizon={self.horizon}, "
            f"features={self.n_features}, "
            f"targets={self.n_targets})"
        )


# ── Convenience factory ────────────────────────────────────────────────────────

def make_splits(
    source: Union[str, Path, pd.DataFrame, np.ndarray],
    window_size: int = 64,
    horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    **dataset_kwargs,
) -> Tuple["TimeSeriesDataset", "TimeSeriesDataset", "TimeSeriesDataset"]:
    """
    Split a data source into train / validation / test datasets.

    The split is done on the raw data before any windowing so there is no
    leakage between splits.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Load once
    raw = TimeSeriesDataset._load(source)

    n = len(raw)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    if isinstance(raw, pd.DataFrame):
        train_src = raw.iloc[:n_train]
        val_src   = raw.iloc[n_train : n_train + n_val]
        test_src  = raw.iloc[n_train + n_val:]
    else:
        train_src = raw[:n_train]
        val_src   = raw[n_train : n_train + n_val]
        test_src  = raw[n_train + n_val:]

    train_ds = TimeSeriesDataset(train_src, window_size=window_size, horizon=horizon, **dataset_kwargs)
    val_ds   = TimeSeriesDataset(val_src,   window_size=window_size, horizon=horizon, **dataset_kwargs)
    test_ds  = TimeSeriesDataset(test_src,  window_size=window_size, horizon=horizon, **dataset_kwargs)

    return train_ds, val_ds, test_ds


# ── Quick demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic multivariate time series
    np.random.seed(42)
    t = np.linspace(0, 8 * np.pi, 2000)
    data = np.column_stack([
        np.sin(t) + 0.05 * np.random.randn(len(t)),
        np.cos(t) + 0.05 * np.random.randn(len(t)),
        np.sin(2 * t) * 0.5,
    ])
    df = pd.DataFrame(data, columns=["sin", "cos", "sin2"])

    # ── Build dataset and DataLoader ───────────────────────────────────────────
    ds = TimeSeriesDataset(
        source=df,
        window_size=64,
        horizon=16,
        target_col="sin",
        normalize=True,
    )
    print(ds)                              # TimeSeriesDataset(samples=…, …)
    print("Dataset shape:", ds.shape)     # (n_samples, window_size, n_features)

    x, y = ds[0]
    print(f"x: {x.shape}, y: {y.shape}") # x: [64, 3], y: [16, 1]

    loader = DataLoader(ds, batch_size=32, shuffle=True)
    xb, yb = next(iter(loader))
    print(f"Batch — x: {xb.shape}, y: {yb.shape}")

    # ── Change window size on-the-fly ──────────────────────────────────────────
    ds.set_window(128)
    print(f"\nAfter set_window(128): {ds}")

    ds.set_window(32, stride=8)
    print(f"After set_window(32, stride=8): {ds}")

    # ── Train/val/test split ───────────────────────────────────────────────────
    train, val, test = make_splits(df, window_size=64, horizon=16, target_col="sin")
    print(f"\nSplits — train: {len(train)}, val: {len(val)}, test: {len(test)}")