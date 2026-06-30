"""Read and write discriminatory-instance arrays and run logs."""
import json

import numpy as np
import pandas as pd


def save_idi_rows(path, rows):
    """Save integer rows as a compact ``.npy`` array without overflow."""
    arr = np.asarray(rows, dtype=np.int64)
    if arr.size:
        peak = max(abs(int(arr.min())), abs(int(arr.max())))
        for dtype in (np.int16, np.int32):
            if peak <= np.iinfo(dtype).max:
                arr = arr.astype(dtype, copy=False)
                break
    np.save(path, arr)


def read_idi_rows(path, drop_last_col=True):
    """Load a current ``.npy`` or legacy CSV artifact.

    The trailing prediction column is dropped by default. Empty inputs return
    an array with shape ``(0, 0)``.
    """
    if str(path).endswith(".npy"):
        arr = np.load(path)
    else:
        try:
            arr = pd.read_csv(path, header=None).to_numpy()
        except pd.errors.EmptyDataError:
            return np.empty((0, 0), dtype=np.int64)
    if arr.ndim < 2 or arr.shape[0] == 0:
        return np.empty((0, 0), dtype=np.int64)
    if drop_last_col and arr.shape[1] > 0:
        arr = arr[:, :-1]
    return arr.astype(np.int64, copy=False)


def read_log(json_path):
    """Load one per-run JSON log written by run_cell."""
    with open(json_path) as fh:
        return json.load(fh)
