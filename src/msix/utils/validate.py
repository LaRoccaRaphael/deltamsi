"""
Lightweight validators reused across msix.

Conventions
-----------
- Parameter misuse → ValueError
- Structural/data-shape issues → ShapeMismatchError
- Name conflicts (e.g., layers) → DuplicateLayerError
- ROI issues → InvalidROIError

Keep this file stdlib + NumPy only (no heavy deps).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .errors import (
    DuplicateLayerError,
    InvalidROIError,
    ShapeMismatchError,
)

__all__ = [
    "ppm",
    "ppm_tol",
    "ppm_window",
    "coords",
    "shape_match",
    "layer_name",
    "mz_sorted_unique",
    "mode",
    "finite",
    "nonnegative",
    "roi",
    "_assert",
]


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


# --------------------------- basic numeric params ---------------------------


def ppm(v: float) -> float:
    """Validate a generic parts-per-million value (> 0 and finite)."""
    if not np.isfinite(v):
        raise ValueError("ppm must be finite")
    if v <= 0:
        raise ValueError("ppm must be > 0")
    return float(v)


def ppm_tol(v: float, *, max_ppm: float = 1000.0) -> float:
    """
    Validate a tolerance in ppm: 0 < v <= max_ppm (default 1000 ppm).
    """
    v = ppm(v)
    if v > max_ppm:
        raise ValueError(f"ppm tolerance too large: {v} > {max_ppm}")
    return v


def ppm_window(mz: float, tol_ppm: float) -> tuple[float, float]:
    """
    Return absolute Da window [lo, hi] for an m/z and tolerance in ppm.
    """
    if not np.isfinite(mz) or mz <= 0:
        raise ValueError("mz must be finite and > 0")
    tol_ppm = ppm(tol_ppm)
    delta = mz * tol_ppm * 1e-6
    return mz - delta, mz + delta


# --------------------------- array/table structure --------------------------


def coords(spatial: np.ndarray, n_obs: int) -> np.ndarray:
    """
    Validate `.obsm['spatial']`-like coordinates: shape (n_obs, 2), finite.

    Returns a float32 view/copy suitable for storage.
    """
    arr = np.asarray(spatial)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] != n_obs:
        raise ShapeMismatchError(
            expected=(n_obs, 2), actual=tuple(arr.shape), context='obsm["spatial"]'
        )
    if not np.isfinite(arr).all():
        raise ValueError('obsm["spatial"] contains non-finite values')
    return arr.astype(np.float32, copy=False)


def shape_match(
    arr_shape: Sequence[int],
    n_obs: int,
    n_vars: int,
    *,
    context: str = "X",
    allow_transpose: bool = False,
) -> None:
    """
    Ensure a matrix matches (n_obs, n_vars). If `allow_transpose=True`, accept (n_vars, n_obs)
    but raise to force callers to correct it explicitly (prevents silent mistakes).
    """
    expected = (n_obs, n_vars)
    actual = tuple(arr_shape)
    if actual == expected:
        return
    if allow_transpose and actual == (n_vars, n_obs):
        raise ShapeMismatchError(
            expected=expected, actual=actual, context=f"{context} (looks transposed)"
        )
    raise ShapeMismatchError(expected=expected, actual=actual, context=context)


def layer_name(name: str, existing: Iterable[str] | None = None) -> str:
    """
    Validate a layer name: non-empty, simple token, not already present.
    Returns the normalized name (stripped).
    """
    import re

    if not isinstance(name, str):
        raise ValueError("layer name must be a string")
    s = name.strip()
    if not s:
        raise ValueError("layer name cannot be empty/whitespace")
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$", s):
        raise ValueError(
            "layer name must start with [A-Za-z0-9] and contain only [A-Za-z0-9_.-]"
        )
    if existing is not None and s in set(existing):
        raise DuplicateLayerError(s)
    return s


def mz_sorted_unique(mz: np.ndarray, *, strict: bool = True) -> np.ndarray:
    """
    Validate a 1D m/z axis:
    - finite and > 0
    - strictly increasing if strict=True, else returns a sorted unique copy.

    Returns float64 array (ascending).
    """
    arr = np.asarray(mz, dtype=np.float64).reshape(-1)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("mz must be a non-empty 1D array")
    if not np.isfinite(arr).all():
        raise ValueError("mz contains non-finite values")
    if (arr <= 0).any():
        raise ValueError("mz must be > 0")
    if strict:
        diffs = np.diff(arr)
        if (diffs <= 0).any():
            raise ValueError("mz must be strictly increasing when strict=True")
        return arr
    # non-strict: normalize to sorted unique
    return np.unique(arr)


def mode(value: str) -> str:
    """
    Validate imzML mode: 'profile' or 'centroid' (case-insensitive).
    Returns the normalized lower-case value.
    """
    if not isinstance(value, str):
        raise ValueError("mode must be a string")
    v = value.strip().lower()
    if v not in {"profile", "centroid"}:
        raise ValueError("mode must be 'profile' or 'centroid'")
    return v


def finite(arr: np.ndarray, *, name: str = "array") -> None:
    """
    Ensure all values are finite (no NaN/Inf).
    """
    a = np.asarray(arr)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")


def nonnegative(arr: np.ndarray, *, name: str = "array") -> None:
    """
    Ensure all values are >= 0.
    """
    a = np.asarray(arr)
    if (a < 0).any():
        raise ValueError(f"{name} contains negative values")


# ---------------------------------- ROI -------------------------------------


def roi(
    mask_or_bbox: np.ndarray | Sequence[int],
    height: int,
    width: int,
) -> np.ndarray:
    """
    Validate/normalize a Region Of Interest into a boolean mask of shape (height, width).

    Accepted inputs:
      - Boolean mask array of shape (height, width)
      - BBox as (y0, x0, y1, x1) in pixel indices (0-based, exclusive end)

    Returns
    -------
    mask : np.ndarray (bool, shape=(height, width))
    """
    H, W = int(height), int(width)
    if H <= 0 or W <= 0:
        raise InvalidROIError(reason="non-positive canvas size", bounds=(H, W))

    # Case 1: provided mask
    arr = np.asarray(mask_or_bbox)
    if arr.ndim == 2 and arr.shape == (H, W):
        if arr.dtype != np.bool_:
            # allow integer masks but validate values are 0/1
            if not np.array_equal(arr, arr.astype(bool)):
                raise InvalidROIError(reason="mask has non-binary values", shape=(H, W))
            arr = arr.astype(bool)
        return arr

    # Case 2: bbox = (y0, x0, y1, x1)
    if arr.ndim == 1 and arr.size == 4:
        try:
            y0, x0, y1, x1 = [int(v) for v in arr.tolist()]
        except Exception as e:  # noqa: BLE001
            raise InvalidROIError(reason="bbox must be four integers") from e
        if not (0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
            raise InvalidROIError(
                reason="bbox out of bounds or empty",
                shape=(H, W),
                bounds=(y1 - 1, x1 - 1),
            )
        mask = np.zeros((H, W), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask

    raise InvalidROIError(
        reason="ROI must be a (H,W) mask or bbox (y0,x0,y1,x1)", shape=(H, W)
    )
