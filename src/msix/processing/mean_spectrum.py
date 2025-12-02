# msix/core/spectrum/mean.py
"""
Mean spectrum computation (streaming, memory-safe).

Two execution paths:
1) Profile imzML:
   - Build a bin grid using either ppm-based variable-width bins or fixed Da.
   - Stream spectra; optionally TIC-normalize per spectrum.
   - Weighted histogram accumulation per spectrum, then average across spectra.

2) Centroid imzML:
   - Sample spectra and merge their peaks into a compact axis (ppm merge).
   - Stream all spectra; for each peak, spread intensity to nearby axis points
     with a Gaussian kernel whose sigma is set in ppm (resolution-aware).
   - Optionally TIC-normalize per spectrum before accumulation.

Both paths return:
   (mz: float64 array, intensity: float32 array) for the mean spectrum.

Notes
-----
- We keep this module free of SciPy so your env remains lightweight.
- 'median_mean' aggregation is approximated by simple mean for now (TODO).
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Literal, List

import numpy as np

from ..params.options import (
    BinningParams,
    MeanSpecParams,
    validate_binning,
    validate_mean,
)
from ..io import imzml as imio


__all__ = [
    "compute_mean_spectrum",
    "compute_mean_spectrum_profile",
    "compute_mean_spectrum_centroid",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _ppm_to_da(mz: np.ndarray | float, ppm: float) -> np.ndarray | float:
    return (np.asarray(mz) * (ppm * 1e-6)).astype(float)


def _safe_tic(intens: np.ndarray) -> float:
    s = float(np.sum(intens, dtype=np.float64))
    return s if s > 0.0 else 1.0


def _winsorize_inplace(y: np.ndarray, pct: float) -> None:
    """Winsorize an array in-place by clipping at [p, 100-p] percentiles."""
    if pct <= 0.0:
        return
    lo = np.percentile(y, pct)
    hi = np.percentile(y, 100.0 - pct)
    np.clip(y, lo, hi, out=y)


# -------------------------- Profile: grid construction ----------------------


def _grid_edges_ppm(mz_min: float, mz_max: float, ppm: float) -> np.ndarray:
    """Construct variable-width bin edges based on ppm spacing."""
    if mz_max <= mz_min:
        raise ValueError("mz_max must be > mz_min")
    if ppm <= 0:
        raise ValueError("ppm must be > 0")

    edges: List[float] = [float(mz_min)]
    # Iteratively step in multiplicative ppm increments
    # m_{k+1} = m_k * (1 + ppm * 1e-6)
    m = float(mz_min)
    factor = 1.0 + ppm * 1e-6
    # guard against infinite loops in pathological settings
    max_bins = 50_000_000  # practical hard cap
    while m < mz_max and len(edges) < max_bins:
        m *= factor
        edges.append(m)
    if edges[-1] < mz_max:
        edges.append(float(mz_max))
    return np.asarray(edges, dtype=np.float64)


def _grid_edges_da(mz_min: float, mz_max: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Da step must be > 0")
    # Include right edge
    n = int(np.ceil((mz_max - mz_min) / step))
    edges = mz_min + np.arange(n + 1, dtype=np.float64) * step
    if edges[-1] < mz_max:
        edges = np.append(edges, mz_max)
    return edges


def _centers_from_edges(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) / 2.0


# -------------------------- Centroid: axis construction ---------------------


def _merge_peaks_ppm(mz: np.ndarray, intens: np.ndarray, tol_ppm: float) -> np.ndarray:
    """
    Merge peaks within tol_ppm (intensity-weighted center). Returns merged mz.
    """
    if mz.size == 0:
        return mz
    order = np.argsort(mz)
    mz = mz[order].astype(np.float64, copy=False)
    w = intens[order].astype(np.float64, copy=False)

    merged: List[float] = []
    cur_m = mz[0]
    cur_w = w[0]
    cur_sum = w[0]

    for j in range(1, mz.size):
        m = mz[j]
        win_da = _ppm_to_da(cur_m, tol_ppm)  # use current cluster center for tol
        if abs(m - cur_m) <= float(win_da):
            # merge into current cluster (weight by intensity)
            cur_sum += w[j]
            cur_m = (cur_m * cur_w + m * w[j]) / (cur_w + w[j])
            cur_w += w[j]
        else:
            merged.append(float(cur_m))
            cur_m = m
            cur_w = w[j]
            cur_sum = w[j]
    merged.append(float(cur_m))
    return np.asarray(merged, dtype=np.float64)


def _build_union_axis_from_sample(
    spectra: Iterable[Tuple[int, np.ndarray, np.ndarray]],
    sample_size: int,
    tol_ppm: float,
    normalize_tic: bool,
) -> np.ndarray:
    """
    Collect peaks from up to `sample_size` spectra and merge to a compact axis.
    """
    mz_all: List[np.ndarray] = []
    int_all: List[np.ndarray] = []
    count = 0
    for _, mz, intens in spectra:
        if mz.size == 0:
            continue
        if normalize_tic:
            intens = intens / _safe_tic(intens)
        mz_all.append(np.asarray(mz, np.float64))
        int_all.append(np.asarray(intens, np.float64))
        count += 1
        if count >= sample_size:
            break
    if not mz_all:
        return np.array([], dtype=np.float64)

    mz_cat = np.concatenate(mz_all)
    int_cat = np.concatenate(int_all)
    # Pre-merge with a relatively tight tolerance (≈ KDE sigma by default)
    return _merge_peaks_ppm(mz_cat, int_cat, tol_ppm=tol_ppm)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_mean_spectrum(
    imzml_path: str,
    *,
    mode: Literal["profile", "centroid"],
    binning: Optional[BinningParams] = None,
    params: Optional[MeanSpecParams] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dispatcher that computes a mean spectrum depending on the imzML mode.

    Parameters
    ----------
    imzml_path : str
        Path to the input .imzML
    mode : {"profile","centroid"}
        Acquisition mode (explicit, validated earlier).
    binning : BinningParams, optional
        Required for profile; ignored for centroid (KDE-like path).
    params : MeanSpecParams, optional
        Controls normalization and (for centroid) KDE sigma/width.

    Returns
    -------
    mz : float64 ndarray
    mean_intensity : float32 ndarray
    """
    mp = params or MeanSpecParams()
    validate_mean(mp)

    if mode == "profile":
        if binning is None:
            binning = BinningParams()  # defaults: ppm, 5 ppm
        validate_binning(binning)
        return compute_mean_spectrum_profile(imzml_path, binning=binning, params=mp)

    # centroid path
    return compute_mean_spectrum_centroid(imzml_path, params=mp)


# ------------------------ Profile imzML: streaming mean ---------------------


def compute_mean_spectrum_profile(
    imzml_path: str,
    *,
    binning: BinningParams,
    params: MeanSpecParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean spectrum from profile data by binning to a grid and averaging.

    Strategy
    --------
    - Determine (mz_min, mz_max) either from options or a quick scan.
    - Build bin edges: ppm or Da.
    - For each spectrum:
        * Optionally TIC-normalize intensities.
        * Optionally winsorize intensities globally (approximate robustness).
        * Accumulate a weighted histogram with weights=intens.
    - Divide by the number of spectra processed (simple mean).

    Returns
    -------
    centers : float64  (bin centers)
    mean_y  : float32  (mean intensity per bin)
    """
    validate_binning(binning)
    validate_mean(params)

    # Probe span if needed
    p = imio.open_parser(imzml_path)
    try:
        mz_lo, mz_hi = imio.quick_mz_span(p, k=16)
    finally:
        imio.close_parser(p)

    mz_min = float(binning.mz_min if binning.mz_min is not None else mz_lo)
    mz_max = float(binning.mz_max if binning.mz_max is not None else mz_hi)

    if binning.mode == "ppm":
        edges = _grid_edges_ppm(mz_min, mz_max, binning.bin_width)
    else:
        edges = _grid_edges_da(mz_min, mz_max, binning.bin_width)

    centers = _centers_from_edges(edges)
    acc = np.zeros((centers.size,), dtype=np.float64)
    n_spec = 0

    for _, mz, intens in imio.iter_spectra(imzml_path):
        if mz.size == 0:
            continue
        y = intens.astype(np.float64, copy=False)

        if params.normalize == "tic":
            y = y / _safe_tic(y)

        # Optional (approximate) winsorization: global on the spectrum
        if params.winsor_pct > 0:
            y = y.copy()
            _winsorize_inplace(y, params.winsor_pct)

        # Weighted histogram into our bin edges
        h, _ = np.histogram(mz, bins=edges, weights=y)
        acc += h
        n_spec += 1

    if n_spec == 0:
        return centers, np.zeros_like(centers, dtype=np.float32)

    mean_y = (acc / n_spec).astype(np.float32)
    return centers, mean_y


# ----------------------- Centroid imzML: KDE-like mean ----------------------


def compute_mean_spectrum_centroid(
    imzml_path: str,
    *,
    params: MeanSpecParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean spectrum from centroid data using resolution-aware Gaussian smear.

    Strategy
    --------
    1) Build a compact union axis:
       - sample up to `params.sample or 256` spectra,
       - normalize per spectrum by TIC if requested,
       - merge peaks within `kde_sigma_ppm` to form a tidy axis.
    2) Stream all spectra:
       - for each peak (m, I), compute sigma_da = m * kde_sigma_ppm * 1e-6,
       - find axis indices with |axis - m| <= k * sigma_da,
       - add I * exp(-0.5 * ((axis - m)/sigma_da)^2) to accumulator,
       - also accumulate the kernel weights (for a normalized average).
    3) Return acc / weights.

    Returns
    -------
    axis_mz : float64
    mean_y  : float32
    """
    validate_mean(params)

    # Determine sample size for axis building
    # If sample is None: default to n_pixels
    p = imio.open_parser(imzml_path)
    try:
        n_pixels = len(getattr(p, "coordinates", []))
    finally:
        imio.close_parser(p)
    sample_n = int(params.sample) if params.sample is not None else n_pixels
    print(n_pixels, sample_n)

    # Build union axis from a sample
    rng = np.random.default_rng(0)
    # Random, but deterministic: create an index stream; avoid reading whole file just to sample
    # We iterate all spectra but only keep the first `sample_n` hits uniformly
    # Simpler: single pass reservoir sampling
    kept: List[Tuple[int, np.ndarray, np.ndarray]] = []
    i_seen = 0
    for i, mz, intens in imio.iter_spectra(imzml_path):
        i_seen += 1
        if len(kept) < sample_n:
            kept.append((i, mz, intens))
        else:
            j = rng.integers(0, i_seen)
            if j < len(kept):
                kept[j] = (i, mz, intens)
    axis = _build_union_axis_from_sample(
        ((i, mz, intens) for (i, mz, intens) in kept),
        sample_size=len(kept),
        tol_ppm=float(params.kde_sigma_ppm),
        normalize_tic=(params.normalize == "tic"),
    )
    # If still empty, return empty mean
    if axis.size == 0:
        return axis, axis.astype(np.float32)

    axis.sort()
    acc = np.zeros_like(axis, dtype=np.float64)
    wsum = np.zeros_like(axis, dtype=np.float64)

    k = float(params.kde_half_width_sigma)
    sigma_ppm = float(params.kde_sigma_ppm)

    for _, mz, intens in imio.iter_spectra(imzml_path):
        if mz.size == 0:
            continue
        y = intens.astype(np.float64, copy=False)
        if params.normalize == "tic":
            y = y / _safe_tic(y)

        # For each peak, smear to nearby axis points
        for m_val, inten in zip(mz, y):
            if inten <= 0.0:
                continue
            m_val = float(m_val)
            sigma_da = float(_ppm_to_da(m_val, sigma_ppm))
            if sigma_da <= 0.0 or not np.isfinite(sigma_da):
                continue
            window_half = k * sigma_da
            idx_left = int(np.searchsorted(axis, m_val - window_half, side="left"))
            idx_right = int(np.searchsorted(axis, m_val + window_half, side="right"))
            if idx_right <= idx_left:
                continue
            window = axis[idx_left:idx_right]
            d = (window - m_val) / sigma_da
            ker = np.exp(-0.5 * d * d, dtype=np.float64)
            acc[idx_left:idx_right] += inten * ker
            wsum[idx_left:idx_right] += ker

    # Normalize by accumulated kernel weights (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_y = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0.0)

    return axis, mean_y.astype(np.float32, copy=False)
