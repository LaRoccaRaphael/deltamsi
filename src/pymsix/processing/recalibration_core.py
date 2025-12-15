#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recalibration core utilities for centroid MSI spectra.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np
from scipy.stats import gaussian_kde
from sklearn import linear_model


@dataclass(frozen=True)
class RecalParams:
    # Matching tolerance (choose one)
    tol_da: float = 0.03  # used if tol_ppm is None
    tol_ppm: Optional[float] = None  # if set, per-peak tol_da = mz * tol_ppm * 1e-6

    # KDE / ROI (in Da)
    kde_bw_da: float = 0.002
    roi_halfwidth_da: float = 0.02
    kde_grid_step_da: float = 1e-4

    # Peak selection + fit
    n_peaks: int = 1000
    min_hits_for_fit: int = 20
    ransac_max_trials: int = 300
    ransac_min_samples: int = 10


def load_database_masses(path: str) -> np.ndarray:
    """Load a calibrant/exact-mass list from text/CSV-like file (1 column)."""
    masses = np.genfromtxt(path, dtype=float)
    masses = np.asarray(masses, dtype=float).ravel()
    masses = masses[np.isfinite(masses)]
    masses = np.unique(masses)
    masses.sort()
    return masses


def select_top_peaks(
    mzs: np.ndarray, intensities: np.ndarray, n_peaks: int
) -> np.ndarray:
    """Return m/z values of the top-N peaks by intensity (descending)."""
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    if n_peaks <= 0:
        raise ValueError("n_peaks must be > 0")
    if intensities.size == 0:
        return np.asarray([], dtype=float)
    n = min(int(n_peaks), intensities.size)
    idx = np.argpartition(intensities, -n)[-n:]
    idx = idx[np.argsort(intensities[idx])[::-1]]
    return mzs[idx]


def tol_da_for_peak(exp_mz: float, *, tol_da: float, tol_ppm: Optional[float]) -> float:
    """Return the matching tolerance in Da for this exp_mz."""
    if tol_ppm is None:
        return float(tol_da)
    return float(exp_mz) * float(tol_ppm) * 1e-6


def _db_hits_indices(sorted_db: np.ndarray, x: float, tol_da: float) -> np.ndarray:
    """Indices in sorted_db within [x-tol, x+tol] using binary search (searchsorted)."""
    left = np.searchsorted(sorted_db, x - tol_da, side="left")
    right = np.searchsorted(sorted_db, x + tol_da, side="right")
    if right <= left:
        return np.asarray([], dtype=int)
    return np.arange(left, right, dtype=int)


def generate_hits(
    peaks_mz: np.ndarray,
    database_exactmass_sorted: np.ndarray,
    *,
    tol_da: float,
    tol_ppm: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each experimental peak, find all database masses within tolerance
    and return (hit_exp_mz, hit_errors_da).
    """
    db = np.asarray(database_exactmass_sorted, dtype=float)
    hit_exp = []
    hit_err = []

    for exp_mz in np.asarray(peaks_mz, dtype=float):
        t_da = tol_da_for_peak(float(exp_mz), tol_da=float(tol_da), tol_ppm=tol_ppm)
        idxs = _db_hits_indices(db, float(exp_mz), t_da)
        if idxs.size == 0:
            continue
        true_mz = db[idxs]
        hit_exp.append(np.full(true_mz.shape, exp_mz, dtype=float))
        hit_err.append(exp_mz - true_mz)

    if not hit_exp:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    return np.concatenate(hit_exp), np.concatenate(hit_err)


def kde_pdf(x: np.ndarray, x_grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """KDE on x evaluated on x_grid (bandwidth in Da)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.zeros_like(x_grid, dtype=float)
    s = float(np.std(x, ddof=1))
    if not np.isfinite(s) or s <= 0:
        out = np.zeros_like(x_grid, dtype=float)
        j = int(np.argmin(np.abs(x_grid - float(np.mean(x)))))
        out[j] = 1.0
        return out
    kde = gaussian_kde(x, bw_method=float(bandwidth) / s)
    return kde.evaluate(x_grid)


def kde_grid_halfwidth_da(peaks_mz: np.ndarray, params: RecalParams) -> float:
    """
    KDE grid halfwidth (Da) used for density plot.
    - If tol_ppm is set, take tolerance at max(mz).
    - Else use tol_da.
    """
    if params.tol_ppm is None:
        return float(params.tol_da)
    peaks_mz = np.asarray(peaks_mz, dtype=float)
    if peaks_mz.size == 0:
        return float(params.tol_da)
    return float(np.max(peaks_mz)) * float(params.tol_ppm) * 1e-6


def estimate_error_mode(
    hit_errors: np.ndarray,
    *,
    grid_halfwidth_da: float,
    kde_bw_da: float,
    grid_step_da: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate error mode via KDE. Returns (mode, x_grid, pdf)."""
    hw = float(grid_halfwidth_da)
    step = float(grid_step_da)
    if hw <= 0 or step <= 0:
        raise ValueError("grid_halfwidth_da and grid_step_da must be > 0")
    x_grid = np.arange(-hw, hw + step, step, dtype=float)
    pdf = kde_pdf(
        np.asarray(hit_errors, dtype=float), x_grid, bandwidth=float(kde_bw_da)
    )
    if pdf.size == 0 or np.all(pdf == 0):
        return float("nan"), x_grid, pdf
    mode = float(x_grid[int(np.argmax(pdf))])
    return mode, x_grid, pdf


def select_hits_roi(
    hit_errors: np.ndarray, *, mode: float, roi_halfwidth_da: float
) -> np.ndarray:
    """Mask selecting hits within [mode - roi_halfwidth, mode + roi_halfwidth]."""
    x = np.asarray(hit_errors, dtype=float)
    return (x >= mode - float(roi_halfwidth_da)) & (x <= mode + float(roi_halfwidth_da))


def fit_ransac_linear_model(
    hit_exp_mz: np.ndarray,
    hit_errors_da: np.ndarray,
    roi_mask: np.ndarray,
    params: RecalParams,
) -> Any:
    """Fit error = a*mz + b on ROI hits using RANSAC. Returns model or None."""
    hit_exp_mz = np.asarray(hit_exp_mz, dtype=float)
    hit_errors_da = np.asarray(hit_errors_da, dtype=float)
    roi_mask = np.asarray(roi_mask, dtype=bool)

    if hit_exp_mz.size == 0 or hit_errors_da.size == 0:
        return None

    X = np.vander(hit_exp_mz, 2)  # [mz, 1]
    X_roi = X[roi_mask]
    y_roi = hit_errors_da[roi_mask]

    if y_roi.size < max(int(params.min_hits_for_fit), int(params.ransac_min_samples)):
        return None

    model = linear_model.RANSACRegressor(
        max_trials=int(params.ransac_max_trials),
        min_samples=int(params.ransac_min_samples),
    )
    model.fit(X_roi, y_roi)
    return model


def correct_mz_with_model(mzs: np.ndarray, model: Any) -> np.ndarray:
    """Apply fitted model to correct m/z values (Da correction)."""
    mzs = np.asarray(mzs, dtype=float)
    X = np.vander(mzs, 2)
    pred_err = model.predict(X)
    return mzs - pred_err
