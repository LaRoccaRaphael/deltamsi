"""
Recalibration Core Utilities
============================

This module implements the mathematical logic for pixel-wise recalibration.
The workflow relies on finding "hits" (matches) in a reference database, 
using Kernel Density Estimation (KDE) to find the most probable mass error 
(mode), and applying RANSAC regression to model the error across the m/z range.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

import numpy as np
from scipy.stats import gaussian_kde

from pymsix.params.options import RecalParams


def load_database_masses(path: str) -> np.ndarray:
    """
    Load exact masses from a text or CSV file.

    Parameters
    ----------
    path : str
        Path to the file containing one exact mass per line.

    Returns
    -------
    np.ndarray
        Sorted, unique array of finite mass values.
    """
    masses = np.genfromtxt(path, dtype=float)
    masses = np.asarray(masses, dtype=float).ravel()
    masses = masses[np.isfinite(masses)]
    masses = np.unique(masses)
    masses.sort()
    return masses


def select_top_peaks(
    mzs: np.ndarray, intensities: np.ndarray, n_peaks: int
) -> np.ndarray:
    """
    Select the N most intense peaks from a spectrum.

    Parameters
    ----------
    mzs : np.ndarray
        Experimental m/z values.
    intensities : np.ndarray
        Experimental intensity values.
    n_peaks : int
        Number of peaks to return.

    Returns
    -------
    np.ndarray
        m/z values of the top-N peaks.
        
    Examples
    --------
    >>> mzs = np.array([100.1, 100.2, 100.3])
    >>> ints = np.array([10, 500, 20])
    >>> select_top_peaks(mzs, ints, n_peaks=1)
    array([100.2])
    """
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
    """
    Return the matching tolerance in Dalton (Da) for a specific m/z.

    Parameters
    ----------
    exp_mz : float
        The experimental m/z value.
    tol_da : float
        The absolute tolerance in Dalton (fallback if tol_ppm is None).
    tol_ppm : float, optional
        The relative tolerance in parts-per-million. If provided, overrides tol_da.

    Returns
    -------
    float
        The tolerance converted to absolute Dalton.
    """
    if tol_ppm is None:
        return float(tol_da)
    return float(exp_mz) * float(tol_ppm) * 1e-6


def _db_hits_indices(sorted_db: np.ndarray, x: float, tol_da: float) -> np.ndarray:
    """
    Find indices in a sorted database within a mass window [x-tol, x+tol].

    Uses binary search (`np.searchsorted`) to achieve logarithmic time complexity,
    which is essential for searching large mass databases.

    Parameters
    ----------
    sorted_db : np.ndarray
        1D array of exact masses, must be sorted in ascending order.
    x : float
        The target m/z value to search for.
    tol_da : float
        The search window radius in Dalton.

    Returns
    -------
    np.ndarray
        Array of indices into `sorted_db` that fall within the window.
    """
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
    Match experimental peaks against a database within a given tolerance.

    Parameters
    ----------
    peaks_mz : np.ndarray
        Experimental peaks.
    database_exactmass_sorted : np.ndarray
        Sorted reference masses.
    tol_da : float
        Search window in Da.
    tol_ppm : float, optional
        Search window in ppm.

    Returns
    -------
    hit_exp : np.ndarray
        The experimental m/z for every match found.
    hit_err : np.ndarray
        The mass error (Experimental - Theoretical) in Da.
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
    """
    Perform Kernel Density Estimation (KDE) on x evaluated on x_grid.

    Parameters
    ----------
    x : np.ndarray
        1D array of observed values (e.g., mass errors).
    x_grid : np.ndarray
        1D array of points where the density is evaluated.
    bandwidth : float
        The smoothing bandwidth in Dalton.

    Returns
    -------
    np.ndarray
        The estimated density profile across the grid.

    Notes
    -----
    The function handles edge cases where the standard deviation is zero 
    by placing a single peak at the mean.
    """
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
    Calculate the KDE grid halfwidth (Da) based on mass accuracy parameters.

    If PPM tolerance is used, the width scales with the maximum m/z in the sample.
    Otherwise, the constant Dalton tolerance is used.

    Parameters
    ----------
    peaks_mz : np.ndarray
        1D array of m/z values found in the pixel/sample.
    params : RecalParams
        An object containing `tol_da` and `tol_ppm` attributes.

    Returns
    -------
    float
        The grid halfwidth in Dalton.
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
    """
    Find the most frequent mass error using Kernel Density Estimation.

    This identifies the systematic shift of the instrument.

    Parameters
    ----------
    hit_errors : np.ndarray
        Array of errors from `generate_hits`.
    grid_halfwidth_da : float
        Range of the error search (e.g., 0.05 Da).
    kde_bw_da : float
        KDE bandwidth.

    Returns
    -------
    mode : float
        The m/z value where the error density is highest.
    x_grid : np.ndarray
        The m/z error values evaluated.
    pdf : np.ndarray
        The density values at each grid point.
    """
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
    """
    Create a mask to filter hits within a specific Region of Interest (ROI).

    Typically used to isolate the most frequent mass error (mode) from 
    random background matches.

    Parameters
    ----------
    hit_errors : np.ndarray
        1D array of mass errors (observed - theoretical).
    mode : float
        The target error center (often the peak of the KDE density).
    roi_halfwidth_da : float
        The radius around the mode to include in the mask.

    Returns
    -------
    np.ndarray
        Boolean array where True represents a hit within the ROI.
    """
    x = np.asarray(hit_errors, dtype=float)
    return (x >= mode - float(roi_halfwidth_da)) & (x <= mode + float(roi_halfwidth_da))


def fit_ransac_linear_model(
    hit_exp_mz: np.ndarray,
    hit_errors_da: np.ndarray,
    roi_mask: np.ndarray,
    params: RecalParams,
) -> Any:
    """
    Fit a robust linear regression to the mass errors.

    Uses RANSAC to ignore "false hits" (random matches) and focus on 
    the true population of calibrated ions.

    Parameters
    ----------
    hit_exp_mz : np.ndarray
        Experimental m/z of hits.
    hit_errors_da : np.ndarray
        Errors of hits.
    roi_mask : np.ndarray
        Boolean mask of hits lying within the expected error region.
    params : RecalParams
        Parameters for RANSAC trials and sample sizes.

    Returns
    -------
    model : sklearn.linear_model.RANSACRegressor or None
        The fitted model, or None if insufficient hits were found.
    """

    from sklearn import linear_model

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

    from sklearn import linear_model

    model = linear_model.RANSACRegressor(
        max_trials=int(params.ransac_max_trials),
        min_samples=int(params.ransac_min_samples),
    )
    model.fit(X_roi, y_roi)
    return model


def correct_mz_with_model(mzs: np.ndarray, model: Any) -> np.ndarray:
    """
    Apply the calculated model to correct a full spectrum.

    Parameters
    ----------
    mzs : np.ndarray
        The raw m/z values to correct.
    model : Any
        A fitted linear model (e.g., from `fit_ransac_linear_model`).

    Returns
    -------
    np.ndarray
        The recalibrated m/z values.

    Notes
    -----
    Correction follows: $m_{new} = m_{old} - \text{predicted\_error}$.
    """
    mzs = np.asarray(mzs, dtype=float)
    X = np.vander(mzs, 2)
    pred_err = model.predict(X)
    return mzs - pred_err
