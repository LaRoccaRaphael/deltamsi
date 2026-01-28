"""
Peak Picking and Matrix Extraction
==================================

This module provides functions to identify relevant m/z features from mean 
spectra and project individual pixel data onto these features.

The workflow typically follows these steps:
1. Identify peaks in a global mean spectrum using `peak_picking`.
2. Extract intensities for these specific peaks across all pixels using 
   `extract_peak_matrix` to create a (pixels x features) dataset.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from typing import Tuple
from pyimzml.ImzMLParser import ImzMLParser

from deltamsi.params.options import PeakPickingOptions, PeakMatrixOptions


def peak_picking(
    mzs: np.ndarray,
    intensities: np.ndarray,
    options: PeakPickingOptions,
) -> np.ndarray:
    """
    Perform greedy peak picking on a spectrum with distance constraints.

    This function identifies local maxima and filters them based on a 
    combination of intensity (ranking) and m/z distance. It ensures that 
    selected peaks are separated by at least `distance_da` or `distance_ppm`.

    Parameters
    ----------
    mzs : np.ndarray
        1D array of m/z values from a mean spectrum.
    intensities : np.ndarray
        1D array of intensities corresponding to `mzs`.
    options : PeakPickingOptions
        Configuration object specifying:
        
        * **topn**: Maximum number of peaks to retain.
        * **binning_p**: Internal resolution for peak localization.
        * **distance_da**: Minimum distance between peaks in Daltons.
        * **distance_ppm**: Minimum distance between peaks in ppm.

    Returns
    -------
    selected_mzs : np.ndarray
        Sorted 1D array of selected peak m/z positions.

    Notes
    -----
    The greedy selection logic prioritizes intensity: the highest peak in a 
    region is always kept, and neighboring candidates within the tolerance 
    window are discarded.

    
    """
    options.validate()

    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    if mzs.ndim != 1 or intensities.ndim != 1:
        raise ValueError("mzs and intensities must be 1D arrays.")
    if mzs.size != intensities.size:
        raise ValueError("mzs and intensities must have the same length.")

    # Guess m/z range from data
    min_mz = float(mzs.min())
    max_mz = float(mzs.max())

    if min_mz >= max_mz:
        raise ValueError("min_mz must be < max_mz.")

    # Regular axis for find_peaks
    mz_axis = np.arange(min_mz, max_mz, options.binning_p, dtype=float)

    # Interpolate intensities onto regular axis (outside range -> 0)
    order = np.argsort(mzs)
    mzs_sorted = mzs[order]
    ints_sorted = intensities[order]

    f = interp1d(
        mzs_sorted,
        ints_sorted,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    ints_reg = f(mz_axis)
    ints_reg[ints_reg < 0] = 0.0

    # Optional scaling (doesn't affect peak positions or spacing logic)
    if ints_reg.max() > 0:
        ints_reg = ints_reg / ints_reg.max()

    # 1) Get all candidate peaks (no distance constraint here)
    candidate_idx, _ = find_peaks(ints_reg)
    if candidate_idx.size == 0:
        return np.array([], dtype=float)

    # 2) Sort candidates by intensity (descending)
    candidate_ints = ints_reg[candidate_idx]
    sort_order = np.argsort(candidate_ints)[::-1]
    candidate_idx = candidate_idx[sort_order]
    candidate_ints = candidate_ints[sort_order]

    # 3) Greedy selection: enforce Da / ppm distance, keep highest intensities
    selected_idx: list[int] = []
    selected_mz: list[float] = []

    for idx_c, inten_c in zip(candidate_idx, candidate_ints):
        mz_c = mz_axis[idx_c]

        if not selected_idx:
            # First (most intense) is always accepted
            selected_idx.append(int(idx_c))
            selected_mz.append(float(mz_c))
            if len(selected_idx) >= options.topn:
                break
            continue

        sel_mz_arr = np.array(selected_mz)
        delta_m = np.abs(sel_mz_arr - mz_c)

        too_close_da = False
        too_close_ppm = False

        if options.distance_da is not None and options.distance_da > 0:
            too_close_da = np.any(delta_m < options.distance_da)

        if options.distance_ppm is not None and options.distance_ppm > 0:
            m_ref = (sel_mz_arr + mz_c) / 2.0
            delta_ppm = delta_m / m_ref * 1e6
            too_close_ppm = np.any(delta_ppm < options.distance_ppm)

        # If too close in Da OR ppm to any already selected peak: reject
        if (options.distance_da is not None and too_close_da) or (
            options.distance_ppm is not None and too_close_ppm
        ):
            continue

        selected_idx.append(int(idx_c))
        selected_mz.append(float(mz_c))
        if len(selected_idx) >= options.topn:
            break

    selected_mz_array = np.array(selected_mz, dtype=float)
    # Sort final peaks by m/z ascending
    order = np.argsort(selected_mz_array)
    return selected_mz_array[order]


def _align_peaks_to_targets(
    mzs: np.ndarray,
    intensities: np.ndarray,
    target_mz: np.ndarray,
    options: PeakMatrixOptions,
) -> np.ndarray:
    """
    Align one spectrum's peaks to a set of target m/z values.

    For each experimental peak (m, I), all target m/z values within the
    tolerance window are assigned intensity max(current, I).

    Parameters
    ----------
    mzs : np.ndarray
        Experimental m/z values (1D).
    intensities : np.ndarray
        Experimental intensities (1D, same length as mzs).
    target_mz : np.ndarray
        Sorted target m/z values (1D).
    tol_da : float, optional
        Mass tolerance in Da (constant window).
    tol_ppm : float, optional
        Mass tolerance in ppm (window scales with m/z).

    Returns
    -------
    pixel_vec : np.ndarray
        1D array of length len(target_mz) with intensities for this spectrum.
    """
    options.validate()

    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    # Ensure sorted experimental peaks
    order = np.argsort(mzs)
    mzs = mzs[order]
    intensities = intensities[order]

    pixel_vec = np.zeros(target_mz.size, dtype=float)

    for m, i in zip(mzs, intensities):
        if i <= 0:
            continue

        if options.tol_ppm is not None:
            tol = m * options.tol_ppm * 1e-6  # ppm -> Da
        else:
            tol = options.tol_da if options.tol_da is not None else 0.0

        if tol <= 0:
            continue

        left_m = m - tol
        right_m = m + tol

        # Find indices in target_mz within [left_m, right_m]
        left = np.searchsorted(target_mz, left_m, side="left")
        right = np.searchsorted(target_mz, right_m, side="right")

        if left >= right:
            continue  # no target in this window

        # Update with max intensity (vectorised)
        pixel_vec[left:right] = np.maximum(pixel_vec[left:right], i)

    return pixel_vec


def extract_peak_matrix(
    imzml_path: str,
    target_mzs: np.ndarray,
    options: PeakMatrixOptions,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract peak intensities from an imzML file into a 2D matrix.

    For every pixel in the MSI dataset, this function finds the maximum 
    intensity within a tolerance window (Da or ppm) around each target m/z. 
    The result is a feature matrix ready for multivariate analysis (PCA, 
    clustering, etc.).

    Parameters
    ----------
    imzml_path : str
        Path to the imzML file.
    target_mzs : np.ndarray
        Sorted array of target m/z values (e.g., from `peak_picking`).
    options : PeakMatrixOptions
        Configuration object specifying the extraction tolerance:
        
        * **tol_da**: Absolute window half-width in Daltons.
        * **tol_ppm**: Relative window half-width in ppm.

    Returns
    -------
    X : np.ndarray
        2D array of shape (n_pixels, n_features) containing intensities.
    coords : np.ndarray
        2D array of shape (n_pixels, 2) containing [x, y] coordinates.

    Notes
    -----
    **Peak Alignment**
    Since mass accuracy varies slightly across a tissue section, this 
    function uses a "max-over-window" approach. For each target m/z $M_j$, 
    the intensity $X_{ij}$ for pixel $i$ is:
    
    $$X_{ij} = \max \{ I(m) \mid m \in [M_j - \text{tol}, M_j + \text{tol}] \}$$

    

    Examples
    --------
    >>> from deltamsi.processing.peaks import extract_peak_matrix
    >>> # Extract 500 features with 5 ppm tolerance
    >>> opts = PeakMatrixOptions(tol_ppm=5.0)
    >>> X, coords = extract_peak_matrix("data.imzML", picked_mzs, opts)
    >>> print(f"Matrix shape: {X.shape}")
    """
    options.validate()

    # Ensure target m/z are sorted (crucial for searchsorted)
    # We assume the caller passes sorted m/z from adata.var, but robust code checks.
    if np.any(np.diff(target_mzs) < 0):
        target_mzs = np.sort(target_mzs)

    parser = ImzMLParser(imzml_path)
    n_spectra = len(parser.coordinates)
    n_targets = target_mzs.size

    X = np.zeros((n_spectra, n_targets), dtype=np.float32)  # float32 saves RAM
    coords = np.zeros((n_spectra, 2), dtype=int)

    for i, (x, y, z) in enumerate(parser.coordinates):
        mzs, intensities = parser.getspectrum(i)
        coords[i, 0] = x
        coords[i, 1] = y

        if mzs is None or intensities is None or len(mzs) == 0:
            continue

        pixel_vec = _align_peaks_to_targets(
            mzs,
            intensities,
            target_mzs,
            options=options,
        )
        X[i, :] = pixel_vec

    return X, coords
