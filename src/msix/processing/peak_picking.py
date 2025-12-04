"""Peak picking functions for MS1 spectra."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from msix.params.options import PeakPickingOptions


def peak_picking(
    mzs: np.ndarray,
    intensities: np.ndarray,
    options: PeakPickingOptions,
) -> np.ndarray:
    """
    Peak picking on a mean spectrum, with Da and ppm distance thresholds.

    Parameters
    ----------
    mzs : np.ndarray
        1D array of m/z values (can be irregular, can have holes).
    intensities : np.ndarray
        1D array of intensities, same length as `mzs`.
    options : PeakPickingOptions
        Configuration object containing topn, binning_p, distance_da, and distance_ppm.

    Returns
    -------
    selected_mzs : np.ndarray
        m/z positions of selected peaks (sorted ascending).
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
