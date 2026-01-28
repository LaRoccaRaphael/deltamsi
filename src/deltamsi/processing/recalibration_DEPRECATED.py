"""
Mass Spectrometry Recalibration Module
======================================

This module provides tools for the recalibration of mass spectrometry data
stored in imzML format. It uses a reference database of exact masses to 
estimate m/z drift using Kernel Density Estimation (KDE) and robust 
linear regression (RANSAC).

The main workflow involves:
    1. Identifying high-intensity peaks in each spectrum.
    2. Matching peaks against a database of known exact masses.
    3. Finding the most probable mass error shift using KDE.
    4. Fitting a linear model to describe error vs. m/z.
    5. Correcting the entire spectrum based on the fitted model.

Functions
---------
recalibrate_imzml_file
    Main entry point to recalibrate an entire imzML dataset.

Notes
-----
The recalibration requires a minimum number of peaks and hits to ensure 
statistical robustness. If these conditions are not met, the original 
spectrum is preserved.

Examples
--------
>>> from deltamsi.params.options import RecalibrationOptions
>>> options = RecalibrationOptions(tol=0.5, step=0.01, dalim=0.05, npeak=100)
>>> db_masses = np.array([121.0509, 322.0481, 622.0289, 922.0097])
>>> success = recalibrate_imzml_file("input.imzML", "output.imzML", db_masses, options)
"""

import numpy as np
import os
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from sklearn import linear_model
from scipy.stats import gaussian_kde
from typing import Optional, List, Tuple, Any

# Import RecalibrationOptions from the parameters module
# Assuming deltamsi package structure is available in the environment
from deltamsi.params.options import RecalibrationOptions

# --- Utility Functions ---


def _peak_selection(ms_intensities: np.ndarray, npeak: int) -> np.ndarray:
    """
    Select the indices of the n most intense peaks.

    Parameters
    ----------
    ms_intensities : np.ndarray
        Array of intensity values for a single spectrum.
    npeak : int
        Number of top peaks to select.

    Returns
    -------
    np.ndarray
        Indices of the top n peaks, sorted by descending intensity.
    """
    intensities_arr = np.array(ms_intensities)
    # Returns the indices corresponding to the n largest values (most intense peaks)
    return intensities_arr.argsort()[::-1][:npeak]


def _compute_mass_error_check(
    experimental_mass: float, database_mass: float, tolerance: float
) -> bool:
    """
    Check if the experimental mass is within a specified tolerance.

    Parameters
    ----------
    experimental_mass : float
        The measured m/z value.
    database_mass : float
        The reference exact mass from the database.
    tolerance : float
        Maximum allowed difference in Dalton (Da).

    Returns
    -------
    bool
        True if the mass is within tolerance, False otherwise.
    """
    if database_mass != 0:
        return abs(experimental_mass - database_mass) <= tolerance
    return False


def _binary_search_tol(arr: np.ndarray, x: float, tolerance: float) -> List[int]:
    """
    Search for all occurrences of a mass in a sorted array within a tolerance.

    Parameters
    ----------
    arr : np.ndarray
        Sorted array of reference exact masses.
    x : float
        Target experimental mass to find.
    tolerance : float
        Search window in Dalton (Da).

    Returns
    -------
    List[int]
        List of indices in `arr` that match the experimental mass.
    """
    left, right = 0, len(arr) - 1
    index = []

    # Initial search for the center point
    while left <= right:
        mid = left + (right - left) // 2
        if _compute_mass_error_check(x, arr[mid], tolerance):
            # Match found. Expand to find all neighbors within tolerance.
            index.append(mid)

            # Extend right
            itpos = mid + 1
            while itpos < len(arr) and _compute_mass_error_check(
                x, arr[itpos], tolerance
            ):
                index.append(itpos)
                itpos += 1

            # Extend left
            itneg = mid - 1
            while itneg >= 0 and _compute_mass_error_check(x, arr[itneg], tolerance):
                index.append(itneg)
                itneg -= 1

            return index
        elif arr[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    return []


def _hits_generation(
    peaks_mz: np.ndarray, database_exactmass: np.ndarray, tolerance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate match hits between experimental peaks and database masses.

    Parameters
    ----------
    peaks_mz : np.ndarray
        Experimental m/z values selected for calibration.
    database_exactmass : np.ndarray
        Sorted reference exact masses.
    tolerance : float
        Maximum search tolerance in Dalton (Da).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - hit_exp: Array of experimental m/z values that found a match.
        - hit_errors: Array of corresponding errors (experimental - true).
    """
    hit_errors = []
    hit_exp = []

    for exp_peak in peaks_mz:
        # Search for database hits for the current experimental peak
        db_ind = _binary_search_tol(database_exactmass, exp_peak, tolerance)
        if db_ind:
            for db_index in db_ind:
                true_peak = database_exactmass[db_index]
                da_error = exp_peak - true_peak
                hit_errors.append(da_error)
                hit_exp.append(exp_peak)

    return (np.asarray(hit_exp), np.asarray(hit_errors))


def _kde_scipy(
    x: np.ndarray, x_grid: np.ndarray, bandwidth: float, **kwargs: Any
) -> np.ndarray:
    """
    Perform Kernel Density Estimation (KDE) on hit errors.

    Parameters
    ----------
    x : np.ndarray
        Input data points (mass errors).
    x_grid : np.ndarray
        The grid of values where the density is evaluated.
    bandwidth : float
        Smoothing parameter for the kernel.
    **kwargs : Any
        Additional arguments passed to `scipy.stats.gaussian_kde`.

    Returns
    -------
    np.ndarray
        The estimated probability density function evaluated on the grid.
    """
    if x.size < 2 or x.std(ddof=1) == 0:
        return np.zeros_like(x_grid)

    # Use the original bandwidth calculation
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def _hits_selection(
    hit_errors: np.ndarray, step: float, tolerance: float, da_limit: float
) -> np.ndarray:
    """
    Select hits belonging to the most frequent error cluster.

    Uses KDE to find the peak of the error distribution and selects points
    within a small window around that peak.

    Parameters
    ----------
    hit_errors : np.ndarray
        Array of mass errors.
    step : float
        Bandwidth/smoothing for the KDE.
    tolerance : float
        Total range of the error grid (-tol to +tol).
    da_limit : float
        Half-width of the window to keep around the found maximum.

    Returns
    -------
    np.ndarray
        Boolean mask of selected hits.
    """
    x = hit_errors
    # Create the grid for KDE evaluation
    x_grid = np.arange(-tolerance, tolerance + 0.0001, 0.0001)

    pdf = _kde_scipy(x, x_grid, bandwidth=step)

    if pdf.size == 0 or np.sum(pdf) == 0:
        return np.zeros_like(hit_errors, dtype=bool)

    # Find the maximum density peak
    max_da_value = x_grid[np.argmax(pdf, axis=0)]

    # Select hits within the Da limit around the max density peak
    roi = (x <= (max_da_value + da_limit)) & (x >= (max_da_value - da_limit))
    return roi


def _create_lm(
    hit_exp: np.ndarray, hit_errors: np.ndarray, options: RecalibrationOptions
) -> Optional[linear_model.RANSACRegressor]:
    """
    Estimate a robust linear model for the m/z error.

    Fits a RANSAC model to the relation: `error = a * m/z + b`.

    Parameters
    ----------
    hit_exp : np.ndarray
        Experimental m/z values.
    hit_errors : np.ndarray
        Experimental mass errors.
    options : RecalibrationOptions
        Hyperparameters for KDE selection and RANSAC fitting.

    Returns
    -------
    Optional[linear_model.RANSACRegressor]
        The fitted RANSAC model, or None if fitting failed or insufficient data.
    """
    # Select hits based on KDE density peak
    roi = _hits_selection(
        hit_errors, options.step, tolerance=options.tol, da_limit=options.dalim
    )

    if np.sum(roi) < 10:  # Minimum of 10 points required for RANSAC
        return None

    X_roi = np.vander(hit_exp[roi], 2)  # Create X matrix (mz^1, mz^0)
    y_roi = hit_errors[roi]

    try:
        # Use RANSAC with fixed parameters from the original script
        model = linear_model.RANSACRegressor(max_trials=300, min_samples=10)
        mz_error_model = model.fit(X_roi, y_roi)
        return mz_error_model
    except ValueError:
        return None


def _correct_mz_lm(
    ms_mzs: np.ndarray, mz_error_model: linear_model.RANSACRegressor
) -> np.ndarray:
    """
    Apply mass correction to a spectrum using the linear model.

    Parameters
    ----------
    ms_mzs : np.ndarray
        Original array of m/z values.
    mz_error_model : linear_model.RANSACRegressor
        The fitted linear model to predict mass errors.

    Returns
    -------
    np.ndarray
        The corrected m/z values.
    """
    X = np.vander(ms_mzs, 2)
    predicted_mz_errors = mz_error_model.predict(X)
    estimated_mz = ms_mzs - predicted_mz_errors
    return estimated_mz


# --- Main Function for MSICube Integration ---


def recalibrate_imzml_file(
    imzml_input_path: str,
    imzml_output_path: str,
    database_exactmass: np.ndarray,
    options: RecalibrationOptions,
) -> bool:
    """
    Recalibrate an entire imzML file and save the output.

    Iterates through all spectra in the input file, computes a recalibration
    model for each pixel, and writes the corrected spectra to a new file.

    Parameters
    ----------
    imzml_input_path : str
        Path to the source .imzML file.
    imzml_output_path : str
        Path where the recalibrated .imzML file will be saved.
    database_exactmass : np.ndarray
        Sorted array of reference exact masses (must be ordered).
    options : RecalibrationOptions
        Configuration object containing:
            - tol: Initial matching tolerance (Da).
            - step: KDE bandwidth.
            - dalim: Error cluster window size.
            - npeak: Number of peaks to use for model fitting.

    Returns
    -------
    bool
        True if the recalibration was completed successfully, False otherwise.

    Examples
    --------
    >>> recalibrate_imzml_file("sample.imzML", "sample_cal.imzML", db, opt)
    """
    try:
        with ImzMLWriter(imzml_output_path) as w:
            p = ImzMLParser(imzml_input_path, parse_lib="ElementTree")

            recalibrated_spectra_count = 0
            npeak = options.npeak

            # Iterate over every pixel
            for idx, (x, y, z) in enumerate(p.coordinates):
                ms_mzs, ms_intensities = p.getspectrum(idx)

                # Step 1: Select top N most intense peaks for calibration
                peaks_ind = _peak_selection(ms_intensities, npeak)
                peaks_mz = ms_mzs[peaks_ind]

                # Check minimum conditions from original script
                if len(peaks_mz) > 30:
                    # Step 2: Generate calibration hits
                    hit_exp, hit_errors = _hits_generation(
                        peaks_mz, database_exactmass, options.tol
                    )

                    if len(hit_errors) > 10:
                        # Step 3: Create and fit RANSAC linear model
                        mz_error_model = _create_lm(hit_exp, hit_errors, options)

                        if mz_error_model:
                            # Step 4: Correct m/z for the ENTIRE spectrum
                            corrected_mzs = _correct_mz_lm(ms_mzs, mz_error_model)

                            # Step 5: Write the corrected spectrum
                            w.addSpectrum(corrected_mzs, ms_intensities, (x, y, z))
                            recalibrated_spectra_count += 1
                        else:
                            # If model fails, write the original spectrum
                            w.addSpectrum(ms_mzs, ms_intensities, (x, y, z))
                    else:
                        # Less than 10 hits, write the original spectrum
                        w.addSpectrum(ms_mzs, ms_intensities, (x, y, z))
                else:
                    # Less than 30 peaks, write the original spectrum
                    w.addSpectrum(ms_mzs, ms_intensities, (x, y, z))

        print(
            f"INFO: Recalibration of {os.path.basename(imzml_output_path)} completed. {recalibrated_spectra_count} spectra corrected."
        )
        return True

    except Exception as e:
        print(f"ERROR: Recalibration failed for {imzml_input_path}: {e}")
        return False
