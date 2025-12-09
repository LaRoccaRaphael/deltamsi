# Fichier: processing/recalibration.py

import numpy as np
import os
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from sklearn import linear_model
from scipy.stats import gaussian_kde
from typing import Optional, List, Tuple, Any

# Import RecalibrationOptions from the parameters module
# Assuming pymsix package structure is available in the environment
from pymsix.params.options import RecalibrationOptions

# --- Utility Functions ---


def _peak_selection(ms_intensities: np.ndarray, npeak: int) -> np.ndarray:
    """Selects the indices of the n most intense peaks."""
    intensities_arr = np.array(ms_intensities)
    # Returns the indices corresponding to the n largest values (most intense peaks)
    return intensities_arr.argsort()[::-1][:npeak]


def _compute_mass_error_check(
    experimental_mass: float, database_mass: float, tolerance: float
) -> bool:
    """Checks if the experimental mass is within the tolerance in Dalton (Da)."""
    if database_mass != 0:
        return abs(experimental_mass - database_mass) <= tolerance
    return False


def _binary_search_tol(arr: np.ndarray, x: float, tolerance: float) -> List[int]:
    """
    Binary search with a tolerance (in Da) from an ordered list.
    Returns a list of indices in the database that match the experimental mass.
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
    For each detected mz, returns the Da errors and the corresponding experimental m/z
    for all matches (hits) found in the database.
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
    """Kernel Density Estimation (KDE) of the hit errors."""
    if x.size < 2 or x.std(ddof=1) == 0:
        return np.zeros_like(x_grid)

    # Use the original bandwidth calculation
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def _hits_selection(
    hit_errors: np.ndarray, step: float, tolerance: float, da_limit: float
) -> np.ndarray:
    """Returns the indices of the hits within the most populated error region (based on KDE peak)."""
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
    Estimates a linear model of the m/z error vs. m/z using the RANSAC algorithm.
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
    """Predicts the Da errors for each detected m/z and corrects them."""
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
    Performs mass recalibration for a single imzML/ibd file and writes the output.

    :param imzml_input_path: Path to the uncalibrated input imzML file.
    :param imzml_output_path: Path for the calibrated output imzML file.
    :param database_exactmass: Sorted numpy array of exact calibration masses.
    :param options: RecalibrationOptions object containing hyperparameters.
    :return: True if the operation was successful, False otherwise.
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
