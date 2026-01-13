"""
MSI Recalibration CLI and I/O Utilities
=======================================

This module serves as the primary entry point for batch processing imzML 
recalibration. It leverages a robust workflow involving:
1. Identifying high-intensity peaks per pixel.
2. Matching peaks against a known database (hits generation).
3. Estimating the mass shift via Kernel Density Estimation (KDE).
4. Refining the calibration using RANSAC linear regression.
5. Exporting the corrected data to a new imzML file.
"""

from __future__ import annotations

import argparse

import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter

from pymsix.params.options import RecalParams
from pymsix.processing.recalibration_core import (
    load_database_masses,
    select_top_peaks,
    generate_hits,
    kde_grid_halfwidth_da,
    estimate_error_mode,
    select_hits_roi,
    fit_ransac_linear_model,
    correct_mz_with_model,
)


def write_corrected_msi(
    imzml_path: str,
    out_imzml_path: str,
    db_masses_sorted: np.ndarray,
    params: RecalParams,
) -> None:
    """
    Read an imzML file, apply pixel-wise recalibration, and write results.

    This function iterates through every spectrum in the input file. For 
    each spectrum, it attempts to find a mass correction model. If successful, 
    the corrected m/z array is written to the output file; otherwise, the 
    spectrum is skipped to ensure data quality.

    Parameters
    ----------
    imzml_path : str
        Path to the source .imzML file.
    out_imzml_path : str
        Path where the recalibrated .imzML (and .ibd) will be saved.
    db_masses_sorted : np.ndarray
        A sorted 1D array of exact reference masses used for alignment.
    params : RecalParams
        A configuration dataclass containing tolerances, KDE settings, 
        and RANSAC thresholds.

    Notes
    -----
    The function enforces quality controls:
    * Pixels with fewer than 30 peaks are ignored.
    * Pixels with fewer than 10 database hits are ignored.
    * Failure to find a finite error mode via KDE results in no correction.

    Examples
    --------
    >>> from pymsix.params import RecalParams
    >>> params = RecalParams(tol_da=0.02, n_peaks=500)
    >>> db = np.array([149.023, 227.031, 413.211])
    >>> write_corrected_msi("input.imzML", "output.imzML", db, params)
    """
    with ImzMLWriter(out_imzml_path) as w:
        p = ImzMLParser(imzml_path, parse_lib="ElementTree")

        for idx, (x, y, z) in enumerate(p.coordinates):
            mzs, intensities = p.getspectrum(idx)
            mzs = np.asarray(mzs, dtype=float)
            intensities = np.asarray(intensities, dtype=float)

            peaks_mz = select_top_peaks(mzs, intensities, params.n_peaks)
            if peaks_mz.size < 30:
                continue

            hit_exp, hit_err = generate_hits(
                peaks_mz, db_masses_sorted, tol_da=params.tol_da, tol_ppm=params.tol_ppm
            )
            if hit_err.size < 10:
                continue

            grid_hw = kde_grid_halfwidth_da(peaks_mz, params)
            mode, _, _ = estimate_error_mode(
                hit_err,
                grid_halfwidth_da=grid_hw,
                grid_step_da=params.kde_grid_step_da,
                kde_bw_da=params.kde_bw_da,
            )
            if not np.isfinite(mode):
                continue

            roi = select_hits_roi(
                hit_err, mode=mode, roi_halfwidth_da=params.roi_halfwidth_da
            )
            model = fit_ransac_linear_model(hit_exp, hit_err, roi, params)
            if model is None:
                continue

            corrected_mzs = correct_mz_with_model(mzs, model)
            w.addSpectrum(corrected_mzs, intensities, (x, y, z))


def build_argparser() -> argparse.ArgumentParser:
    """
    Construct the argument parser for the recalibration CLI.

    Returns
    -------
    argparse.ArgumentParser
        An initialized parser with options for input/output paths and 
        recalibration hyperparameters.
    """
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-i", "--input", required=True, help="Input imzML")
    ap.add_argument(
        "-db", "--database", required=True, help="Exact mass list (txt/CSV)"
    )
    ap.add_argument("-o", "--output", required=True, help="Output imzML (recalibrated)")
    ap.add_argument(
        "--tol-da",
        type=float,
        default=0.03,
        help="Matching tolerance (Da) used if --tol-ppm is not set",
    )
    ap.add_argument(
        "--tol-ppm",
        type=float,
        default=None,
        help="Matching tolerance (ppm). If set, used for binary-search matching",
    )
    ap.add_argument("--kde-bw-da", type=float, default=0.002, help="KDE bandwidth (Da)")
    ap.add_argument(
        "--roi-da",
        type=float,
        default=0.02,
        help="ROI half-width around error mode (Da)",
    )
    ap.add_argument(
        "--n-peaks", type=int, default=1000, help="Top-N peaks per pixel used for hits"
    )
    ap.add_argument(
        "--min-hits", type=int, default=20, help="Min hits (after ROI) needed to fit"
    )
    return ap


def main() -> None:
    """
    Main entry point for the recalibration script.

    Loads the database, initializes parameters from CLI arguments, 
    and executes the `write_corrected_msi` workflow.

    Usage
    -----
    From the terminal:
    $ python recalibration_cli.py -i in.imzML -db db.txt -o out.imzML --tol-ppm 10
    """
    args = build_argparser().parse_args()

    params = RecalParams(
        tol_da=float(args.tol_da),
        tol_ppm=(None if args.tol_ppm is None else float(args.tol_ppm)),
        kde_bw_da=float(args.kde_bw_da),
        roi_halfwidth_da=float(args.roi_da),
        n_peaks=int(args.n_peaks),
        min_hits_for_fit=int(args.min_hits),
    )

    db = load_database_masses(args.database)
    write_corrected_msi(args.input, args.output, db, params)
    print(f"Wrote recalibrated imzML: {args.output}")


if __name__ == "__main__":
    main()
