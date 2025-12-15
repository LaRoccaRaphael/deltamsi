#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI recalibration for imzML using the refactored core in `recalibration_core.py`.


Example
-------
python recalibration_cli_clean.py \
  -i input.imzML -db calibrants.txt -o output.imzML \
  --tol-da 0.03 --kde-bw-da 0.002 --roi-da 0.02 --n-peaks 1000
"""

from __future__ import annotations

import argparse

import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter

from recalibration_core import (
    RecalParams,
    load_database_masses,
    select_top_peaks,
    generate_hits,
    effective_grid_halfwidth_da,
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

            grid_hw = effective_grid_halfwidth_da(peaks_mz, params)
            mode, _, _ = estimate_error_mode(
                hit_err, grid_halfwidth_da=grid_hw, kde_bw_da=params.kde_bw_da
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
