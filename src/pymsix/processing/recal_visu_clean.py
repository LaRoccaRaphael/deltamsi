"""
MSI Recalibration Diagnostics
=============================

This module generates visual reports to validate the mass recalibration process.
It compares experimental peaks against a reference database and visualizes
the error distribution and the linear regression model used for correction.

The diagnostics focus on two primary plots per pixel:
1. **KDE Density**: Distribution of mass errors to identify the systematic shift.
2. **Error vs m/z**: Scatter plot of hits and the RANSAC fit used for correction.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyimzml.ImzMLParser import ImzMLParser

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


def _parse_coord_token(tok: str) -> Tuple[int, int, int]:
    """
    Parse a string coordinate token into an (X, Y, Z) integer tuple.

    Parameters
    ----------
    tok : str
        String representing coordinates, either "x,y" or "x,y,z".

    Returns
    -------
    Tuple[int, int, int]
        A 3-tuple of integers (x, y, z). If z is not provided, it defaults to 1.

    Raises
    ------
    ValueError
        If the token does not contain 2 or 3 comma-separated parts or if parts 
        cannot be cast to integers.
    """
    parts = tok.split(",")
    if len(parts) not in (2, 3):
        raise ValueError(f"Bad coordinate token '{tok}'. Expected 'x,y' or 'x,y,z'.")
    x = int(parts[0].strip())
    y = int(parts[1].strip())
    z = int(parts[2].strip()) if len(parts) == 3 else 1
    return x, y, z


def _coords_to_indices(
    p: ImzMLParser, coords: Sequence[Tuple[int, int, int]]
) -> List[int]:
    """
    Map physical spatial coordinates to internal spectrum indices.

    Parameters
    ----------
    p : ImzMLParser
        The parser object containing the `coordinates` attribute of the MSI file.
    coords : Sequence[Tuple[int, int, int]]
        A sequence of (x, y, z) tuples representing the targets.

    Returns
    -------
    List[int]
        A list of integer indices corresponding to the provided coordinates.

    Raises
    ------
    KeyError
        If a provided coordinate does not exist in the imzML metadata.
    """
    coord_map = {tuple(c): i for i, c in enumerate(p.coordinates)}
    out: List[int] = []
    for c in coords:
        if c in coord_map:
            out.append(coord_map[c])
            continue
        c2 = (c[0], c[1], 1)
        if c2 in coord_map:
            out.append(coord_map[c2])
            continue
        raise KeyError(f"Coordinate {c} not found in imzML coordinates.")
    return out


def select_pixels(
    p: ImzMLParser,
    *,
    pixel_idx: Optional[Sequence[int]] = None,
    pixel_coord: Optional[Sequence[str]] = None,
    n_random: Optional[int] = None,
    seed: int = 0,
) -> Any:
    """
    Select a subset of pixels for diagnostic visualization.

    Parameters
    ----------
    p : ImzMLParser
        The parser object for the MSI dataset.
    pixel_idx : Sequence[int], optional
        List of specific 0-based pixel indices to plot.
    pixel_coord : Sequence[str], optional
        List of coordinate strings in 'x,y' or 'x,y,z' format.
    n_random : int, optional
        Number of random pixels to select if no specific pixels are provided.
    seed : int, default 0
        Random seed for reproducible pixel selection.

    Returns
    -------
    List[int]
        A list of validated pixel indices.
    """
    n_total = len(p.coordinates)

    if pixel_idx:
        idx = [int(i) for i in pixel_idx]
        bad = [i for i in idx if i < 0 or i >= n_total]
        if bad:
            raise IndexError(f"Pixel indices out of range: {bad} (n_pixels={n_total})")
        return idx

    if pixel_coord:
        coords = [_parse_coord_token(t) for t in pixel_coord]
        return _coords_to_indices(p, coords)

    if n_random is None:
        n_random = 5

    n = min(int(n_random), n_total)
    rng = np.random.default_rng(int(seed))
    return rng.choice(n_total, size=n, replace=False).tolist()


def diagnostics_for_pixel(
    p: ImzMLParser,
    idx: int,
    db_masses_sorted: np.ndarray,
    params: RecalParams,
) -> Any:
    """
    Compute recalibration statistics for a single pixel.

    This function performs a "mock" recalibration: it identifies hits 
    against the database, calculates the error mode, fits a RANSAC model, 
    and re-calculates errors after applying the correction.

    Parameters
    ----------
    p : ImzMLParser
        The imzML parser.
    idx : int
        The internal index of the pixel to analyze.
    db_masses_sorted : np.ndarray
        Sorted array of reference exact masses.
    params : RecalParams
        Configuration object containing tolerances and KDE bandwidths.

    Returns
    -------
    dict
        A dictionary containing intermediate results:
        - ``"hit_err"``: Errors before correction.
        - ``"hit_err_corr"``: Errors after applying the model.
        - ``"model"``: The RANSAC linear model.
        - ``"pdf"``: The KDE density values for plotting.
    """
    mzs, intensities = p.getspectrum(idx)
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    peaks_mz = select_top_peaks(mzs, intensities, params.n_peaks)
    hit_exp, hit_err = generate_hits(
        peaks_mz, db_masses_sorted, tol_da=params.tol_da, tol_ppm=params.tol_ppm
    )

    grid_hw = kde_grid_halfwidth_da(peaks_mz, params)
    mode, x_grid, pdf = estimate_error_mode(
        hit_err,
        grid_halfwidth_da=grid_hw,
        kde_bw_da=params.kde_bw_da,
        grid_step_da=params.kde_grid_step_da,
    )

    roi_mask = (
        select_hits_roi(hit_err, mode=mode, roi_halfwidth_da=params.roi_halfwidth_da)
        if np.isfinite(mode)
        else np.zeros_like(hit_err, dtype=bool)
    )
    model = fit_ransac_linear_model(hit_exp, hit_err, roi_mask, params)

    hit_err_corr = None
    if model is not None and peaks_mz.size:
        peaks_mz_corr = correct_mz_with_model(peaks_mz, model)
        _, hit_err_corr = generate_hits(
            peaks_mz_corr,
            db_masses_sorted,
            tol_da=params.tol_da,
            tol_ppm=params.tol_ppm,
        )

    return dict(
        idx=idx,
        coord=tuple(p.coordinates[idx]),
        peaks_mz=peaks_mz,
        hit_exp=hit_exp,
        hit_err=hit_err,
        mode=mode,
        x_grid=x_grid,
        pdf=pdf,
        roi_mask=roi_mask,
        model=model,
        hit_err_corr=hit_err_corr,
    )


def plot_diagnostics(diag: Any, params: RecalParams) -> Any:
    """
    Create a diagnostic figure for a pixel.

    The figure consists of two panels:
    - **Left**: KDE of mass errors. The blue line shows the uncorrected shift; 
      the orange dotted line shows the distribution centered around zero 
      after correction.
    - **Right**: Error vs m/z scatter plot. Highlights the Region of Interest (ROI) 
      and the RANSAC regression line.

    Parameters
    ----------
    diag : dict
        The result dictionary from `diagnostics_for_pixel`.
    params : RecalParams
        Recalibration parameters for drawing ROI boundaries.

    Returns
    -------
    matplotlib.figure.Figure
        The generated diagnostic figure.
    """
    idx = diag["idx"]
    coord = diag["coord"]
    hit_exp = diag["hit_exp"]
    hit_err = diag["hit_err"]
    roi_mask = diag["roi_mask"]
    mode = diag["mode"]
    x_grid = diag["x_grid"]
    pdf = diag["pdf"]
    model = diag["model"]
    hit_err_corr = diag["hit_err_corr"]

    fig = plt.figure(figsize=(10, 4), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # (A) Error density
    ax0 = fig.add_subplot(gs[0, 0])

    ax0.plot(x_grid, pdf, linestyle="-", color="tab:blue", label="before (uncorrected)")

    if np.isfinite(mode):
        ax0.axvline(
            mode - params.roi_halfwidth_da, linestyle="--", color="0.4", linewidth=1
        )
        ax0.axvline(
            mode + params.roi_halfwidth_da, linestyle="--", color="0.4", linewidth=1
        )
        ax0.axvline(mode, linestyle=":", color="0.4", linewidth=1)

    if hit_err_corr is not None and np.size(hit_err_corr):
        mode2, x2, pdf2 = estimate_error_mode(
            hit_err_corr,
            grid_halfwidth_da=(
                float(np.max(np.abs(x_grid))) if np.size(x_grid) else params.tol_da
            ),
            kde_bw_da=params.kde_bw_da,
            grid_step_da=params.kde_grid_step_da,
        )
        ax0.plot(
            x2, pdf2, linestyle=":", color="tab:orange", label="after (recomputed)"
        )

    ax0.set_xlabel("Mass error (Da)")
    ax0.set_ylabel("Density (a.u.)")
    ax0.set_title("Hit error density (KDE)")
    ax0.legend(frameon=False)

    # (B) Error vs m/z + fit
    ax1 = fig.add_subplot(gs[0, 1])
    if hit_exp.size:
        ax1.plot(hit_exp, hit_err, "o", markersize=3, alpha=0.35, label="hits")
        if roi_mask.size:
            ax1.plot(
                hit_exp[roi_mask],
                hit_err[roi_mask],
                "o",
                markersize=3,
                alpha=0.8,
                label="ROI hits",
            )
    if model is not None and hit_exp.size:
        mz_min, mz_max = float(np.min(hit_exp)), float(np.max(hit_exp))
        mz_grid = np.linspace(mz_min, mz_max, 200)
        y_grid = model.predict(np.vander(mz_grid, 2))
        ax1.plot(mz_grid, y_grid, linewidth=2, label="RANSAC fit")
    ax1.axhline(0, linewidth=1, alpha=0.5)
    ax1.set_xlabel("m/z (experimental)")
    ax1.set_ylabel("Mass error (Da)")
    ax1.set_title("Error vs m/z (fit)")
    ax1.legend(frameon=False)

    fig.suptitle(f"Pixel idx={idx} coord={coord}", y=0.98, fontsize=11)
    # Reserve space for suptitle in PDFs
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def build_argparser() -> argparse.ArgumentParser:
    """
    Construct the command-line argument parser for the processing script.

    The parser includes groups for:
    - IO (input imzML, database).
    - Matching tolerances (Da, PPM).
    - Statistical parameters (KDE bandwidth, ROI widths).
    - Pixel selection (indices, coordinates, or random sampling).
    - Output settings (PDF generation).

    Returns
    -------
    argparse.ArgumentParser
        The fully configured argument parser.
    """
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-i", "--imzml", required=True, help="Path to input imzML")
    ap.add_argument(
        "-db", "--database", required=True, help="Text/CSV file with exact masses"
    )

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
        "--kde-step-da", type=float, default=1e-4, help="KDE grid step (Da)"
    )
    ap.add_argument(
        "--n-peaks", type=int, default=1000, help="Top-N peaks per pixel used for hits"
    )
    ap.add_argument(
        "--min-hits", type=int, default=20, help="Min hits (after ROI) needed to fit"
    )

    ap.add_argument(
        "--pixel-idx", nargs="+", type=int, help="Pixel indices (0-based) to visualize"
    )
    ap.add_argument(
        "--pixel-coord",
        nargs="+",
        help="Pixel coordinates: 'x,y' or 'x,y,z' (repeatable)",
    )
    ap.add_argument(
        "--n-random",
        type=int,
        help="If no pixel selection is given, draw N random pixels",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="RNG seed for random pixel selection"
    )

    ap.add_argument(
        "--outpdf",
        type=str,
        default=None,
        help="If set, write a multi-page PDF instead of showing plots",
    )
    return ap


def main() -> None:
    """
    Command-line interface for generating recalibration PDF reports.

    Examples
    --------
    Generate random diagnostics and show them interactively:
    $ python diagnostics.py -i data.imzML -db references.csv --n-random 3

    Save diagnostics for specific coordinates to a PDF:
    $ python diagnostics.py -i data.imzML -db ref.csv --pixel-coord "10,10" "20,30" --outpdf report.pdf
    """
    args = build_argparser().parse_args()

    params = RecalParams(
        tol_da=float(args.tol_da),
        tol_ppm=(None if args.tol_ppm is None else float(args.tol_ppm)),
        kde_bw_da=float(args.kde_bw_da),
        roi_halfwidth_da=float(args.roi_da),
        kde_grid_step_da=float(args.kde_step_da),
        n_peaks=int(args.n_peaks),
        min_hits_for_fit=int(args.min_hits),
    )

    db = load_database_masses(args.database)
    p = ImzMLParser(args.imzml, parse_lib="ElementTree")

    pix = select_pixels(
        p,
        pixel_idx=args.pixel_idx,
        pixel_coord=args.pixel_coord,
        n_random=args.n_random,
        seed=args.seed,
    )

    if args.outpdf:
        with PdfPages(args.outpdf) as pdf:
            info = pdf.infodict()
            info["Title"] = "MSI recalibration diagnostics"
            info["Subject"] = f"imzML={args.imzml} | db={args.database}"
            for idx in pix:
                diag = diagnostics_for_pixel(p, idx, db, params)
                fig = plot_diagnostics(diag, params)
                pdf.savefig(fig)
                plt.close(fig)
        print(f"Wrote diagnostics PDF: {args.outpdf}")
    else:
        for idx in pix:
            diag = diagnostics_for_pixel(p, idx, db, params)
            _ = plot_diagnostics(diag, params)
            plt.show()


if __name__ == "__main__":
    main()
