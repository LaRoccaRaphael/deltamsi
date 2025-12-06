import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple, Iterable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from msix.core.msicube import MSICube


def _plot_mean_spectrum_windows_core(
    mean_spectra: Iterable[Tuple[np.ndarray, np.ndarray]],
    peak_mzs: Sequence[float],
    span_da: float,
    *,
    tol_da: Optional[float] = None,
    tol_ppm: Optional[float] = None,
    ncols: int = 3,
    labels: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Internal low-level function to plot zoomed windows of multiple mean spectra.
    """
    if span_da <= 0:
        raise ValueError("span_da must be > 0.")

    if (tol_da is None and tol_ppm is None) or (
        tol_da is not None and tol_ppm is not None
    ):
        raise ValueError("Provide exactly one of tol_da or tol_ppm.")

    # Convert to np.ndarray explicitly and assign to new variable to help type checker
    peak_mzs_arr = np.asarray(peak_mzs, dtype=float)
    mean_spectra = list(mean_spectra)
    n_specs = len(mean_spectra)

    if n_specs == 0:
        raise ValueError("mean_spectra is empty.")

    if labels is not None and len(labels) != n_specs:
        raise ValueError("labels must have same length as mean_spectra.")

    # Use the array for array operations
    n_peaks = len(peak_mzs_arr)
    nrows = int(np.ceil(n_peaks / ncols))

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for ax_idx, (ax, mz0) in enumerate(zip(axes_flat, peak_mzs_arr)):
        x_min = mz0 - span_da
        x_max = mz0 + span_da

        # Track y-range across all spectra in this window
        local_ymax = 0.0

        # Plot all mean spectra in this window
        for spec_idx, (mzs, ints) in enumerate(mean_spectra):
            mzs = np.asarray(mzs, dtype=float)
            ints = np.asarray(ints, dtype=float)
            # Basic shape check handled by extraction usually, but good to keep

            mask = (mzs >= x_min) & (mzs <= x_max)
            if not np.any(mask):
                continue

            mz_win = mzs[mask]
            int_win = ints[mask]

            current_max = float(int_win.max()) if int_win.size else 0.0
            if current_max > local_ymax:
                local_ymax = current_max

            if labels is not None:
                label = labels[spec_idx]
            else:
                label = None

            ax.plot(mz_win, int_win, label=label, linewidth=1.2, alpha=0.8)

        # If nothing was plotted in this window
        if local_ymax == 0.0:
            ax.set_title(f"{mz0:.4f} m/z (no data)")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, 1)
            ax.set_xlabel("m/z")
            ax.set_ylabel("Intensity")
            continue

        # Tolerance in Da around mz0
        if tol_ppm is not None:
            tol_cur_da = mz0 * tol_ppm * 1e-6
        else:
            # Use tol_da, defaulting to 0.0 if not set (though error handling prevents this path)
            tol_cur_da = tol_da if tol_da is not None else 0.0

        # Vertical dashed lines at mz0 ± tolerance
        ax.axvline(
            mz0 - tol_cur_da, linestyle="--", color="gray", linewidth=0.8, alpha=0.5
        )
        ax.axvline(
            mz0 + tol_cur_da, linestyle="--", color="gray", linewidth=0.8, alpha=0.5
        )

        # Red vertical solid lines for all peaks in peak_mzs_arr that fall in the window
        # (To see neighboring peaks)
        in_window = (peak_mzs_arr >= x_min) & (peak_mzs_arr <= x_max)
        for mz_peak in peak_mzs_arr[in_window]:
            ax.axvline(mz_peak, color="red", linewidth=1, linestyle="-", alpha=0.6)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, local_ymax * 1.1)  # Add 10% headroom
        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
        title_tol = f"tol ≈ {tol_cur_da:.4g} Da"
        ax.set_title(f"{mz0:.4f} m/z\nspan ±{span_da:.3f} Da | {title_tol}")

        if labels is not None:
            ax.legend(fontsize="x-small")

    # Hide unused axes
    for ax in axes_flat[n_peaks:]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def plot_mean_spectrum_windows(
    msicube: "MSICube",
    labels: Sequence[str],
    peak_mzs: Sequence[float],
    span_da: float = 0.1,
    *,
    tol_da: Optional[float] = None,
    tol_ppm: Optional[float] = None,
    ncols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    High-level API to plot zoomed windows of mean spectra for specific samples
    stored in an MSICube.

    Parameters
    ----------
    msicube : MSICube
        Initialized MSICube containing mean spectra in adata.uns['mean_spectra'].
    labels : sequence of str
        List of sample names to plot (keys in adata.uns['mean_spectra']).
    peak_mzs : sequence of float
        List of target m/z values to center the windows on.
    span_da : float
        Half-width of the zoom window in Da.
    tol_da : float, optional
        Tolerance in Da to show dashed lines.
    tol_ppm : float, optional
        Tolerance in ppm to show dashed lines.
    ncols : int
        Number of columns in subplot grid.
    figsize : tuple, optional
        Figure size.
    """
    if msicube.adata is None:
        raise ValueError("MSICube.adata is None.")

    if "mean_spectra" not in msicube.adata.uns:
        raise ValueError(
            "adata.uns['mean_spectra'] is missing. "
            "Run compute_all_mean_spectra() first."
        )

    mean_spectra_storage = msicube.adata.uns["mean_spectra"]
    spectra_list: List[Tuple[np.ndarray, np.ndarray]] = []
    found_labels: List[str] = []

    for label in labels:
        if label not in mean_spectra_storage:
            print(f"WARNING: Sample '{label}' not found in mean spectra. Skipping.")
            continue

        data = mean_spectra_storage[label]
        # Assuming structure {"mz": array, "intensity": array}
        if "mz" not in data or "intensity" not in data:
            print(f"WARNING: Malformed data for sample '{label}'. Skipping.")
            continue

        spectra_list.append((data["mz"], data["intensity"]))
        found_labels.append(label)

    if not spectra_list:
        raise ValueError("No valid samples found to plot.")

    # Call the core plotting function
    _plot_mean_spectrum_windows_core(
        mean_spectra=spectra_list,
        peak_mzs=peak_mzs,
        span_da=span_da,
        tol_da=tol_da,
        tol_ppm=tol_ppm,
        ncols=ncols,
        labels=found_labels,
        figsize=figsize,
    )
