"""
Mean Spectrum Visualization Module
==================================

This module provides tools for comparative visualization of mean mass spectra.
It allows for detailed "windowed" inspections of specific m/z regions across 
multiple samples, facilitating peak verification and alignment checks.

The visualization highlights target m/z values and their associated 
tolerance ranges (in Da or ppm) using standardized scientific plotting styles.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple, Iterable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from deltamsi.core.msicube import MSICube


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

    This function handles the heavy lifting of grid layout, window slicing, 
    and multi-trace overlay.

    Parameters
    ----------
    mean_spectra : iterable of (np.ndarray, np.ndarray)
        An iterable containing tuples of (mz_array, intensity_array).
    peak_mzs : sequence of float
        Target m/z values to center each subplot window.
    span_da : float
        The total width of the m/z window (horizontal axis range) in Daltons.
    tol_da : float, optional
        Absolute tolerance in Daltons to display as vertical dashed lines.
    tol_ppm : float, optional
        Relative tolerance in parts per million (ppm) to display as vertical 
        dashed lines. Overrides `tol_da` for the visual lines if both provided.
    ncols : int, default 3
        Number of columns in the subplot grid.
    labels : sequence of str, optional
        Labels for each mean spectrum to be displayed in the legend.
    figsize : tuple of float, optional
        Width and height of the figure in inches.

    Raises
    ------
    ValueError
        If `span_da` <= 0, if neither or both tolerances are provided, 
        or if `mean_spectra` is empty.
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

    n_windows = len(peak_mzs_arr)
    nrows = int(np.ceil(n_windows / ncols))

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(n_windows):
        ax = axes[i]
        peak_mz = peak_mzs_arr[i]

        # Déterminer la tolérance pour la ligne pointillée (Da)
        # Si tol_ppm est défini, il est prioritaire
        if tol_ppm is not None:
            tol_da_for_peak = peak_mz * tol_ppm * 1e-6
        elif tol_da is not None:
            tol_da_for_peak = tol_da
        else:
            tol_da_for_peak = 0.0  # Ne devrait pas arriver

        # Définir les limites de la fenêtre
        mz_min = peak_mz - span_da / 2
        mz_max = peak_mz + span_da / 2

        # Tracer chaque spectre moyen dans cette fenêtre
        for j, (mzs, intensities) in enumerate(mean_spectra):
            # Filtrer les données dans la fenêtre
            mask = (mzs >= mz_min) & (mzs <= mz_max)
            mzs_window = mzs[mask]
            intensities_window = intensities[mask]

            # Obtenir le label (si disponible)
            label = labels[j] if labels is not None else f"Spectrum {j+1}"

            # Tracer la trace
            ax.plot(mzs_window, intensities_window, label=label)

        # Ajouter la ligne centrale (m/z cible)
        ax.axvline(
            peak_mz, color="red", linestyle="--", linewidth=1, label="Target m/z"
        )

        # Ajouter les lignes de tolérance (tol_da_for_peak)
        if tol_da_for_peak > 0:
            ax.axvline(
                peak_mz - tol_da_for_peak,
                color="gray",
                linestyle=":",
                linewidth=0.8,
                label=f"Tol (+/-{tol_da_for_peak:.4f} Da)",
            )
            ax.axvline(
                peak_mz + tol_da_for_peak, color="gray", linestyle=":", linewidth=0.8
            )

        # Mise en forme du subplot
        ax.set_xlim(mz_min, mz_max)
        ax.set_title(f"Window around m/z {peak_mz:.4f}", fontsize=10)
        ax.set_xlabel("m/z")
        ax.tick_params(axis="x", rotation=45)

        # Ajouter la légende si plusieurs spectres sont tracés
        if n_specs > 1:
            ax.legend(fontsize=8, loc="upper right")

    # Masquer les axes non utilisés
    for j in range(n_windows, nrows * ncols):
        fig.delaxes(axes[j])

    fig.tight_layout(rect=[0, 0, 1, 0.98])


def plot_mean_spectrum_windows(
    msicube: "MSICube",
    peak_mzs: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    span_da: float = 1.0,
    tol_da: Optional[float] = None,
    tol_ppm: Optional[float] = None,
    ncols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot zoomed m/z windows around specified peaks from MSICube mean spectra.

    This function retrieves pre-computed mean spectra from the ``MSICube`` 
    metadata and generates a grid of subplots. Each subplot displays an 
    overlay of the selected samples' spectra centered around a specific m/z.

    Parameters
    ----------
    msicube : MSICube
        The MSICube instance. Must have ``adata.uns['mean_spectra']`` populated 
        (usually via ``compute_all_mean_spectra()``).
    peak_mzs : sequence of float
        The m/z values used as centers for the zoomed subplots.
    labels : sequence of str, optional
        The names of specific samples to plot (must exist in 
        ``adata.uns['mean_spectra']``). If ``None``, all available samples 
        in the metadata are plotted.
    span_da : float, default 1.0
        Total width of the m/z window in Daltons.
    tol_da : float, optional
        Show tolerance boundaries as dashed lines using absolute Daltons.
    tol_ppm : float, optional
        Show tolerance boundaries as dashed lines using relative ppm.
    ncols : int, default 3
        Number of columns in the subplot grid.
    figsize : tuple of float, optional
        Figure size in inches (width, height). If ``None``, the size is 
        automatically determined based on the number of subplots.

    Notes
    -----
    The function displays a red dashed line at the exact ``peak_mz`` and grey 
    dotted lines for the tolerance boundaries. If multiple samples are plotted, 
    a legend is automatically added to each subplot.

    

    Examples
    --------
    >>> from deltamsi.plotting.spectrum import plot_mean_spectrum_windows
    >>> # Plot windows for 3 specific m/z values across all samples
    >>> plot_mean_spectrum_windows(
    ...     msicube, 
    ...     peak_mzs=[184.073, 520.34, 760.58], 
    ...     span_da=0.5, 
    ...     tol_ppm=10.0
    ... )

    >>> # Compare only two specific samples in a 0.2 Da window
    >>> plot_mean_spectrum_windows(
    ...     msicube, 
    ...     peak_mzs=[760.58], 
    ...     labels=["Control_01", "Treated_01"],
    ...     span_da=0.2, 
    ...     tol_da=0.005
    ... )
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

    # Si labels est None, on prend TOUS les labels disponibles.
    if labels is None:
        labels = list(mean_spectra_storage.keys())

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

    # Appel de la fonction core
    _plot_mean_spectrum_windows_core(
        mean_spectra=spectra_list,
        peak_mzs=peak_mzs,
        span_da=span_da,
        tol_da=tol_da,
        tol_ppm=tol_ppm,
        ncols=ncols,
        labels=found_labels,  # PASSAGE DES LABELS POUR LA LÉGENDE
        figsize=figsize,
    )
