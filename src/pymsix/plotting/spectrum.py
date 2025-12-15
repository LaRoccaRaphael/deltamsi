import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple, Iterable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from pymsix.core.msicube import MSICube


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
    Plots zoomed windows around specified m/z peaks from one or more mean spectra.

    Args:
        msicube: The MSICube object containing mean spectra in adata.uns['mean_spectra'].
        peak_mzs: Sequence of m/z values to center the plots around.
        labels: Names of the samples (as found in adata.obs['sample']) to plot.
                If None, all mean spectra found in adata.uns['mean_spectra'] are plotted.
                (Note: Cette implémentation ne supporte pas 'None' pour les labels).
        span_da: Total width of the m/z window in Dalton (Da).
        tol_da: Tolerance in Dalton (Da) to show dashed lines.
        tol_ppm: Tolerance in ppm to show dashed lines.
        ncols: Number of columns in subplot grid.
        figsize: Figure size.
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
