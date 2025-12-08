import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pymsix.core.msicube import MSICube


def _plot_ion_images_core(
    selected_peaks: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    *,
    var_indices: Optional[Sequence[int]] = None,
    mz_list: Optional[Sequence[float]] = None,
    ncols: int = 4,
    cmap: str = "viridis",
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Internal low-level function to plot ion images from numpy arrays.
    """
    selected_peaks = np.asarray(selected_peaks, dtype=float)
    # Ensure X is dense if it's sparse (common in AnnData)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    coords = np.asarray(coords)

    if (var_indices is None and mz_list is None) or (
        var_indices is not None and mz_list is not None
    ):
        raise ValueError("Provide exactly one of var_indices or mz_list.")

    n_spectra, n_peaks = X.shape
    if selected_peaks.shape[0] != n_peaks:
        raise ValueError(
            f"selected_peaks length ({selected_peaks.shape[0]}) "
            f"must match X.shape[1] ({n_peaks})."
        )

    # Determine which columns to plot and their m/z labels
    col_indices = []
    labels = []

    if var_indices is not None:
        for idx in var_indices:
            if idx < 0 or idx >= n_peaks:
                raise IndexError(f"var_index {idx} out of bounds for {n_peaks} peaks.")
            col_indices.append(int(idx))
            labels.append(float(selected_peaks[idx]))
    else:
        target_mzs: Sequence[float] = cast(Sequence[float], mz_list)

        # For each target m/z, find closest in selected_peaks
        for mz in target_mzs:
            mz = float(mz)
            j = int(np.argmin(np.abs(selected_peaks - mz)))
            col_indices.append(j)
            labels.append(float(selected_peaks[j]))

    col_indices = np.array(col_indices, dtype=int)
    labels = np.array(labels, dtype=float)

    # Build image grid geometry from coords
    x = coords[:, 0].astype(int)
    y = coords[:, 1].astype(int)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # Prepare figure
    n_imgs = len(col_indices)
    nrows = int(np.ceil(n_imgs / ncols))
    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Handle single subplot case (axes is not an array)
    if n_imgs == 1:
        axes = np.array([axes])

    # Flatten axes for easy iteration and ensure it's iterable
    axes_flat = np.atleast_1d(axes).flatten()

    for k, (ax, col_idx, mz_label) in enumerate(zip(axes_flat, col_indices, labels)):
        # Initialise image with NaNs (in case of gaps)
        img = np.full((height, width), np.nan, dtype=float)

        intens = X[:, col_idx]

        # Map each spectrum intensity to the image grid
        # Note: row index corresponds to y, column to x
        # Adjust coordinates to be 0-indexed relative to min
        img[y - y_min, x - x_min] = intens

        # Choose vmax per-image if not provided
        _vmax = vmax if vmax is not None else np.nanmax(img)
        if np.isnan(_vmax) or _vmax <= 0:
            _vmax = 1.0

        im = ax.imshow(
            img,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=_vmax,
            interpolation="nearest",
        )
        ax.set_title(f"{mz_label:.4f} m/z")
        ax.axis("off")

        # Add colorbar
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused axes
    for ax in axes_flat[n_imgs:]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def plot_ion_images(
    msicube: "MSICube",
    sample_name: str,
    *,
    mz_list: Optional[Sequence[float]] = None,
    var_indices: Optional[Sequence[int]] = None,
    ncols: int = 4,
    cmap: str = "viridis",
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    High-level API to plot ion images for a specific sample within an MSICube.

    Parameters
    ----------
    msicube : MSICube
        The initialized MSICube object containing data.
    sample_name : str
        The name of the sample to visualize (must exist in adata.obs['sample']).
    mz_list : sequence of float, optional
        Target m/z values to visualize. Closest peaks in adata.var['m/z'] are chosen.
    var_indices : sequence of int, optional
        Direct indices of the features (columns) to plot.
    ncols : int
        Number of columns in the subplot grid.
    cmap : str
        Colormap to use (default: 'viridis').
    vmin, vmax : float, optional
        Visualization limits.
    figsize : tuple
        Size of the figure.

    Raises
    ------
    ValueError
        If the sample is not found or data is missing.
    """
    if msicube.adata is None:
        raise ValueError("MSICube.adata is None. Run extract_peak_matrix first.")

    # 1. Check Metadata
    if "sample" not in msicube.adata.obs:
        raise ValueError("adata.obs['sample'] column missing. Cannot filter by sample.")

    if "spatial" not in msicube.adata.obsm:
        raise ValueError("adata.obsm['spatial'] missing. Cannot plot images.")

    if "m/z" not in msicube.adata.var:
        raise ValueError("adata.var['m/z'] missing. Cannot resolve m/z values.")

    # 2. Filter data for the specific sample
    # Using numpy boolean indexing on the obs column
    mask = msicube.adata.obs["sample"] == sample_name

    if not np.any(mask):
        available = msicube.adata.obs["sample"].unique().tolist()
        raise ValueError(
            f"Sample '{sample_name}' not found in dataset. Available: {available}"
        )

    # 3. Extract Subset
    # Note: Depending on backend, slicing might return a view or copy.
    # _plot_ion_images_core handles converting to dense array if needed.
    X_sample = msicube.adata.X[mask, :]
    coords_sample = msicube.adata.obsm["spatial"][mask]

    # Coordinates might need to be ensured as numpy array (if it's a DataFrame)
    if hasattr(coords_sample, "values"):
        coords_sample = coords_sample.values

    selected_mzs = msicube.adata.var["m/z"].values

    # 4. Call Core Plotting
    _plot_ion_images_core(
        selected_peaks=selected_mzs,
        X=X_sample,
        coords=coords_sample,
        mz_list=mz_list,
        var_indices=var_indices,
        ncols=ncols,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
    )
