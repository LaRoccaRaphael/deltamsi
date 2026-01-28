"""
Ion Image Visualization Module
==============================

This module provides high-level plotting functions for Mass Spectrometry Imaging 
(MSI) data. It is specifically designed to work with ``MSICube`` objects and 
their underlying ``AnnData`` structures.

The main focus is on producing "publication-ready" visualizations with consistent 
intensity scales, scientific color maps, and clear layout management for 
multi-sample or multi-ion comparisons.

Design Philosophy
-----------------
* **Non-destructive**: Input data is never modified in-place.
* **Scale Awareness**: Multiple scaling modes (global, per-sample, per-ion) 
  ensure that intensity comparisons are biologically and technically meaningful.
* **Flexibility**: Supports plotting both raw m/z channels and aggregated 
  labels (e.g., from clustering or decomposition).

Notes
-----
The module configures global ``matplotlib.rcParams`` on import to ensure 
consistent font sizes and DPI settings across all generated figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple, Union, Literal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
})


def plot_ion_images(
    msicube,
    mz: Union[float, str, Sequence[Union[float, str]]],
    samples: Union[str, Sequence[str]],
    *,
    layer: Optional[str] = None,
    scale_mode: Literal["per_sample", "per_ion", "global", "local"] = "per_sample",
    ncols: int = 3,
    cmap: str = "turbo",
    figsize: Optional[Tuple[float, float]] = None,
    show_axes: bool = True,
    obsm_key: Optional[str] = None,
    label_obsm_key: str = "X_by_label",
) -> None:
    """
    Plot modern, clean, and scientifically robust MSI ion images.

    This function extracts spatial coordinates and intensity data from an 
    AnnData object to reconstruct 2D ion heatmaps. It automatically handles 
    m/z lookup or label-based retrieval (e.g., from factorization results).

    Parameters
    ----------
    msicube : MSICube
        The MSICube instance containing the ``adata`` to plot.
    mz : float, str or sequence of (float or str)
        The m/z value(s) or label(s) to visualize. If floats are provided, 
        the closest m/z in the dataset is selected.
    samples : str or sequence of str
        The sample name(s) to include in the plot (must exist in ``adata.obs['sample']``).
    layer : str, optional
        Name of the ``AnnData.layers`` entry to plot. If ``None``, 
        uses the main data matrix ``adata.X``.
    scale_mode : {"per_sample", "per_ion", "global", "local"}, default "per_sample"
        Intensity scaling strategy:
        
        * ``"per_sample"``: Same scale for all ions of a given sample. Recommended 
          for comparing relative intensities within a tissue.
        * ``"per_ion"``: Same scale for one specific ion across all samples. 
          Ideal for comparing abundance across different tissues.
        * ``"global"``: Same min/max for every single subplot in the figure.
        * ``"local"``: Each image is scaled independently to its own min/max.
    ncols : int, default 3
        Number of columns in the figure grid.
    cmap : str, default "turbo"
        The Matplotlib colormap used for the heatmaps.
    figsize : tuple of float, optional
        Figure size (width, height) in inches. If ``None``, it is 
        automatically calculated based on `ncols`.
    show_axes : bool, default True
        Whether to show pixel coordinates (X/Y) and labels on the axes.
    obsm_key : str, optional
        Specific key in ``adata.obsm`` to use if plotting labels instead of m/z.
    label_obsm_key : str, default "X_by_label"
        Default key in ``adata.obsm`` used for aggregated feature plotting.

    Raises
    ------
    ValueError
        If the MSICube does not contain an AnnData object.
    KeyError
        If the specified `layer` is not found in the AnnData object.

    Notes
    -----
    The underlying ``AnnData`` object is never modified in-place. Intensity 
    values are copied before any scaling so that the source data remains unchanged.

    

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from deltamsi import MSICube
        import anndata

        # Create dummy MSI data
        X = np.random.rand(100, 10)
        spatial = np.array([[i % 10, i // 10] for i in range(100)])
        adata = anndata.AnnData(X, obsm={"spatial": spatial})
        adata.obs['sample'] = "Sample_A"
        adata.var['m/z'] = np.linspace(100, 1000, 10)
        
        class MockCube: adata = None
        cube = MockCube(); cube.adata = adata

        # Plot two m/z values
        from deltamsi.plotting.ion_images import plot_ion_images
        plot_ion_images(cube, mz=[100.0, 500.0], samples="Sample_A", ncols=2)
    """

    if msicube.adata is None:
        raise ValueError("MSICube.adata is empty.")

    adata = msicube.adata

    # ------------------------------------------------------------------
    # 1. Normalize inputs
    # ------------------------------------------------------------------
    target_features = [mz] if isinstance(mz, (float, int, str)) else list(mz)
    target_samples = [samples] if isinstance(samples, str) else list(samples)

    # ------------------------------------------------------------------
    # 2. Resolve m/z vs aggregated labels
    # ------------------------------------------------------------------
    aggregated_key = obsm_key or label_obsm_key
    label_names = adata.uns.get(f"{aggregated_key}_labels")
    label_lookup = {str(name): idx for idx, name in enumerate(label_names or [])}

    using_labels = (
        bool(label_lookup)
        and all(isinstance(f, str) and f in label_lookup for f in target_features)
    )

    if using_labels:
        col_indices = [label_lookup[str(f)] for f in target_features]
        actual_names = [str(f) for f in target_features]
        data_matrix = adata.obsm[aggregated_key]
    else:
        mz_values = (
            adata.var["m/z"].values
            if "m/z" in adata.var
            else adata.var["mz"].values
        )

        col_indices, actual_names = [], []
        for m in target_features:
            idx = int(np.argmin(np.abs(mz_values - float(m))))
            col_indices.append(idx)
            actual_names.append(float(mz_values[idx]))

        if layer is None:
            data_matrix = adata.X
        else:
            if layer not in adata.layers:
                raise KeyError(f"Layer '{layer}' not found in AnnData.layers")
            data_matrix = adata.layers[layer]

    # ------------------------------------------------------------------
    # 3. Extract image data
    # ------------------------------------------------------------------
    plot_data = []

    for sample in target_samples:
        mask = adata.obs["sample"] == sample
        if not np.any(mask):
            continue

        coords = adata.obsm["spatial"][mask]
        coords = coords.values if hasattr(coords, "values") else coords

        intens_src = data_matrix[mask, :]
        intens = (
            intens_src[:, col_indices].toarray()
            if hasattr(intens_src, "toarray")
            else intens_src[:, col_indices]
        )
        intens = np.array(intens, copy=True)

        x, y = coords[:, 0].astype(int), coords[:, 1].astype(int)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        for i, name in enumerate(actual_names):
            img = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan)
            img[y - y_min, x - x_min] = intens[:, i]

            local_max = np.nanmax(img) if not np.all(np.isnan(img)) else 0.0
            local_min = np.nanmin(img) if not np.all(np.isnan(img)) else 0.0

            plot_data.append({
                "img": img,
                "sample": sample,
                "name": name,
                "extent": (x_min, x_max, y_min, y_max),
                "max": local_max,
                "min": local_min,
            })

    if not plot_data:
        return

    # ------------------------------------------------------------------
    # 4. Compute intensity scales
    # ------------------------------------------------------------------
    max_by_sample = defaultdict(lambda: -np.inf)
    max_by_ion = defaultdict(lambda: -np.inf)
    min_by_sample = defaultdict(lambda: np.inf)
    min_by_ion = defaultdict(lambda: np.inf)
    global_max = -np.inf
    global_min = np.inf

    for d in plot_data:
        max_by_sample[d["sample"]] = max(max_by_sample[d["sample"]], d["max"])
        max_by_ion[d["name"]] = max(max_by_ion[d["name"]], d["max"])
        min_by_sample[d["sample"]] = min(min_by_sample[d["sample"]], d["min"])
        min_by_ion[d["name"]] = min(min_by_ion[d["name"]], d["min"])
        global_max = max(global_max, d["max"])
        global_min = min(global_min, d["min"])

    # ------------------------------------------------------------------
    # 5. Layout
    # ------------------------------------------------------------------
    n_plots = len(plot_data)
    nrows = int(np.ceil(n_plots / ncols))

    if figsize is None:
        figsize = (5.2 * ncols, 5.0 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        constrained_layout=True,
    )

    fig.set_facecolor("white")
    axes = np.atleast_1d(axes).flatten()

    # ------------------------------------------------------------------
    # 6. Plot images
    # ------------------------------------------------------------------
    for i, ax in enumerate(axes):
        if i >= n_plots:
            ax.axis("off")
            continue

        d = plot_data[i]

        if scale_mode == "per_sample":
            vmin, vmax = min_by_sample[d["sample"]], max_by_sample[d["sample"]]
        elif scale_mode == "per_ion":
            vmin, vmax = min_by_ion[d["name"]], max_by_ion[d["name"]]
        elif scale_mode == "global":
            vmin, vmax = global_min, global_max
        else:  # local
            vmin, vmax = d["min"], d["max"]

        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = 0.0
        if vmin == vmax:
            eps = 1e-12 if vmin == 0 else abs(vmin) * 1e-6
            eps = max(eps, 1e-12)
            vmin -= eps / 2
            vmax += eps / 2

        im = ax.imshow(
            d["img"],
            origin="lower",
            cmap=cmap,
            interpolation="none",
            extent=d["extent"],
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        ion_label = d["name"] if using_labels else f"{d['name']:.4f} m/z"
        ax.set_title(
            f"{ion_label}\nSample: {d['sample']}",
            pad=12,
            fontweight="bold",
        )

        if show_axes:
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="6%", pad=0.55)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Ion intensity (a.u.)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        cbar.formatter.set_powerlimits((-2, 3))
        cbar.update_ticks()

    # ------------------------------------------------------------------
    # 7. Figure title
    # ------------------------------------------------------------------
    fig.suptitle(
        f"Mass Spectrometry Imaging – Ion distributions "
        f"(scale: {scale_mode.replace('_', ' ')})",
        fontsize=17,
        fontweight="bold",
    )

    plt.show()
