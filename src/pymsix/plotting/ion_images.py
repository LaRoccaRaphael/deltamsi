import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple, Union, Literal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict


# ------------------------------------------------------------------
# Global plotting style (modern & scientific)
# ------------------------------------------------------------------
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
    scale_mode: Literal["per_sample", "per_ion", "global", "local"] = "per_sample",
    ncols: int = 3,
    cmap: str = "turbo",
    figsize: Optional[Tuple[float, float]] = None,
    show_axes: bool = True,
    obsm_key: Optional[str] = None,
    label_obsm_key: str = "X_by_label",
) -> None:
    """
    Plot modern, clean and scientifically robust MSI ion images.

    Parameters
    ----------
    scale_mode:
        - "per_sample": same scale for all ions of a given sample (RECOMMENDED)
        - "per_ion":    same scale for one ion across samples
        - "global":     same scale for all plots
        - "local":      independent scale per image
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

        data_matrix = adata.X

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

        x, y = coords[:, 0].astype(int), coords[:, 1].astype(int)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        for i, name in enumerate(actual_names):
            img = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan)
            img[y - y_min, x - x_min] = intens[:, i]

            local_max = np.nanmax(img) if not np.all(np.isnan(img)) else 0.0

            plot_data.append({
                "img": img,
                "sample": sample,
                "name": name,
                "extent": (x_min, x_max, y_min, y_max),
                "max": local_max,
            })

    if not plot_data:
        return

    # ------------------------------------------------------------------
    # 4. Compute intensity scales
    # ------------------------------------------------------------------
    max_by_sample = defaultdict(float)
    max_by_ion = defaultdict(float)
    global_max = 0.0

    for d in plot_data:
        max_by_sample[d["sample"]] = max(max_by_sample[d["sample"]], d["max"])
        max_by_ion[d["name"]] = max(max_by_ion[d["name"]], d["max"])
        global_max = max(global_max, d["max"])

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
            vmax = max_by_sample[d["sample"]]
        elif scale_mode == "per_ion":
            vmax = max_by_ion[d["name"]]
        elif scale_mode == "global":
            vmax = global_max
        else:  # local
            vmax = d["max"]

        vmax = max(vmax, 1e-12)

        im = ax.imshow(
            d["img"],
            origin="lower",
            cmap=cmap,
            interpolation="none",
            extent=d["extent"],
            vmin=0,
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
