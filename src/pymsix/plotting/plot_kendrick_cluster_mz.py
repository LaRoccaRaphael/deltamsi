"""
Kendrick Analysis for Clustering Results
========================================

This module provides tools to bridge mass clustering results with Kendrick Mass 
Defect (KMD) analysis. It allows for the construction of structured DataFrames 
and high-quality visualizations that highlight how clusters align with 
homologous series.

The module automatically handles the synchronization between clustering labels 
and Kendrick coordinates stored in ``AnnData.varm``.
"""

from __future__ import annotations

import logging
from typing import Sequence, Mapping, Optional, Tuple, List

import anndata as ad
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt

from pymsix.processing.kendrick import compute_kendrick_varm, default_kendrick_varm_key

logger = logging.getLogger(__name__)


def _ensure_kendrick_coordinates(
    adata: ad.AnnData,
    *,
    base: str | float | Tuple[float, float],
    kmd_mode: str,
    kendrick_varm_key: Optional[str],
    mz_key: str,
) -> tuple[str, NDArray[np.floating], Optional[dict]]:
    """
    Ensure that Kendrick coordinates exist in ``adata.varm`` and return them.

    This helper checks for existing Kendrick coordinates under the specified or 
    default key. If missing, it triggers an automatic computation and updates 
    the ``AnnData`` object. It also validates that existing data matches the 
    requested calculation mode.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object containing variable (m/z) information.
    base : str or float or tuple of (float, float)
        The Kendrick base used for scaling (e.g., "CH2", 14.0156, or (14.0156, 14)).
    kmd_mode : {"fraction", "defect"}
        The type of Kendrick coordinate to ensure:
        - "fraction": The fractional part of the Kendrick mass.
        - "defect": Nominal Kendrick mass minus exact Kendrick mass.
    kendrick_varm_key : str, optional
        Specific key in ``adata.varm`` to look for. If None, a default key 
        is generated using the base and mode.
    mz_key : str
        Key in ``adata.var`` containing the m/z values.

    Returns
    -------
    target_key : str
        The key in ``adata.varm`` where the coordinates are stored.
    coords : NDArray[np.floating]
        2D array of shape ``(n_vars, 2)`` containing [KM, KMD/KF].
    info : dict, optional
        Metadata associated with the calculation, retrieved from ``adata.uns``.

    Raises
    ------
    ValueError
        - If ``kmd_mode`` is not 'fraction' or 'defect'.
        - If existing coordinates are not 2D or have fewer than 2 columns.
        - If the stored ``kmd_mode`` in metadata conflicts with the requested mode.
    """

    if kmd_mode not in {"fraction", "defect"}:
        raise ValueError("kmd_mode must be 'fraction' or 'defect'")

    target_key = kendrick_varm_key or default_kendrick_varm_key(base, kmd_mode)

    if target_key not in adata.varm:
        logger.info(
            "Kendrick coordinates '%s' not found; computing them with base %s.",
            target_key,
            base,
        )
        target_key = compute_kendrick_varm(
            adata,
            mz_key=mz_key,
            base=base,
            kmd_mode=kmd_mode,
            varm_key=target_key,
        )

    coords: NDArray[np.floating] = np.asarray(adata.varm[target_key])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"adata.varm['{target_key}'] must be a 2D array with KM and KMD columns"
        )

    info = adata.uns.get(f"{target_key}_info") if hasattr(adata, "uns") else None
    if info is not None:
        stored_mode = info.get("kmd_mode")
        if stored_mode and stored_mode != kmd_mode:
            raise ValueError(
                f"adata.varm['{target_key}'] was computed with kmd_mode='{stored_mode}',"
                f" but kmd_mode='{kmd_mode}' was requested."
            )
    return target_key, coords, info


def kendrick_df_from_clustering(
    masses: Sequence[float],
    clustering_result: Mapping[str, object],
    *,
    adata: ad.AnnData,
    family: Optional[Sequence[object]] = None,
    base: str | float | Tuple[float, float] = "CH2",
    kmd_mode: str = "fraction",
    kendrick_varm_key: Optional[str] = None,
    mz_key: str = "mz",
    mass_col: str = "exact_molecular_weight",
) -> pd.DataFrame:
    """
    Build a DataFrame combining clustering labels and Kendrick coordinates.

    This function aligns mass values with their corresponding cluster IDs and 
    retrieves (or computes) Kendrick coordinates from the provided AnnData object.

    Parameters
    ----------
    masses : sequence of float
        The list of m/z values or exact masses that were clustered.
    clustering_result : mapping
        A dictionary-like object containing a ``'labels'`` key (e.g., the output 
        of candidate-based clustering).
    adata : ad.AnnData
        The AnnData object containing variable metadata. Must align with `masses`.
    family : sequence of object, optional
        Secondary grouping labels (e.g., chemical families) to include in the 
        resulting DataFrame.
    base : str, float or tuple, default "CH2"
        The Kendrick base used for scaling (e.g., "CH2", "H2O").
    kmd_mode : {"fraction", "defect"}, default "fraction"
        The type of Kendrick Mass Defect to calculate/retrieve.
    kendrick_varm_key : str, optional
        Specific key in ``adata.varm`` to look for. If None, a default key is 
        generated based on `base` and `kmd_mode`.
    mz_key : str, default "mz"
        The column name in ``adata.var`` used for m/z values during computation.
    mass_col : str, default "exact_molecular_weight"
        The name for the mass column in the output DataFrame.

    Returns
    -------
    pd.DataFrame
        A table containing masses, clusters, families (if provided), 
        ``kendrick_mass``, and the requested KMD column.
        The DataFrame includes a ``kendrick_info`` attribute in ``.attrs``.

    Raises
    ------
    ValueError
        If lengths of inputs do not match or if Kendrick coordinates are 
        incompatible with the requested mode.
    """

    masses_arr: NDArray[np.floating] = np.asarray(masses, dtype=float)
    labels: NDArray[np.integer] = np.asarray(clustering_result["labels"], dtype=int)

    if masses_arr.size != labels.size:
        raise ValueError("masses and clustering_result['labels'] must have same length")

    varm_key_used, coords, info = _ensure_kendrick_coordinates(
        adata,
        base=base,
        kmd_mode=kmd_mode,
        kendrick_varm_key=kendrick_varm_key,
        mz_key=mz_key,
    )

    if coords.shape[0] != masses_arr.size:
        raise ValueError(
            "Kendrick coordinates length does not match the number of masses/variables."
        )

    df = pd.DataFrame({mass_col: masses_arr, "cluster": labels})

    if family is not None:
        fam_arr: NDArray[np.object_] = np.asarray(family, dtype=object)
        if fam_arr.size != masses_arr.size:
            raise ValueError("'family' must have same length as masses")
        df["family"] = fam_arr

    df["kendrick_mass"] = coords[:, 0]
    if kmd_mode == "fraction":
        df["kmd_fraction"] = coords[:, 1]
        df["kmd_defect"] = np.nan
    else:
        df["kmd_fraction"] = np.nan
        df["kmd_defect"] = coords[:, 1]

    kendrick_info = {
        "varm_key": varm_key_used,
        "kmd_mode": kmd_mode,
    }
    if info:
        kendrick_info.update({
            "scale": info.get("scale"),
            "base_exact": info.get("base_exact"),
            "base_nominal": info.get("base_nominal"),
            "base": info.get("base", base),
        })
    df.attrs["kendrick_info"] = {k: v for k, v in kendrick_info.items() if v is not None}
    return df


# ---- plotting (with cluster filters) ----
def plot_kendrick_from_clustering(
    masses: Sequence[float],
    clustering_result: Mapping[str, object],
    *,
    adata: ad.AnnData,
    kendrick_varm_key: Optional[str] = None,
    family: Optional[Sequence[object]] = None,
    primary_color_by: str = "cluster",
    base: str | float | Tuple[float, float] = "CH2",
    mass_col: str = "exact_molecular_weight",
    # axes & style
    x_axis: str = "kendrick_mass",  # 'kendrick_mass' or 'm_over_z'
    kmd_mode: str = "fraction",  # 'fraction' or 'defect'
    point_size: float = 24,
    alpha: float = 0.9,
    hgrid_step: float = 1.0,  # KMD grid step (in 'fraction' units). For 'defect', use e.g. 0.1
    jitter: float = 0.0,  # horizontal jitter on x-axis
    annotate: bool = False,
    max_ann_per_group: int = 0,  # annotate up to N points per color group (by label index)
    # cluster filters
    top_k_clusters: Optional[int] = None,  # keep only K largest clusters
    selected_clusters: Optional[
        List[int]
    ] = None,  # explicit cluster IDs to keep (e.g. [-1,0,3])
    include_minus1_in_top: bool = True,
    min_cluster_size: int = 1,
    # layout
    two_panels: bool = True,  # second panel colored by 'family' (if provided)
    figsize: Tuple[float, float] = (9, 4.5),
) -> tuple[plt.Figure, list[plt.Axes] | plt.Axes, pd.DataFrame]:
    """
    Generate Kendrick plots directly from mass clustering outputs.

    This function produces scatter plots where points are colored by their 
    assigned cluster. It supports advanced filtering (top K clusters, 
    size-based pruning) and optional secondary panels for chemical families.

    Parameters
    ----------
    masses : sequence of float
        The list of m/z values that were clustered.
    clustering_result : mapping
        The output from clustering containing labels.
    adata : ad.AnnData
        The AnnData object where Kendrick coordinates are stored or will be computed.
    kendrick_varm_key : str, optional
        Key to retrieve pre-computed coordinates from ``adata.varm``.
    family : sequence of object, optional
        Labels for a second panel (e.g., compound classes). Only used if 
        `two_panels` is True.
    primary_color_by : {"cluster", "family"}, default "cluster"
        Which column to use for coloring the primary panel when rendering a
        single plot (``two_panels=False``). Requires ``family`` data when set
        to ``"family"``.
    base : str, float or tuple, default "CH2"
        The Kendrick base for the X-axis scaling.
    mass_col : str, default "exact_molecular_weight"
        The label for the mass column.
    x_axis : {"kendrick_mass", "m_over_z"}, default "kendrick_mass"
        Which value to display on the horizontal axis.
    kmd_mode : {"fraction", "defect"}, default "fraction"
        The KMD calculation mode for the vertical axis.
    point_size : float, default 24
        Size of the scatter markers.
    alpha : float, default 0.9
        Transparency of the points.
    hgrid_step : float, default 1.0
        The interval for horizontal dashed grid lines.
    jitter : float, default 0.0
        Standard deviation of random noise added to the X-axis to reduce 
        point overlap.
    annotate : bool, default False
        If True, adds row indices as text labels to points.
    max_ann_per_group : int, default 0
        Maximum number of points to annotate within each cluster/family.
    top_k_clusters : int, optional
        If set, only the `k` largest clusters are displayed.
    selected_clusters : list of int, optional
        Explicit list of cluster IDs to plot (e.g., [1, 5, 12]).
    include_minus1_in_top : bool, default True
        Whether to consider cluster "-1" (noise/unassigned) when 
        calculating the top `k` clusters.
    min_cluster_size : int, default 1
        Clusters with fewer members than this value will be filtered out.
    two_panels : bool, default True
        If True and `family` is provided, creates a side-by-side plot 
        (Cluster vs. Family).
    figsize : tuple of float, default (9, 4.5)
        Width and height of the figure in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    axes : list of matplotlib.axes.Axes or Axes
        The axes containing the plots.
    df_used : pd.DataFrame
        The filtered DataFrame containing all plotted data and Kendrick coordinates.

    Notes
    -----
    
    
    The Kendrick Mass Defect (KMD) allows the identification of homologous 
    series. In this plot, points belonging to the same series (differing 
    only by the `base` unit) will align horizontally.

    Examples
    --------
    >>> import anndata as ad
    >>> from pymsix.plotting.kendrick import plot_kendrick_from_clustering
    >>> # Assuming clustering_res is the output of mass clustering
    >>> fig, ax, df = plot_kendrick_from_clustering(
    ...     masses=my_mass_list,
    ...     clustering_result=clustering_res,
    ...     adata=my_adata,
    ...     top_k_clusters=10,
    ...     base="CH2"
    ... )
    >>> fig.show()
    """
    # Build full DF
    df = kendrick_df_from_clustering(
        masses,
        clustering_result,
        adata=adata,
        kendrick_varm_key=kendrick_varm_key,
        family=family,
        base=base,
        kmd_mode=kmd_mode,
        mz_key=mass_col,
        mass_col=mass_col,
    )

    # Apply cluster filters
    if selected_clusters is not None:
        allow = set(int(c) for c in selected_clusters)
    elif top_k_clusters is not None:
        vc = df["cluster"].value_counts().sort_values(ascending=False)
        if not include_minus1_in_top:
            vc = vc[vc.index != -1]
        if min_cluster_size > 1:
            vc = vc[vc >= int(min_cluster_size)]
        allow = set(int(c) for c in vc.head(int(top_k_clusters)).index.tolist())
    else:
        allow = None
    if allow is not None:
        df = df[df["cluster"].astype(int).isin(allow)].reset_index(drop=True)
        if df.empty:
            raise ValueError("No points remain after cluster filtering.")

    # Choose coordinates
    x = (
        df["kendrick_mass"].to_numpy(float)
        if x_axis == "kendrick_mass"
        else df[mass_col].to_numpy(float)
    )
    y = (
        df["kmd_fraction"].to_numpy(float)
        if kmd_mode == "fraction"
        else df["kmd_defect"].to_numpy(float)
    )
    if jitter:
        rng = np.random.default_rng(0)
        x = x + rng.normal(0.0, float(jitter), size=len(x))

    if primary_color_by not in {"cluster", "family"}:
        raise ValueError("primary_color_by must be 'cluster' or 'family'")
    if primary_color_by == "family" and "family" not in df.columns:
        raise ValueError("primary_color_by='family' requires family labels.")

    # Set up figure(s)
    if two_panels and ("family" in df.columns):
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=140, sharey=True)
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=140)
        axes = [ax1]

    # Colormaps
    cmap = plt.get_cmap("tab20")
    palette = list(getattr(cmap, "colors", [])) or [
        cmap(i / max(1, cmap.N - 1)) for i in range(cmap.N)
    ]

    # ---- Panel A: color by cluster or family ----
    df["_x"] = x
    df["_y"] = y
    if primary_color_by == "family":
        groups = sorted(df["family"].astype(str).unique())
        color_map = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
        for g in groups:
            gd = df[df["family"].astype(str) == g]
            ax1.scatter(
                gd["_x"],
                gd["_y"],
                s=point_size,
                alpha=alpha,
                color=color_map[g],
                label=str(g),
            )
            if annotate and max_ann_per_group > 0:
                for idx in gd.head(max_ann_per_group).index:
                    ax1.annotate(
                        str(idx),
                        (gd.loc[idx, "_x"], gd.loc[idx, "_y"]),
                        fontsize=8,
                        alpha=0.8,
                    )
        ax1.set_title("Kendrick by FAMILY")
        ax1.legend(
            title="family", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False
        )
    else:
        groups = sorted(df["cluster"].astype(int).unique())
        color_map = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
        for g in groups:
            gd = df[df["cluster"].astype(int) == g]
            ax1.scatter(
                gd["_x"],
                gd["_y"],
                s=point_size,
                alpha=alpha,
                color=color_map[g],
                label=str(g),
            )
            if annotate and max_ann_per_group > 0:
                # annotate first N points of each group (by row order)
                for idx in gd.head(max_ann_per_group).index:
                    ax1.annotate(
                        str(idx),
                        (gd.loc[idx, "_x"], gd.loc[idx, "_y"]),
                        fontsize=8,
                        alpha=0.8,
                    )
        ax1.set_title("Kendrick by CLUSTER")
        ax1.legend(
            title="cluster",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=False,
        )

    # grid lines for KMD
    if kmd_mode == "fraction":
        ymin, ymax = 0.0, 1.0
        ax1.set_ylim(ymin, ymax)
        if hgrid_step and hgrid_step > 0:
            for k in np.arange(0, 1 + 1e-9, hgrid_step):
                ax1.axhline(k, ls=":", lw=0.8, c="grey", alpha=0.4, zorder=0)
    else:  # defect in [-0.5,0.5]
        ax1.set_ylim(-0.5, 0.5)
        if hgrid_step and hgrid_step > 0:
            for k in np.arange(-0.5, 0.5 + 1e-9, hgrid_step):
                ax1.axhline(k, ls=":", lw=0.8, c="grey", alpha=0.4, zorder=0)

    ax1.set_xlabel(
        "Kendrick mass (base {})".format(base) if x_axis == "kendrick_mass" else "mz"
    )
    ax1.set_ylabel("KMD ({})".format(kmd_mode))
    ax1.grid(True, ls=":", alpha=0.25)

    # ---- Panel B: color by family (optional) ----
    if len(axes) == 2:
        ax2 = axes[1]
        fams = sorted(df["family"].astype(str).unique())
        color_map_f = {f: palette[i % len(palette)] for i, f in enumerate(fams)}
        for f in fams:
            gd = df[df["family"].astype(str) == f]
            ax2.scatter(
                gd["_x"],
                gd["_y"],
                s=point_size,
                alpha=alpha,
                color=color_map_f[f],
                label=str(f),
            )
            if annotate and max_ann_per_group > 0:
                for idx in gd.head(max_ann_per_group).index:
                    ax2.annotate(
                        str(idx),
                        (gd.loc[idx, "_x"], gd.loc[idx, "_y"]),
                        fontsize=8,
                        alpha=0.8,
                    )
        ax2.set_xlabel(
            "Kendrick mass (base {})".format(base)
            if x_axis == "kendrick_mass"
            else "mz"
        )
        ax2.set_title("Kendrick by FAMILY")
        ax2.grid(True, ls=":", alpha=0.25)
        ax2.legend(
            title="family", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False
        )

    # housekeeping
    for a in axes:
        a.margins(x=0.02, y=0.02)
    fig.tight_layout()
    # drop helper cols before returning
    df = df.drop(columns=["_x", "_y"])
    return fig, axes, df
