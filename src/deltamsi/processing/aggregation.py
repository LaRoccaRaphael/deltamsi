"""
MSI Data Aggregation Utilities
==============================

This module provides tools for grouping and summarizing variables (m/z channels) 
within an MSI dataset. It is primarily used to collapse redundant features 
or to create "component images" based on clustering or decomposition labels.

The main function allows for spatial intensity matrices to be aggregated by 
metadata labels, facilitating downstream visualization of molecular families 
or co-localized ions.
"""

from __future__ import annotations

from typing import Literal, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:  # pragma: no cover
    from deltamsi.core.msicube import MSICube

Agg = Literal["mean", "median", "max"]


def aggregate_vars_by_label(
    msicube: "MSICube",
    label_col: str,
    *,
    layer: Optional[str] = None,
    agg: Agg = "mean",
    obsm_key: str = "X_by_label",
    dropna: bool = True,
    keep_order: bool = True,
    as_df: bool = False,
    dtype: Union[np.dtype, type] = np.float32,
) -> pd.Index:
    """
    Aggregate variables (m/z columns) that share the same label in metadata.

    This function groups m/z channels based on a column in ``adata.var`` 
    and computes a summary statistic (mean, median, or max) for each group. 
    The resulting matrix represents aggregated ion distributions (images) 
    stored in the ``obsm`` slot of the AnnData object.

    Parameters
    ----------
    msicube : MSICube
        The MSICube instance containing the MSI data to aggregate.
    label_col : str
        Column name in ``adata.var`` containing labels used for grouping 
        (e.g., "cluster_id", "chemical_family").
    layer : str, optional
        If provided, aggregates values from ``adata.layers[layer]`` instead 
        of the main data matrix ``adata.X``.
    agg : {"mean", "median", "max"}, default "mean"
        The aggregation strategy to apply across the variables of each group.
        Note: "median" is not supported for sparse matrices.
    obsm_key : str, default "X_by_label"
        Key under which the aggregated matrix will be stored in ``adata.obsm``.
    dropna : bool, default True
        If True, variables with NaN in `label_col` are ignored. If False, 
        they are grouped under the label "NA".
    keep_order : bool, default True
        If True, the columns in the output matrix follow the order in which 
        labels first appear in ``adata.var``. If False, labels are sorted 
        alphabetically.
    as_df : bool, default False
        If True, stores a ``pandas.DataFrame`` in ``obsm``; otherwise, 
        stores a ``numpy.ndarray``.
    dtype : dtype or type, default numpy.float32
        The numerical precision of the resulting aggregated matrix.

    Returns
    -------
    pd.Index
        An index containing the unique labels in the order they appear in 
        the aggregated matrix.

    Raises
    ------
    ValueError
        If ``msicube.adata`` is None, if an unknown aggregation method is 
        provided, or if "median" is requested for sparse data.
    KeyError
        If `label_col` is not found in ``adata.var`` or if the specified 
        `layer` is missing.

    Notes
    -----
    Aggregation is a key step after mass clustering. By computing the 
    average intensity of all ions within a cluster, you obtain a single 
    spatial distribution that represents the entire molecular family.

    

    The result is stored in ``adata.obsm[obsm_key]`` with a shape 
    of ``(n_pixels, n_unique_labels)``. Metadata about the process is 
    stored in ``adata.uns[f"{obsm_key}_source"]``.

    Examples
    --------
    >>> from deltamsi.processing.aggregation import aggregate_vars_by_label
    >>> # Aggregate ions by their cluster ID to get cluster-specific images
    >>> labels = aggregate_vars_by_label(
    ...     msicube, 
    ...     label_col="cluster", 
    ...     agg="max"
    ... )
    >>> print(f"Aggregated into {len(labels)} unique groups")
    
    >>> # Access the resulting matrix
    >>> cluster_images = msicube.adata.obsm["X_by_label"]
    """

    if msicube.adata is None:
        raise ValueError("MSICube.adata is None. Run data extraction first.")

    adata = msicube.adata

    if label_col not in adata.var.columns:
        raise KeyError(f"{label_col!r} not found in adata.var.columns")

    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        X = adata.layers[layer]

    labels = adata.var[label_col]

    if dropna:
        mask = labels.notna()
        labels = labels[mask]
        var_idx = np.where(mask.to_numpy())[0]
    else:
        labels = labels.fillna("NA")
        var_idx = np.arange(adata.n_vars)

    labels = labels.astype(str).to_numpy()

    if keep_order:
        seen = set()
        uniq = []
        for lab in labels:
            if lab not in seen:
                seen.add(lab)
                uniq.append(lab)
        uniq = np.array(uniq, dtype=object)
    else:
        uniq = np.unique(labels)

    n_obs = adata.n_obs
    n_lab = len(uniq)
    out = np.empty((n_obs, n_lab), dtype=dtype)
    is_sparse = sparse.issparse(X)

    for j, lab in enumerate(uniq):
        cols = var_idx[labels == lab]
        if cols.size == 0:
            out[:, j] = np.nan
            continue

        sub = X[:, cols]

        if agg == "mean":
            if is_sparse:
                out[:, j] = np.asarray(sub.mean(axis=1)).ravel()
            else:
                out[:, j] = np.nanmean(sub, axis=1)

        elif agg == "max":
            if is_sparse:
                out[:, j] = np.asarray(sub.max(axis=1)).ravel()
            else:
                out[:, j] = np.nanmax(sub, axis=1)

        elif agg == "median":
            if is_sparse:
                raise ValueError(
                    "median aggregation is not supported for sparse matrices "
                    "without densifying. Consider agg='mean'/'max' or convert to dense."
                )
            out[:, j] = np.nanmedian(sub, axis=1)

        else:
            raise ValueError(f"Unknown agg={agg!r}. Use 'mean', 'median', or 'max'.")

    if as_df:
        adata.obsm[obsm_key] = pd.DataFrame(out, index=adata.obs_names, columns=uniq)
    else:
        adata.obsm[obsm_key] = out

    adata.uns[f"{obsm_key}_labels"] = uniq.tolist()
    adata.uns[f"{obsm_key}_source"] = {
        "label_col": label_col,
        "layer": layer,
        "agg": agg,
    }

    return pd.Index(uniq, name=label_col)

