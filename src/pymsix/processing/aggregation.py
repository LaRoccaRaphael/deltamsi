"""Aggregation utilities for MSI cubes."""

from __future__ import annotations

from typing import Literal, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:  # pragma: no cover
    from pymsix.core.msicube import MSICube

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
    Aggregate variables (m/z columns) that share the same label in ``adata.var[label_col]``.

    The aggregated ion images are stored in ``adata.obsm[obsm_key]`` with shape
    ``(n_obs, n_labels)``. Label names (column order) are stored in
    ``adata.uns[f"{obsm_key}_labels"]``.

    Parameters
    ----------
    msicube : MSICube
        The cube containing the AnnData object to aggregate.
    label_col : str
        Column in ``adata.var`` containing the labels used to group variables.
    layer : str | None, default None
        If provided, aggregate ``adata.layers[layer]`` instead of ``adata.X``.
    agg : {"mean", "median", "max"}, default "mean"
        Aggregation strategy applied across variables within a label.
    obsm_key : str, default "X_by_label"
        Key under which the aggregated matrix is stored in ``adata.obsm``.
    dropna : bool, default True
        Whether to drop variables with missing labels. If ``False``, missing labels are
        replaced with "NA".
    keep_order : bool, default True
        Preserve the first-seen order of labels instead of sorting them.
    as_df : bool, default False
        Store the aggregated matrix as a pandas DataFrame rather than a NumPy array.
    dtype : dtype or type, default numpy.float32
        Data type of the aggregated matrix.

    Returns
    -------
    pandas.Index
        Index of the aggregated label names, named after ``label_col``.

    Raises
    ------
    ValueError
        If ``msicube.adata`` is missing.
    KeyError
        If ``label_col`` is not present in ``adata.var`` or ``layer`` is not found.
    ValueError
        If ``agg='median'`` is used with a sparse matrix.
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

