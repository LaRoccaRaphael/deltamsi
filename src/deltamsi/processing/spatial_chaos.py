"""
Spatial Chaos Analysis Module
=============================

This module provides tools to quantify and compare spatial structure in 
Mass Spectrometry Imaging (MSI) data through spatial chaos scores. It is 
designed to work with ``AnnData`` objects and raw NumPy arrays.

The core metric, the Spatial Chaos Score ($S$), measures the fragmentation 
of an ion image across multiple intensity thresholds. A score of 1 
represents a perfectly structured image (single cluster), while 0 
represents total spatial randomness (high fragmentation).

Functions
---------
spatial_chaos_score
    Computes the chaos score for a single 2D ion image.
spatial_chaos_fold_change
    Computes structure-based fold change between two experimental groups.
compute_spatial_chaos_matrix
    Calculates chaos scores for all ions and samples in an AnnData object.
spatial_chaos_fold_change_from_adata
    Wrapper to compute fold change directly from stored AnnData metadata.

Examples
--------
>>> import numpy as np
>>> # Generate a dummy structured image (a central square)
>>> img = np.zeros((10, 10))
>>> img[3:7, 3:7] = 1.0
>>> score = spatial_chaos_score(img)
>>> print(f"Chaos score: {score:.2f}")
Chaos score: 0.94
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from typing import Any, Dict, List, Optional, Sequence, Tuple
from scipy.ndimage import label


def spatial_chaos_fold_change(
    chaos_scores: np.ndarray,
    sample_groups: Sequence,
    control_label: Any,
    interaction_label: Any,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    Compute structure-based fold change $FC_S(m)$ from spatial chaos scores.

    Parameters
    ----------
    chaos_scores : np.ndarray
        2D array of shape ``(n_mz, n_samples)`` containing spatial chaos scores.
    sample_groups : Sequence
        Sequence of length ``n_samples`` assigning a group label to each column.
    control_label : Any
        The label identifying the control group in ``sample_groups``.
    interaction_label : Any
        The label identifying the treated/interaction group in ``sample_groups``.
    eps : float, default 1e-6
        Small constant to avoid division by zero if the control score is 0.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - ``"S_control_max"``: Max chaos score per m/z in control group.
        - ``"S_interaction_max"``: Max chaos score per m/z in interaction group.
        - ``"FC_S"``: The spatial fold change ratio.

    Examples
    --------
    >>> scores = np.array([[0.8, 0.9, 0.1, 0.2], [0.5, 0.5, 0.6, 0.7]])
    >>> groups = ["interaction", "interaction", "control", "control"]
    >>> res = spatial_chaos_fold_change(scores, groups, "control", "interaction")
    >>> res["FC_S"]
    array([4.5, 0.71428571])
    """

    chaos_scores = np.asarray(chaos_scores, dtype=float)
    sample_groups = np.asarray(sample_groups)

    if chaos_scores.ndim != 2:
        raise ValueError("chaos_scores must be 2D with shape (n_mz, n_samples)")

    n_mz, n_samples = chaos_scores.shape
    if sample_groups.shape[0] != n_samples:
        raise ValueError("sample_groups length must match chaos_scores.shape[1]")

    ctrl_mask = sample_groups == control_label
    inter_mask = sample_groups == interaction_label

    if ctrl_mask.sum() == 0:
        raise ValueError(f"No samples found with control_label={control_label!r}")
    if inter_mask.sum() == 0:
        raise ValueError(
            f"No samples found with interaction_label={interaction_label!r}"
        )

    S_control_max = np.max(chaos_scores[:, ctrl_mask], axis=1)
    S_interaction_max = np.max(chaos_scores[:, inter_mask], axis=1)

    denom = np.maximum(S_control_max, eps)
    FC_S = S_interaction_max / denom

    return {
        "S_control_max": S_control_max,
        "S_interaction_max": S_interaction_max,
        "FC_S": FC_S,
    }


def spatial_chaos_score(
    image: np.ndarray,
    n_thresholds: int = 30,
) -> float:
    """
    Compute the Spatial Chaos Score $S$ for a 2D ion image.

    The score is based on the ratio of the number of connected components 
    (clusters) to the number of positive pixels across multiple intensity 
    thresholds.

    Parameters
    ----------
    image : np.ndarray
        2D array of non-negative intensities (ion image).
    n_thresholds : int, default 30
        Number of intensity levels $N$ used to binarize the image.

    Returns
    -------
    float
        The chaos score in $[0, 1]$. Returns ``0.0`` if the image has no signal.

    Notes
    -----
    The score is calculated as:
    $$S = 1 - \\frac{\\sum_{k=1}^{N} C_k}{\\sum_{k=1}^{N} P_k}$$
    where $C_k$ is the number of 4-connected clusters and $P_k$ is the number 
    of positive pixels at threshold $k$.
    """

    im = np.asarray(image, dtype=float)

    if im.ndim != 2:
        raise ValueError("spatial_chaos_score expects a 2D array")

    # Ignore NaNs when computing max; treat NaNs as 0 elsewhere
    max_val = np.nanmax(im)
    if not np.isfinite(max_val) or max_val <= 0:
        # No signal → return zero chaos
        return 0.0

    # 1) Normalization to [0, 1]
    im_norm = np.nan_to_num(im, nan=0.0) / max_val

    # 2) Thresholds evenly spaced in (0, 1):
    #    t_k = k / (N + 1), k = 1..N
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]

    # 4-connectivity (diamond) structure
    structure = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        dtype=bool,
    )

    C_tot = 0.0
    P_tot = 0.0

    # 3) Loop over thresholds
    for t in thresholds:
        # Binary image: B_k(x,y) = 1 if I_norm >= t, else 0
        B = im_norm >= t

        # P_k: number of positive pixels
        P_k = int(np.count_nonzero(B))
        if P_k == 0:
            # No signal at this threshold: contributes 0 to both sums
            continue

        # C_k: number of connected components (4-connectivity)
        _, C_k = label(B, structure=structure)

        C_tot += C_k
        P_tot += P_k

    if P_tot == 0:
        # No positive pixels across thresholds: define as zero chaos
        return 0.0

    # 5) Spatial chaos score
    S = 1.0 - (C_tot / P_tot)

    # Numerical safety clamp
    S = float(max(0.0, min(1.0, S)))
    return S


def _get_data_matrix(adata: ad.AnnData, layer: Optional[str]) -> Any:
    """
    Extract the main data matrix (X) or a specific layer from an AnnData object.

    This helper ensures that the requested data exists and provides clear 
    error messages if the slots are empty or missing.

    Parameters
    ----------
    adata : ad.AnnData
        The annotated data object from which to extract the matrix.
    layer : str, optional
        The name of the layer to retrieve. If None, the function attempts 
        to return ``adata.X``.

    Returns
    -------
    Any
        The data matrix, typically a ``numpy.ndarray`` or a ``scipy.sparse`` 
        matrix.

    Raises
    ------
    ValueError
        If `layer` is None and ``adata.X`` is also None.
    KeyError
        If a specific `layer` string is provided but does not exist in 
        ``adata.layers``.

    Examples
    --------
    >>> import anndata as ad
    >>> import numpy as np
    >>> adata = ad.AnnData(np.ones((3, 2)))
    >>> # Extract the main matrix X
    >>> X = _get_data_matrix(adata, layer=None)
    >>> # Extract a specific layer
    >>> adata.layers["raw"] = np.zeros((3, 2))
    >>> raw_layer = _get_data_matrix(adata, layer="raw")
    """
    if layer is None:
        if adata.X is None:
            raise ValueError("AnnData.X is empty; provide a valid layer instead.")
        return adata.X

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in AnnData.layers")
    return adata.layers[layer]


def compute_spatial_chaos_matrix(
    adata: ad.AnnData,
    *,
    layer: Optional[str] = None,
    obsm_key: str = "spatial",
    sample_key: str = "sample",
    n_thresholds: int = 30,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute spatial chaos scores for every variable (ion) and MSI sample in an AnnData.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object.
    layer : str, optional
        Layer to use for intensities. If None, uses ``adata.X``.
    obsm_key : str, default "spatial"
        Key in ``adata.obsm`` containing spatial coordinates (X, Y).
    sample_key : str, default "sample"
        Column in ``adata.obs`` identifying unique MSI samples/images.
    n_thresholds : int, default 30
        Precision of the chaos score calculation.

    Returns
    -------
    chaos : np.ndarray
        Matrix of shape ``(n_vars, n_samples)`` with chaos scores.
    samples : List[str]
        List of sample names corresponding to the columns of the matrix.
    """
    if sample_key not in adata.obs:
        raise KeyError(f"Column '{sample_key}' not found in AnnData.obs")
    if obsm_key not in adata.obsm:
        raise KeyError(f"Key '{obsm_key}' not found in AnnData.obsm")

    data_matrix = _get_data_matrix(adata, layer)
    samples = list(pd.unique(adata.obs[sample_key]))

    n_vars = adata.n_vars
    n_samples = len(samples)

    chaos = np.full((n_vars, n_samples), np.nan, dtype=float)

    coords_all = adata.obsm[obsm_key]
    coords_all = coords_all.values if hasattr(coords_all, "values") else coords_all

    for sample_idx, sample in enumerate(samples):
        mask = adata.obs[sample_key] == sample
        if not np.any(mask):
            continue

        coords = coords_all[mask]
        x = coords[:, 0].astype(int)
        y = coords[:, 1].astype(int)

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        sample_block = data_matrix[mask, :]
        if sp.issparse(sample_block):
            sample_block = sample_block.toarray()
        sample_block = np.asarray(sample_block, dtype=float)

        height = y_max - y_min + 1
        width = x_max - x_min + 1

        row_idx = y - y_min
        col_idx = x - x_min

        for var_idx in range(n_vars):
            img = np.full((height, width), np.nan, dtype=float)
            img[row_idx, col_idx] = sample_block[:, var_idx]
            chaos[var_idx, sample_idx] = spatial_chaos_score(
                img, n_thresholds=n_thresholds
            )

    return chaos, samples


def spatial_chaos_fold_change_from_adata(
    adata: ad.AnnData,
    *,
    groupby: str,
    control_label: Any,
    interaction_label: Any,
    varm_key: str = "spatial_chaos",
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    Compute spatial chaos fold change using scores stored in ``adata.varm``.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.
    groupby : str
        The column in ``adata.obs`` that defines the experimental group 
        (e.g., 'condition').
    control_label : Any
        The name of the control group.
    interaction_label : Any
        The name of the condition group.
    varm_key : str, default "spatial_chaos"
        The key in ``adata.varm`` where the chaos score matrix is stored.
    eps : float, default 1e-6
        Numerical floor for denominator.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with FC results and sample group mapping.
    """

    if varm_key not in adata.varm:
        raise KeyError(
            f"Spatial chaos scores not found in adata.varm['{varm_key}']."
        )

    chaos_scores = np.asarray(adata.varm[varm_key], dtype=float)
    if chaos_scores.ndim != 2:
        raise ValueError(
            f"adata.varm['{varm_key}'] must be 2D with shape (n_vars, n_samples)"
        )

    spatial_meta = adata.uns.get("spatial_chaos", {})
    sample_key = spatial_meta.get("sample_key", "sample")
    samples = spatial_meta.get("samples")
    if samples is None:
        if sample_key not in adata.obs:
            raise KeyError(
                "Sample information missing. Provide 'samples' in adata.uns['spatial_chaos'] "
                f"or a '{sample_key}' column in adata.obs."
            )
        samples = list(pd.unique(adata.obs[sample_key]))

    if chaos_scores.shape[1] != len(samples):
        raise ValueError(
            "Number of columns in chaos scores does not match stored samples order."
        )

    if groupby not in adata.obs:
        raise KeyError(f"Column '{groupby}' not found in AnnData.obs")
    if sample_key not in adata.obs:
        raise KeyError(f"Column '{sample_key}' not found in AnnData.obs")

    sample_groups: List[Any] = []
    for sample in samples:
        mask = adata.obs[sample_key] == sample
        if not np.any(mask):
            raise ValueError(f"No observations found for sample '{sample}'")

        labels = pd.unique(adata.obs.loc[mask, groupby].dropna())
        if len(labels) == 0:
            raise ValueError(
                f"No non-NaN labels found in column '{groupby}' for sample '{sample}'"
            )
        if len(labels) > 1:
            raise ValueError(
                f"Multiple labels found for sample '{sample}' in column '{groupby}': {labels}"
            )

        sample_groups.append(labels[0])

    fold_change = spatial_chaos_fold_change(
        chaos_scores=chaos_scores,
        sample_groups=sample_groups,
        control_label=control_label,
        interaction_label=interaction_label,
        eps=eps,
    )

    fold_change["sample_groups"] = np.asarray(sample_groups)
    fold_change["samples"] = np.asarray(samples)

    return fold_change
