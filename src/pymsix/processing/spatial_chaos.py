import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from typing import Any, List, Optional, Tuple
from scipy.ndimage import label


def spatial_chaos_score(
    image: np.ndarray,
    n_thresholds: int = 30,
) -> float:
    """
    Spatial chaos score S as defined in Section 2.3.1 of Chapter 2:
      1) Normalize ion image to [0, 1] by its max intensity.
      2) Define N thresholds t_k evenly spaced in (0, 1).
      3) For each threshold:
           B_k(x,y) = 1 if I_norm(x,y) >= t_k else 0
         - C_k = number of clusters (4-connectivity) in B_k
         - P_k = number of positive pixels in B_k
      4) C_tot = sum_k C_k, P_tot = sum_k P_k
      5) S = 1 - C_tot / P_tot   (0 <= S <= 1)

    Parameters
    ----------
    image : 2D ndarray
        Ion image with non-negative intensities.
    n_thresholds : int, default 30
        Number of thresholds N.

    Returns
    -------
    S : float
        Spatial chaos score in [0, 1]. Higher S = more spatial structure.
        Returns np.nan if image has no positive intensities.
    """
    im = np.asarray(image, dtype=float)

    if im.ndim != 2:
        raise ValueError("spatial_chaos_score expects a 2D array")

    # Ignore NaNs when computing max; treat NaNs as 0 elsewhere
    max_val = np.nanmax(im)
    if not np.isfinite(max_val) or max_val <= 0:
        # No signal → undefined / non-informative score
        return np.nan

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
        # Should not happen if max_val > 0, but be safe
        return np.nan

    # 5) Spatial chaos score
    S = 1.0 - (C_tot / P_tot)

    # Numerical safety clamp
    S = float(max(0.0, min(1.0, S)))
    return S


def _get_data_matrix(adata: ad.AnnData, layer: Optional[str]) -> Any:
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
    Compute spatial chaos scores for every variable (ion) and every MSI sample.

    The output matrix has shape ``(n_vars, n_samples)`` where columns follow the
    order of unique ``adata.obs[sample_key]`` values.
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
