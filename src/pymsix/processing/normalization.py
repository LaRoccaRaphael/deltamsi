"""
Normalization Utilities for MSI
==============================

This module provides standard normalization workflows for Mass Spectrometry 
Imaging (MSI) datasets. Normalization is essential to ensure that pixel 
intensities are comparable across different regions of a tissue section or 
between different samples.

The primary method implemented here is TIC normalization, which scales 
each spectrum so that the sum of all intensities equals a constant value.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:  # pragma: no cover
    from pymsix.core.msicube import MSICube


def tic_normalize_msicube(
    msicube: "MSICube",
    *,
    target_sum: float = 1e6,
    layer: Optional[str] = None,
    store_tic_in_obs: Optional[str] = "tic",
    copy: bool = False,
) -> Optional["MSICube"]:
    """
    Apply Total Ion Current (TIC) normalization to an MSICube intensity matrix.

    TIC normalization rescales each spectrum (row) in the data matrix so that 
    the sum of its intensities equals the `target_sum`. This method is widely 
    used to compensate for pixel-to-pixel variations in total ion yield.

    Parameters
    ----------
    msicube : MSICube
        The MSICube instance containing the intensity matrix (AnnData).
    target_sum : float, default 1e6
        The desired sum for each spectrum after normalization. 
        Setting this to 1.0 results in fractional abundance values.
    layer : str, optional
        The specific AnnData layer to normalize. If None, normalizes ``adata.X``.
    store_tic_in_obs : str, optional, default "tic"
        Key used to save the original TIC values (before normalization) in 
        ``adata.obs``. This is useful for quality control.
    copy : bool, default False
        If True, returns a new MSICube instance. If False, the operation 
        is performed in-place on the original object.

    Returns
    -------
    MSICube or None
        If ``copy=True``, returns the normalized MSICube. 
        If ``copy=False``, returns None and modifies the input in-place.

    Notes
    -----
    **Mathematical Formulation**
    For a spectrum $x = [x_1, x_2, ..., x_n]$, the TIC-normalized spectrum $x'$ 
    is calculated as:
    
    $$x'_i = \frac{x_i}{\sum_{j=1}^{n} x_j} \times \text{target\_sum}$$

    

    **Implementation Details**
    The function efficiently handles both dense NumPy arrays and Scipy 
    sparse matrices. For sparse data, it uses diagonal matrix multiplication 
    to maintain performance and memory efficiency.

    **Warning**
    This method assumes non-negative intensities. If your data has been 
    centered or baseline-corrected in a way that introduced negative values, 
    apply a floor (clipping) or shift before calling this function.

    Examples
    --------
    >>> from pymsix.processing.normalization import tic_normalize_msicube
    >>> # Normalize in-place to a target sum of 100
    >>> tic_normalize_msicube(msicube, target_sum=100.0)
    >>> # The original TIC values are now stored in msicube.adata.obs['tic']
    >>> print(msicube.adata.X[0].sum())
    100.0
    """

    if msicube.adata is None:
        raise ValueError("MSICube.adata is None. Run data extraction first.")

    obj = deepcopy(msicube) if copy else msicube
    adata = obj.adata

    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        X = adata.layers[layer]

    if sp.issparse(X):
        tic = np.asarray(X.sum(axis=1)).ravel()
        if store_tic_in_obs is not None:
            adata.obs[store_tic_in_obs] = tic

        scale = np.zeros_like(tic, dtype=np.float64)
        nz = tic > 0
        scale[nz] = target_sum / tic[nz]

        X_csr = X.tocsr(copy=True)
        X_norm = sp.diags(scale).dot(X_csr)
    else:
        X_arr = np.asarray(X)
        tic = X_arr.sum(axis=1)
        if store_tic_in_obs is not None:
            adata.obs[store_tic_in_obs] = tic

        scale = np.zeros_like(tic, dtype=np.float64)
        nz = tic > 0
        scale[nz] = target_sum / tic[nz]

        X_norm = X_arr * scale[:, None]

    if layer is None:
        adata.X = X_norm
    else:
        adata.layers[layer] = X_norm

    adata.uns.setdefault("tic_normalize", {})
    adata.uns["tic_normalize"].update({"target_sum": float(target_sum), "layer": layer})

    return obj if copy else None

