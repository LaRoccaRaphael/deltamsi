"""Normalization utilities for MSI cubes."""

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
    Apply Total Ion Current (TIC) normalization to an ``MSICube`` intensity matrix.

    Parameters
    ----------
    msicube : MSICube
        The cube containing the intensity matrix to normalize. ``msicube.adata``
        must be populated.
    target_sum : float, default 1e6
        After normalization, each spectrum (row) sums to ``target_sum``. Use ``1.0``
        to obtain fractional TIC values.
    layer : str | None, default None
        If ``None``, normalize ``adata.X``. Otherwise, normalize ``adata.layers[layer]``.
    store_tic_in_obs : str | None, default "tic"
        If a string, store the pre-normalization TIC in ``adata.obs[store_tic_in_obs]``.
        If ``None``, do not store TIC values.
    copy : bool, default False
        If ``True``, operate on and return a deep copy of the ``MSICube``. Otherwise,
        modify in place and return ``None``.

    Returns
    -------
    MSICube | None
        A normalized copy if ``copy=True``, otherwise ``None``.

    Notes
    -----
    Assumes non-negative intensities. If your data contains negatives
    (baseline-corrected/centered), consider shifting or clipping before applying
    TIC normalization.
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

