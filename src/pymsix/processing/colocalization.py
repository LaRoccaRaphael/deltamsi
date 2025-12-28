"""Cosine colocalization utilities for MSI ion images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:  # pragma: no cover
    from pymsix.core.msicube import MSICube


@dataclass
class CosineColocParams:
    """Parameters controlling cosine-based ion colocalization computation."""

    layer: Optional[str] = None
    dtype: Union[np.dtype, str] = "float32"
    mode: Literal["dense", "topk_sparse"] = "topk_sparse"
    topk: int = 50
    min_sim: float = 0.2
    chunk_size: int = 256
    symmetrize: bool = True
    include_self: bool = False
    store_varp_key: Optional[str] = "ion_cosine"
    store_keep_mask_key: Optional[str] = "keep_coloc"
    keep_rule: Literal["any_edge", "max_sim"] = "any_edge"


def _get_X(msicube: "MSICube", layer: Optional[str]) -> Union[np.ndarray, sp.spmatrix]:
    adata = msicube.adata
    if adata is None:
        raise ValueError("MSICube.adata is None. Run data extraction first.")

    if layer is None:
        return adata.X

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers")

    return adata.layers[layer]


def _col_l2_norms(X: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
    if sp.issparse(X):
        return np.sqrt(X.power(2).sum(axis=0)).A1
    return np.linalg.norm(X, axis=0)


def _normalize_columns(
    X: Union[np.ndarray, sp.spmatrix], norms: np.ndarray, dtype: np.dtype
) -> Union[np.ndarray, sp.spmatrix]:
    norms = norms.astype(dtype, copy=False)
    inv = np.zeros_like(norms)
    nz = norms > 0
    inv[nz] = 1.0 / norms[nz]

    if sp.issparse(X):
        Xc = X.tocsc(copy=False).astype(dtype, copy=False)
        return Xc.multiply(inv)

    Xd = np.asarray(X, dtype=dtype, order="F")
    return Xd * inv


def _cosine_dense(Xn: Union[np.ndarray, sp.spmatrix], include_self: bool) -> np.ndarray:
    if sp.issparse(Xn):
        S = (Xn.T @ Xn).toarray()
    else:
        S = Xn.T @ Xn

    if not include_self:
        np.fill_diagonal(S, 0.0)

    return S


def _cosine_topk_sparse(
    Xn_csc: sp.csc_matrix,
    *,
    topk: int,
    min_sim: float,
    chunk_size: int,
    symmetrize: bool,
    include_self: bool,
) -> sp.csr_matrix:
    n_obs, n_vars = Xn_csc.shape
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    data_all: list[np.ndarray] = []

    for start in range(0, n_vars, chunk_size):
        end = min(n_vars, start + chunk_size)
        block = (Xn_csc[:, start:end].T @ Xn_csc).tocsr()

        for bi in range(end - start):
            i = start + bi
            row = block.getrow(bi)
            if row.nnz == 0:
                continue

            idx = row.indices
            val = row.data

            if not include_self:
                m = idx != i
                idx = idx[m]
                val = val[m]

            if min_sim is not None:
                m = val >= float(min_sim)
                idx = idx[m]
                val = val[m]

            if idx.size == 0:
                continue

            if topk is not None and idx.size > int(topk):
                kth = idx.size - int(topk)
                sel = np.argpartition(val, kth)[kth:]
                sel = sel[np.argsort(val[sel])[::-1]]
                idx = idx[sel]
                val = val[sel]
            else:
                order = np.argsort(val)[::-1]
                idx = idx[order]
                val = val[order]

            rows_all.append(np.full(idx.shape, i, dtype=np.int32))
            cols_all.append(idx.astype(np.int32, copy=False))
            data_all.append(val.astype(np.float32, copy=False))

    if not rows_all:
        return sp.csr_matrix((n_vars, n_vars), dtype=np.float32)

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    data = np.concatenate(data_all)

    S = sp.coo_matrix((data, (rows, cols)), shape=(n_vars, n_vars), dtype=np.float32).tocsr()

    if symmetrize:
        S = S.maximum(S.T)

    if include_self:
        S.setdiag(1.0)
    else:
        S.setdiag(0.0)

    S.eliminate_zeros()
    return S


def compute_mz_cosine_colocalization(
    msicube: "MSICube",
    *,
    params: CosineColocParams = CosineColocParams(),
) -> Tuple[Union[np.ndarray, sp.csr_matrix], Optional[np.ndarray]]:
    """Compute cosine similarity between ion images stored on an :class:`MSICube`.

    Parameters
    ----------
    msicube : MSICube
        Cube containing an AnnData object with ion images in ``adata.X`` or a specified
        ``adata.layers`` entry.
    params : CosineColocParams, optional
        Controls storage location, sparsity mode, and masking behavior.

    Returns
    -------
    S : ndarray | scipy.sparse.csr_matrix
        Cosine similarity matrix between variables (ions).
    keep_mask : ndarray | None
        Boolean mask of variables to keep when ``store_keep_mask_key`` is set.
    """

    X = _get_X(msicube, params.layer)
    dtype = np.dtype(params.dtype)

    norms = _col_l2_norms(X)
    Xn = _normalize_columns(X, norms, dtype=dtype)

    if params.mode == "dense":
        S = _cosine_dense(Xn, include_self=params.include_self)
    else:
        if not sp.issparse(Xn):
            Xn = sp.csc_matrix(Xn)
        else:
            Xn = Xn.tocsc(copy=False)

        S = _cosine_topk_sparse(
            Xn,
            topk=params.topk,
            min_sim=params.min_sim,
            chunk_size=params.chunk_size,
            symmetrize=params.symmetrize,
            include_self=params.include_self,
        )

    keep_mask = None
    adata = msicube.adata
    if params.store_keep_mask_key is not None and adata is not None:
        if sp.issparse(S):
            if params.keep_rule == "any_edge":
                keep_mask = (np.asarray((S > 0).sum(axis=1)).ravel() > 0)
            else:
                keep_mask = (S.max(axis=1).toarray().ravel() >= float(params.min_sim))
        else:
            if params.keep_rule == "any_edge":
                keep_mask = (np.sum(S > 0, axis=1) > 0)
            else:
                keep_mask = (np.max(S, axis=1) >= float(params.min_sim))

        adata.var[params.store_keep_mask_key] = keep_mask

    if params.store_varp_key is not None and adata is not None:
        if sp.issparse(S):
            adata.varp[params.store_varp_key] = S
        else:
            adata.varp[params.store_varp_key] = np.asarray(S, dtype=np.float32)

        adata.uns[f"{params.store_varp_key}_params"] = {
            "layer": params.layer,
            "mode": params.mode,
            "topk": params.topk,
            "min_sim": params.min_sim,
            "chunk_size": params.chunk_size,
            "symmetrize": params.symmetrize,
            "include_self": params.include_self,
            "keep_mask_key": params.store_keep_mask_key,
            "keep_rule": params.keep_rule,
            "dtype": str(dtype),
        }

    return S, keep_mask

