"""
Ion Image Colocalization Utilities
==================================

This module provides efficient methods to compute spatial colocalization between 
ion images using cosine similarity. It is designed to handle large-scale MSI 
datasets by providing both dense and memory-efficient sparse (Top-K) 
computation modes.

Colocalization analysis helps in identifying highly correlated m/z images, 
which is a crucial step for adduct identification and molecular family grouping.


"""

from __future__ import annotations

from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from deltamsi.params.options import CosineColocParams

if TYPE_CHECKING:  # pragma: no cover
    from deltamsi.core.msicube import MSICube


def _get_X(msicube: "MSICube", layer: Optional[str]) -> Union[np.ndarray, sp.spmatrix]:
    """
    Extract the data matrix from the associated AnnData object.

    Parameters
    ----------
    msicube : MSICube
        The MSICube instance containing the data.
    layer : str, optional
        Specific layer name to extract. If None, uses ``adata.X``.

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        The data matrix (samples x variables).

    Raises
    ------
    ValueError
        If the AnnData object is not initialized.
    KeyError
        If the specified layer is missing.
    """
    adata = msicube.adata
    if adata is None:
        raise ValueError("MSICube.adata is None. Run data extraction first.")

    if layer is None:
        return adata.X

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers")

    return adata.layers[layer]


def _col_l2_norms(X: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
    """
    Compute the L2 norm for each column of the matrix.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix of shape (n_samples, n_variables).

    Returns
    -------
    np.ndarray
        1D array of shape (n_variables,) containing L2 norms.
    """
    if sp.issparse(X):
        return np.sqrt(X.power(2).sum(axis=0)).A1
    return np.linalg.norm(X, axis=0)


def _normalize_columns(
    X: Union[np.ndarray, sp.spmatrix], norms: np.ndarray, dtype: np.dtype
) -> Union[np.ndarray, sp.spmatrix]:
    """
    Normalize columns of a matrix to unit L2 length.

    Parameters
    ----------
    X : Union[np.ndarray, sp.spmatrix]
        Input matrix to normalize.
    norms : np.ndarray
        Pre-computed L2 norms for each column.
    dtype : np.dtype
        Desired output data type.

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        Column-normalized matrix.
    """
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
    """
    Compute the full cosine similarity matrix.

    Parameters
    ----------
    Xn : Union[np.ndarray, sp.spmatrix]
        Column-normalized matrix (unit length variables).
    include_self : bool
        Whether to keep the diagonal (self-similarity) as 1.0 or set to 0.0.

    Returns
    -------
    np.ndarray
        Dense similarity matrix of shape (n_variables, n_variables).
    """
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
    """
    Compute a sparsified cosine similarity matrix using top-K neighbors.

    Processes the variables in chunks to manage memory and applies filtering
    based on similarity thresholds and neighbor counts.

    Parameters
    ----------
    Xn_csc : sp.csc_matrix
        Column-normalized sparse matrix in CSC format.
    topk : int
        Number of highest similarity neighbors to keep per variable.
    min_sim : float
        Minimum similarity threshold to consider a match.
    chunk_size : int
        Number of variables to process per iteration.
    symmetrize : bool
        If True, ensures the output matrix is symmetric using ``max(S, S.T)``.
    include_self : bool
        Whether to include the diagonal elements.

    Returns
    -------
    sp.csr_matrix
        Sparse similarity matrix (n_variables, n_variables).

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> X = sp.random(100, 50, density=0.1, format='csc')
    >>> # ... normalization ...
    >>> S = _cosine_topk_sparse(X, topk=5, min_sim=0.1, chunk_size=10, 
    ...                         symmetrize=True, include_self=False)
    """
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
) -> Union[np.ndarray, sp.csr_matrix]:
    """
    Compute cosine similarity between ion images stored on an MSICube.

    This function calculates the spatial correlation between every pair of 
    ion images. It first normalizes each ion image by its L2 norm (spatial 
    intensity vector) and then computes the dot product.

    Parameters
    ----------
    msicube : MSICube
        The MSICube instance containing an AnnData object with ion images.
    params : CosineColocParams, optional
        A configuration object controlling computation behavior, sparsity, 
        and storage. If not provided, default parameters are used.

    Returns
    -------
    S : np.ndarray or scipy.sparse.csr_matrix
        A square matrix of shape ``(n_vars, n_vars)`` containing the cosine 
        similarities. Returns a dense NumPy array if `mode="dense"`, 
        otherwise a Scipy CSR sparse matrix.

    Notes
    -----
    The cosine similarity $S$ between two ion images (vectors) $u$ and $v$ is 
    calculated as:
    
    $$S(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

    In ``"topk_sparse"`` mode, the function uses a block-wise approach to 
    remain memory-efficient even with tens of thousands of ions.

    

    The result is automatically stored in ``msicube.adata.varp[params.store_varp_key]`` 
    and the parameters are logged in ``msicube.adata.uns``.

    Examples
    --------
    >>> from deltamsi.params import CosineColocParams
    >>> from deltamsi.processing.colocalization import compute_mz_cosine_colocalization
    >>> # Standard sparse computation keeping only top 10 neighbors
    >>> params = CosineColocParams(topk=10, min_sim=0.3)
    >>> sim_matrix = compute_mz_cosine_colocalization(msicube, params=params)
    
    >>> # To check the similarity between the first two ions:
    >>> if not sp.issparse(sim_matrix):
    ...     print(f"Similarity: {sim_matrix[0, 1]:.3f}")
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

    adata = msicube.adata
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
            "dtype": str(dtype),
        }

    return S
