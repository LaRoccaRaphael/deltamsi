from __future__ import annotations

from typing import Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _resolve_var_pos(adata: ad.AnnData, mz_id: Union[int, str]) -> int:
    if isinstance(mz_id, (int, np.integer)):
        pos = int(mz_id)
        if pos < 0 or pos >= adata.n_vars:
            raise IndexError(f"mz_id={pos} out of range [0, {adata.n_vars}).")
        return pos
    mz_id = str(mz_id)
    try:
        return int(adata.var_names.get_loc(mz_id))
    except KeyError as e:
        raise KeyError(f"mz_id {mz_id!r} not found in adata.var_names.") from e


def _get_edges_df(adata: ad.AnnData, mass_uns_key: str, edges_key: str) -> pd.DataFrame:
    if mass_uns_key not in adata.uns:
        raise KeyError(f"adata.uns[{mass_uns_key!r}] not found.")
    obj = adata.uns[mass_uns_key]
    if not isinstance(obj, dict):
        raise TypeError(f"adata.uns[{mass_uns_key!r}] must be a dict-like object.")
    if edges_key not in obj:
        raise KeyError(f"adata.uns[{mass_uns_key!r}][{edges_key!r}] not found.")
    edges = obj[edges_key]
    if not isinstance(edges, pd.DataFrame):
        raise TypeError(
            f"adata.uns[{mass_uns_key!r}][{edges_key!r}] must be a pandas DataFrame."
        )
    return edges


def _get_cosine_matrix(adata: ad.AnnData, cosine_key: str):
    if cosine_key not in adata.varp:
        raise KeyError(f"adata.varp[{cosine_key!r}] not found.")
    S = adata.varp[cosine_key]
    if not (isinstance(S, np.ndarray) or sp.issparse(S)):
        raise TypeError(
            f"adata.varp[{cosine_key!r}] must be a numpy array or scipy sparse matrix."
        )
    if S.shape != (adata.n_vars, adata.n_vars):
        raise ValueError(f"Cosine matrix shape {S.shape} != {(adata.n_vars, adata.n_vars)}.")
    return S


def _cosine_row(S, src: int, nbrs: np.ndarray) -> np.ndarray:
    if nbrs.size == 0:
        return np.zeros((0,), dtype=float)
    if sp.issparse(S):
        r = S[src, nbrs]
        return (r.toarray().ravel() if sp.issparse(r) else np.asarray(r).ravel()).astype(
            float
        )
    return np.asarray(S[src, nbrs], dtype=float).ravel()


def direct_mass_neighbors(
    adata: ad.AnnData,
    mz_id: Union[int, str],
    *,
    mass_uns_key: str = "mass_clustering",
    edges_key: str = "edges",
    cosine_key: str = "ion_cosine",
    var_cols: Optional[Sequence[str]] = None,
    edge_cols: Optional[Sequence[str]] = None,
    include_src_cols: bool = False,
) -> pd.DataFrame:
    """
    Return direct neighbors (distance=1) of mz_id in the mass-difference network.

    Output: one row per edge incident to src, containing neighbor + edge info
    + cosine(src, nbr).
    """
    src = _resolve_var_pos(adata, mz_id)
    edges_df = _get_edges_df(adata, mass_uns_key, edges_key)

    if not {"i", "j"}.issubset(edges_df.columns):
        raise ValueError("edges_df must contain columns 'i' and 'j'.")

    ii = np.asarray(edges_df["i"], dtype=int)
    jj = np.asarray(edges_df["j"], dtype=int)

    m = (ii == src) | (jj == src)
    if not np.any(m):
        return pd.DataFrame()

    ed = edges_df.loc[m].copy()

    nbr = np.where(ii[m] == src, jj[m], ii[m]).astype(int)
    ed["src_pos"] = src
    ed["dst_pos"] = nbr
    ed["src_id"] = str(adata.var_names[src])
    ed["dst_id"] = adata.var_names[nbr].astype(str).to_numpy()

    S = _get_cosine_matrix(adata, cosine_key)
    ed[f"cosine_{cosine_key}"] = _cosine_row(S, src, nbr)

    if edge_cols is None:
        keep = [
            "i",
            "j",
            "mz_i",
            "mz_j",
            "dm",
            "cand_delta",
            "cand_score",
            "cand_label",
            "weight",
            "err",
            "tol_da_used",
        ]
        edge_cols_use = [c for c in keep if c in ed.columns]
    else:
        edge_cols_use = [c for c in edge_cols if c in ed.columns]

    base_cols = [
        "src_pos",
        "src_id",
        "dst_pos",
        "dst_id",
        f"cosine_{cosine_key}",
    ]
    out = ed[base_cols + edge_cols_use].reset_index(drop=True)

    if var_cols is None:
        var_cols_use = list(adata.var.columns)
    else:
        var_cols_use = [c for c in var_cols if c in adata.var.columns]

    if var_cols_use:
        nbr_meta = adata.var.iloc[nbr][var_cols_use].reset_index(drop=True)
        nbr_meta.columns = [f"dst_{c}" for c in nbr_meta.columns]
        out = pd.concat([out, nbr_meta], axis=1)

        if include_src_cols:
            src_meta = adata.var.iloc[[src]][var_cols_use].reset_index(drop=True)
            src_meta = pd.concat([src_meta] * out.shape[0], ignore_index=True)
            src_meta.columns = [f"src_{c}" for c in src_meta.columns]
            out = pd.concat([out, src_meta], axis=1)

    out = out.sort_values(f"cosine_{cosine_key}", ascending=False).reset_index(
        drop=True
    )
    return out
