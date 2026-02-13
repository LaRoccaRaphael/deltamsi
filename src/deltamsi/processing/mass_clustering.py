"""
Ion Clustering Utilities
========================

This module provides graph-based clustering methods to group ions based on 
chemical or spatial similarity. It leverages the Leiden algorithm to find 
communities in large molecular networks.

Two main clustering strategies are supported:
1. **Network Annotation**: Building edges based on known mass differences 
   (e.g., adducts, isotopic shifts).
2. **Colocalization Clustering**: Building edges based on spatial 
   correlation between ion images.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Optional, Callable, Tuple, Union, Any, Dict


# --- helper: singleton communities -> -1 (counted as a single cluster)
def _singletons_to_minus1(labels: np.ndarray) -> np.ndarray:
    """
    Convert single-member cluster labels to -1 (noise).

    Parameters
    ----------
    labels : np.ndarray
        Array of integer labels (e.g., from clustering).

    Returns
    -------
    np.ndarray
        Cleaned labels where all singletons are replaced by -1.
    """
    from collections import Counter

    lab = np.asarray(labels, dtype=int)
    cnt = Counter(lab.tolist())
    singletons = {c for c, k in cnt.items() if k == 1}
    if not singletons:
        return lab
    return np.array([-1 if c in singletons else c for c in lab], dtype=int)


# --- helper: tolerance in Da for a target delta (supports ('da', v) or ('ppm', v))
def _tol_da_for(target_delta: float, tol: Union[float, Tuple[str, float]]) -> float:
    """
    Calculate tolerance in Dalton (Da) for a specific target mass difference.

    Parameters
    ----------
    target_delta : float
        The theoretical or target mass difference (m/z or Da).
    tol : Union[float, Tuple[str, float]]
        The tolerance specification. If a float, it is treated as Da.
        If a tuple, format must be ('da', value) or ('ppm', value).

    Returns
    -------
    float
        The tolerance converted to absolute Dalton.

    Raises
    ------
    ValueError
        If the tolerance mode in the tuple is not 'da' or 'ppm'.
    """
    if isinstance(tol, tuple):
        mode, val = tol
        mode = str(mode).lower()
        if mode == "da":
            return float(val)
        elif mode == "ppm":
            return float(val) * float(target_delta) * 1e-6
        else:
            raise ValueError(f"Unknown tol mode: {mode!r}")
    else:
        return float(tol)


# --- helper: score -> weight (bigger is better for Leiden)
def _weight_from_score(
    score: float, scheme: Union[str, Callable[[float], float]] = "inv1p", **kw: Any
) -> float:
    """
    Transform a distance or error score into a graph edge weight.

    Leiden and other community detection algorithms usually require weights 
    where larger values indicate stronger similarities.

    Parameters
    ----------
    score : float
        The input score (usually an error or distance where smaller is better).
    scheme : Union[str, Callable], default "inv1p"
        The transformation method:
        - "inv1p": $1 / (1 + score)$
        - "exp": $\exp(-\alpha \cdot score)$, requires `alpha` in `kw`.
        - "one": Returns 1.0 regardless of score.
        - Callable: Custom function applied to the score.
    **kw : Any
        Additional parameters for specific schemes (e.g., `alpha` for "exp").

    Returns
    -------
    float
        The calculated edge weight.
    """
    if callable(scheme):
        return float(scheme(float(score)))
    scheme = str(scheme).lower()
    s = float(score)
    if scheme == "inv1p":  # default: 1/(1+score)
        return 1.0 / (1.0 + s)
    elif scheme == "exp":  # exp(-alpha * score)
        alpha = float(kw.get("alpha", 1.0))
        return float(np.exp(-alpha * s))
    elif scheme == "one":  # ignore scores, all edges weight=1
        return 1.0
    else:
        raise ValueError(f"Unknown weight transform: {scheme}")


def _prune_edges_knn_df(
    n_nodes: int,
    edges_df: pd.DataFrame,
    *,
    k: int,
    mode: str = "union",  # "union" or "mutual"
    weight_col: str = "weight",
    err_col: str = "err",
    score_col: str = "cand_score",
    dm_col: str = "dm",
) -> pd.DataFrame:
    """
    Prune an undirected edge list to keep only the top-k neighbors per node.

    Edges are ranked based on a hierarchical key:
    1. Maximum weight (descending)
    2. Minimum absolute error (ascending)
    3. Minimum candidate score (ascending)
    4. Minimum delta mass (ascending)

    Parameters
    ----------
    n_nodes : int
        Total number of nodes in the graph.
    edges_df : pd.DataFrame
        DataFrame containing edge list with columns 'i' and 'j' (node indices).
    k : int
        The number of neighbors to keep for each node.
    mode : {"union", "mutual"}, default "union"
        - "union": Keep edge if it is in the top-k for node `i` OR node `j`.
        - "mutual": Keep edge only if it is in the top-k for BOTH node `i` AND node `j`.
    weight_col : str, default "weight"
        Column name for weights (primary sorting key).
    err_col : str, default "err"
        Column name for mass error.
    score_col : str, default "cand_score"
        Column name for the candidate score.
    dm_col : str, default "dm"
        Column name for the $\Delta m$ value.

    Returns
    -------
    pd.DataFrame
        A pruned DataFrame containing only the selected edges.

    Notes
    -----
    If $k$ is None or non-positive, or if the DataFrame is empty, the original 
    DataFrame is returned.
    """
    if k is None or k <= 0 or edges_df.empty:
        return edges_df

    i_arr = edges_df["i"].to_numpy(int)
    j_arr = edges_df["j"].to_numpy(int)
    w_arr = edges_df[weight_col].to_numpy(float)
    e_arr = edges_df.get(err_col, pd.Series(np.zeros(len(edges_df)))).to_numpy(float)
    s_arr = edges_df.get(score_col, pd.Series(np.zeros(len(edges_df)))).to_numpy(float)
    d_arr = edges_df.get(dm_col, pd.Series(np.zeros(len(edges_df)))).to_numpy(float)

    # Build neighbor lists for each node: entries are (neighbor, edge_index, sort_key)
    neigh: list[list[tuple[int, int, tuple[float, float, float, float, int]]]] = [
        [] for _ in range(n_nodes)
    ]
    for idx in range(len(edges_df)):
        u, v = int(i_arr[idx]), int(j_arr[idx])
        # sort key: (-weight, +err, +score, +dm)
        key = (-w_arr[idx], e_arr[idx], s_arr[idx], d_arr[idx], idx)
        neigh[u].append((v, idx, key))
        neigh[v].append((u, idx, key))

    # Select top-k per node
    topk_sets: list[set[int]] = [set() for _ in range(n_nodes)]
    for u in range(n_nodes):
        if not neigh[u]:
            continue
        # sort by key then keep first k edge indices
        neigh[u].sort(key=lambda x: x[2])
        chosen = neigh[u][:k]
        topk_sets[u] = {edge_idx for (_, edge_idx, _) in chosen}

    keep = np.zeros(len(edges_df), dtype=bool)
    if str(mode).lower() == "mutual":
        for idx in range(len(edges_df)):
            u, v = int(i_arr[idx]), int(j_arr[idx])
            # keep only if edge is chosen by both endpoints
            if (idx in topk_sets[u]) and (idx in topk_sets[v]):
                keep[idx] = True
    else:  # union (default)
        for idx in range(len(edges_df)):
            u, v = int(i_arr[idx]), int(j_arr[idx])
            if (idx in topk_sets[u]) or (idx in topk_sets[v]):
                keep[idx] = True

    pruned = edges_df.loc[keep].reset_index(drop=True)
    return pruned


def cluster_masses_with_candidates(
    masses: Union[np.ndarray, list[float]],
    candidates_df: pd.DataFrame,
    *,
    delta_col: str = "delta_da",  # candidate :math:`\Delta m` in Da
    score_col: str = "score",  # smaller is better
    label_col: Optional[str] = "label",  # optional (e.g., CHON delta string)
    tol: Union[float, Tuple[str, float]] = ("da", 0.005),  # absolute or ppm tolerance
    edge_max_delta_m: Optional[float] = None,  # prune long-range edges
    keep_mask: Optional[np.ndarray] = None,  # NxN {0,1} matrix to keep/remove edges
    resolution: float = 1.0,  # Leiden resolution_parameter
    weight_transform: Union[str, Callable[[float], float]] = "inv1p",
    weight_kwargs: Optional[Dict[str, Any]] = None,
    return_graph: bool = False,  # if True, return igraph object as well
    knn_k: Optional[int] = None,  # e.g., 15 to enable k-NN pruning
    knn_mode: str = "union",  # "union" or "mutual"
    seed: Optional[int] = 0,
) -> dict[str, object]:
    """
    Cluster masses by matching experimental m/z differences to a chemical catalog.

    This function builds a graph where nodes are experimental m/z values. 
    An edge is created between two nodes if the difference between their 
    masses matches a candidate mass difference (e.g., $Na^+ - H^+$) 
    within a specified tolerance.

    Parameters
    ----------
    masses : array-like
        The experimental m/z values to cluster.
    candidates_df : pd.DataFrame
        A "catalog" of expected mass shifts. Must contain a column for 
        the mass difference (delta) and a quality score.
    delta_col : str, default "delta_da"
        The column in `candidates_df` containing the $\Delta m$ in Daltons.
    score_col : str, default "score"
        The column containing candidate scores (smaller is better).
    tol : float or tuple, default ("da", 0.005)
        Matching tolerance. Can be absolute (Daltons) or relative (ppm).
    resolution : float, default 1.0
        The Leiden resolution parameter. Higher values lead to more, 
        smaller clusters.
    knn_k : int, optional
        If provided, prunes the graph to keep only the top-k strongest 
        edges per node.
    return_graph : bool, default False
        Whether to include the `igraph.Graph` object in the output.
    seed : int, optional
        Random seed passed to Leiden for deterministic partitioning when possible.

    Returns
    -------
    dict
        A dictionary containing:
        - ``"labels"``: Cluster assignment for each mass (singletons are -1).
        - ``"n_clusters"``: Total number of clusters detected.
        - ``"edges"``: A DataFrame of all matched chemical relationships.
        - ``"compression"``: Ratio of clusters to total input masses.

    Notes
    -----
    The algorithm breaks ties between multiple matching candidates by 
    selecting the one with the lowest score, then the lowest mass error.

    

    Examples
    --------
    >>> import pandas as pd
    >>> from deltamsi.processing.mass_clustering import cluster_masses_with_candidates
    >>> # Catalog of common adducts/isotopic shifts
    >>> catalog = pd.DataFrame({"delta_da": [1.0033, 21.9819], "score": [0.1, 0.2]})
    >>> results = cluster_masses_with_candidates(mass_list, catalog, tol=("ppm", 5.0))
    >>> print(f"Found {results['n_clusters']} molecular families.")
    """
    weight_kwargs = weight_kwargs or {}

    masses = np.asarray(masses, dtype=float)
    n = masses.size
    if n < 2:
        raise ValueError("Need at least two masses.")

    # Validate keep_mask early
    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask)
        if keep_mask.shape != (n, n):
            raise ValueError(f"keep_mask must be shape {(n,n)}, got {keep_mask.shape}")
        # use only upper triangle
        keep_mask = np.triu(keep_mask, k=1).astype(bool)

    # Prepare candidates: sort by delta for fast search
    if delta_col not in candidates_df.columns or score_col not in candidates_df.columns:
        raise ValueError(
            f"candidates_df must contain columns '{delta_col}' and '{score_col}'"
        )
    cand = (
        candidates_df[
            [delta_col, score_col]
            + (
                [label_col]
                if (label_col and label_col in candidates_df.columns)
                else []
            )
        ]
        .dropna(subset=[delta_col, score_col])
        .copy()
    )
    cand[delta_col] = cand[delta_col].astype(float)
    cand[score_col] = cand[score_col].astype(float)
    if label_col and label_col in cand.columns:
        cand[label_col] = cand[label_col].astype(str)
    cand = cand.sort_values(delta_col, kind="mergesort").reset_index(drop=True)
    cand_vals = cand[delta_col].to_numpy(float)

    # Build edges
    rows = []
    for i in range(n):
        mi = masses[i]
        for j in range(i + 1, n):
            mj = masses[j]
            dm = abs(mi - mj)

            if edge_max_delta_m is not None and dm > float(edge_max_delta_m):
                continue
            if keep_mask is not None and not keep_mask[i, j]:
                continue

            # Prefilter window (for performance):
            # - if tol is ppm, approximate with ppm at dm; then exact-check per candidate below
            if isinstance(tol, tuple) and str(tol[0]).lower() == "ppm":
                tol_win = float(tol[1]) * dm * 1e-6
            else:
                tol_win = _tol_da_for(dm, tol)

            lo = dm - tol_win
            hi = dm + tol_win
            # binary search on sorted candidate deltas
            import bisect

            a = bisect.bisect_left(cand_vals, lo)
            b = bisect.bisect_right(cand_vals, hi)
            if a >= b:
                continue  # nothing in coarse window

            # precise filtering + pick min score, tie-break on mass error
            best_k = None
            best_score = None
            best_err = None
            best_tol_da = None
            for k in range(a, b):
                cdm = cand_vals[k]
                tol_da_k = _tol_da_for(cdm, tol)  # exact tol for this candidate
                err = abs(dm - cdm)
                if err <= tol_da_k:
                    s = cand.iloc[k][score_col]
                    if (
                        (best_k is None)
                        or (s < best_score)
                        or (np.isclose(s, best_score) and err < best_err)
                    ):
                        best_k = k
                        best_score = float(s)
                        best_err = float(err)
                        best_tol_da = float(tol_da_k)

            if best_k is None or best_score is None:
                continue  # no candidate actually within exact tolerance

            # Compute weight
            w = _weight_from_score(best_score, weight_transform, **weight_kwargs)
            lbl = (
                cand.iloc[best_k][label_col]
                if (label_col and label_col in cand.columns)
                else None
            )

            rows.append(
                {
                    "i": i,
                    "j": j,
                    "mz_i": mi,
                    "mz_j": mj,
                    "dm": dm,
                    "cand_delta": float(cand_vals[best_k]),
                    "cand_score": best_score,
                    "cand_label": lbl,
                    "weight": float(w),
                    "err": best_err,
                    "tol_da_used": best_tol_da,
                }
            )

    edges_df = pd.DataFrame(rows)
    # If no edges, everyone is singleton -> all -1
    if edges_df.empty:
        labels = np.full(n, -1, dtype=int)
        uniq = np.unique(labels)
        return {
            "labels": labels,
            "n_clusters": int(uniq.size),
            "n_minus1": int((labels == -1).sum()),
            "compression": float(uniq.size) / float(n),
            "edges": edges_df,
            **({"graph": None} if return_graph else {}),
        }

    if knn_k is not None and knn_k > 0:
        edges_df = _prune_edges_knn_df(
            n_nodes=n,
            edges_df=edges_df,
            k=int(knn_k),
            mode=str(knn_mode).lower(),
            weight_col="weight",
            err_col="err",
            score_col="cand_score",
            dm_col="dm",
        )
        if edges_df.empty:
            # All edges pruned: all singletons -> -1
            labels = np.full(n, -1, dtype=int)
            uniq = np.unique(labels)
            return {
                "labels": labels,
                "n_clusters": int(uniq.size),
                "n_minus1": int((labels == -1).sum()),
                "compression": float(uniq.size) / float(n),
                "edges": edges_df,
                **({"graph": None} if return_graph else {}),
            }

    # Build igraph & run Leiden
    try:
        import igraph as ig
        import leidenalg as la
    except Exception as e:
        raise ImportError(
            "Leiden requires 'python-igraph' and 'leidenalg', which are installed "
            "with deltamsi. Ensure their system dependencies (e.g., igraph shared "
            "libraries) are available on your platform."
        ) from e

    # Map to igraph index space (0..n-1, same as array indices)
    g = ig.Graph(n=n, edges=list(zip(edges_df["i"].tolist(), edges_df["j"].tolist())))
    g.es["weight"] = edges_df["weight"].astype(float).tolist()
    g.vs["mass"] = masses.tolist()

    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=seed,
    )
    labels = np.array(part.membership, dtype=int)

    # Convert singleton clusters to -1 (count -1 as one real cluster)
    labels = _singletons_to_minus1(labels)
    n_clusters = int(np.unique(labels).size)
    n_minus1 = int((labels == -1).sum())
    compression = float(n_clusters) / float(n)

    result = {
        "labels": labels,
        "n_clusters": n_clusters,
        "n_minus1": n_minus1,
        "compression": compression,
        "edges": edges_df,
    }
    if return_graph:
        result["graph"] = g
    return result


def cluster_masses_from_colocalization(
    coloc_matrix: Union[np.ndarray, sp.spmatrix],
    *,
    keep_mask: Optional[np.ndarray] = None,
    resolution: float = 1.0,
    edge_max_delta_cosine: Optional[float] = None,
    knn_k: Optional[int] = None,
    knn_mode: str = "union",
    return_graph: bool = False,
    seed: Optional[int] = 0,
) -> dict[str, object]:
    """
    Cluster ions based on spatial colocalization similarity.

    Instead of using chemical knowledge, this function groups ions that 
    look the same spatially. It uses a similarity matrix (typically 
    cosine similarity) to build a graph of co-occurring molecular features.

    Parameters
    ----------
    coloc_matrix : np.ndarray or sparse matrix
        A square similarity matrix where `coloc[i, j]` is the correlation 
        between ion image $i$ and $j$.
    edge_max_delta_cosine : float, optional
        Minimum similarity threshold. Edges with similarity below this 
        value are ignored.
    resolution : float, default 1.0
        Leiden resolution parameter for community detection.
    knn_k : int, optional
        Retain only the top-k most similar spatial neighbors for each ion.
    seed : int, optional
        Random seed passed to Leiden for deterministic partitioning when possible.

    Returns
    -------
    dict
        A dictionary with keys: ``"labels"``, ``"n_clusters"``, ``"edges"``, 
        and optionally ``"graph"``.

    Notes
    -----
    This method is purely data-driven and does not require a chemical 
    database. It is highly effective for discovering adducts and 
    fragments that are strictly co-localized in tissue.
    """

    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if knn_k is not None and knn_k < 0:
        raise ValueError("knn_k cannot be negative.")
    if knn_mode not in ["union", "mutual"]:
        raise ValueError("knn_mode must be 'union' or 'mutual'.")

    if coloc_matrix.shape[0] != coloc_matrix.shape[1]:
        raise ValueError("coloc_matrix must be square.")

    n = coloc_matrix.shape[0]
    labels = np.full(n, -1, dtype=int)

    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask)
        if keep_mask.shape != (n, n):
            raise ValueError(f"keep_mask must be shape {(n, n)}, got {keep_mask.shape}")
        keep_mask = np.triu(keep_mask, k=1).astype(bool)

    # Build edges from the upper triangle
    rows = []
    threshold = edge_max_delta_cosine

    if sp.issparse(coloc_matrix):
        coo = coloc_matrix.tocoo()
        for i, j, val in zip(coo.row, coo.col, coo.data):
            if i >= j:
                continue
            if keep_mask is not None and not keep_mask[i, j]:
                continue
            if threshold is not None and float(val) < float(threshold):
                continue
            rows.append({"i": int(i), "j": int(j), "cosine": float(val), "weight": float(val)})
    else:
        tri = np.triu_indices(n, k=1)
        vals = np.asarray(coloc_matrix)[tri]
        mask = np.ones(vals.shape, dtype=bool)
        if threshold is not None:
            mask &= vals >= float(threshold)
        if keep_mask is not None:
            mask &= keep_mask[tri]
        if mask.any():
            i_sel = tri[0][mask]
            j_sel = tri[1][mask]
            v_sel = vals[mask]
            rows = [
                {"i": int(i), "j": int(j), "cosine": float(v), "weight": float(v)}
                for i, j, v in zip(i_sel, j_sel, v_sel)
            ]

    edges_df_local = pd.DataFrame(rows)

    if edges_df_local.empty:
        uniq = np.unique(labels)
        return {
            "labels": labels,
            "n_clusters": int(uniq.size),
            "n_minus1": int((labels == -1).sum()),
            "compression": float(uniq.size) / float(n),
            "edges": pd.DataFrame(columns=["i", "j", "cosine", "weight"]),
            **({"graph": None} if return_graph else {}),
        }

    if knn_k is not None and knn_k > 0:
        edges_df_local = _prune_edges_knn_df(
            n_nodes=n,
            edges_df=edges_df_local,
            k=int(knn_k),
            mode=str(knn_mode).lower(),
            weight_col="weight",
            err_col="err",  # columns missing -> default zeros in helper
            score_col="cand_score",
            dm_col="dm",
        )
        if edges_df_local.empty:
            uniq = np.unique(labels)
            return {
                "labels": labels,
                "n_clusters": int(uniq.size),
                "n_minus1": int((labels == -1).sum()),
                "compression": float(uniq.size) / float(n),
                "edges": pd.DataFrame(columns=["i", "j", "cosine", "weight"]),
                **({"graph": None} if return_graph else {}),
            }

    try:
        import igraph as ig
        import leidenalg as la
    except Exception as e:
        raise ImportError(
            "Leiden requires 'python-igraph' and 'leidenalg', which are installed "
            "with deltamsi. Ensure their system dependencies (e.g., igraph shared "
            "libraries) are available on your platform."
        ) from e

    g = ig.Graph(
        n=n, edges=list(zip(edges_df_local["i"].tolist(), edges_df_local["j"].tolist()))
    )
    g.es["weight"] = edges_df_local["weight"].astype(float).tolist()

    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=seed,
    )
    labels = np.array(part.membership, dtype=int)
    labels = _singletons_to_minus1(labels)

    n_clusters = int(np.unique(labels).size)
    n_minus1 = int((labels == -1).sum())
    compression = float(n_clusters) / float(n)

    edges_df = edges_df_local[["i", "j", "cosine", "weight"]].copy()

    result = {
        "labels": labels,
        "n_clusters": n_clusters,
        "n_minus1": n_minus1,
        "compression": compression,
        "edges": edges_df,
    }
    if return_graph:
        result["graph"] = g
    return result
