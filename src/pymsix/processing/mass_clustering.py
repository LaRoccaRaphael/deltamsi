import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Optional, Callable, Tuple, Union, Any, Dict


# --- helper: singleton communities -> -1 (counted as a single cluster)
def _singletons_to_minus1(labels: np.ndarray) -> np.ndarray:
    from collections import Counter

    lab = np.asarray(labels, dtype=int)
    cnt = Counter(lab.tolist())
    singletons = {c for c, k in cnt.items() if k == 1}
    if not singletons:
        return lab
    return np.array([-1 if c in singletons else c for c in lab], dtype=int)


# --- helper: tolerance in Da for a target delta (supports ('da', v) or ('ppm', v))
def _tol_da_for(target_delta: float, tol: Union[float, Tuple[str, float]]) -> float:
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
    k-NN pruning for an undirected edge list in a DataFrame.

    Keeps only edges that are among the top-k neighbors of a node, ranked by:
      1) larger weight first,
      2) smaller mass error (err),
      3) smaller candidate score,
      4) smaller |Δm| (dm).

    mode="union": keep edge if it's in top-k of either endpoint.
    mode="mutual": keep edge only if it's in top-k of both endpoints.
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
    delta_col: str = "delta_da",  # candidate Δm in Da
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
) -> dict[str, object]:
    """
    Build a graph on experimental masses using an external catalog of mass differences and cluster with Leiden.

    Parameters
    ----------
    masses : array-like of shape (n,)
        Experimental m/z (treated as masses).
    candidates_df : pd.DataFrame
        Must contain at least [delta_col, score_col], and optionally label_col.
        - Each row is a candidate Δm (in Da) with a 'score' where smaller is better.
        - If multiple candidates match a pair within tolerance, pick the one with lowest score,
          and break remaining ties by smallest mass error |Δm_exp - Δm_candidate|.
    delta_col : str
        Column name for candidate Δm in Da.
    score_col : str
        Column name for the candidate score (positive, smaller is better).
    label_col : str or None
        Optional column name for a candidate label (e.g., atomic composition).
    tol : float or ('da', value) or ('ppm', value)
        Matching tolerance. For ppm, tolerance is applied around each candidate Δm (exactly).
    edge_max_delta_m : float or None
        If set, edges are only considered when |m_i - m_j| <= edge_max_delta_m.
    keep_mask : np.ndarray or None
        Optional NxN binary matrix; if provided, edges are only added where keep_mask[i,j] == 1.
        The function only reads the upper triangle (i<j). Diagonal is ignored.
    resolution : float
        Leiden resolution_parameter.
    weight_transform : {"inv1p","exp","one"} or callable(score)->weight
        How to convert candidate score (smaller is better) into a graph weight (bigger is better).
    weight_kwargs : dict
        Extra kwargs for the weight transform (e.g., {"alpha": 0.5} for 'exp').
    return_graph : bool
        If True, also return the igraph Graph object.

    Returns
    -------
    result : dict
        {
          "labels": np.ndarray of shape (n,), with singleton clusters set to -1,
          "n_clusters": int (includes -1 as one cluster),
          "n_minus1": int (count of nodes labeled -1),
          "compression": float (n_clusters / n_samples),
          "edges": pd.DataFrame with columns:
              ["i","j","mz_i","mz_j","dm","cand_delta","cand_score","cand_label","weight","err","tol_da_used"],
          "graph": igraph.Graph (only if return_graph=True)
        }

    Notes
    -----
    - Uses python-igraph + leidenalg (installed with pymsix). Ensure system
      requirements for igraph are available on your platform.
    - Tolerance for ppm is computed per candidate (exact), with a prefilter window around dm using ppm at dm for speed.
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
            "with pymsix. Ensure their system dependencies (e.g., igraph shared "
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
) -> dict[str, object]:
    """Cluster m/z values using a cosine colocalization matrix and Leiden.

    This function is analogous to :func:`cluster_masses_with_candidates` but
    operates directly on a cosine colocalization matrix (e.g., the output of
    :func:`pymsix.processing.colocalization.compute_mz_cosine_colocalization`).

    Parameters
    ----------
    coloc_matrix : np.ndarray | scipy.sparse.spmatrix
        Square matrix of cosine similarities between ions. Only the upper
        triangle is used.
    keep_mask : np.ndarray | None
        Optional boolean mask of ions to include in the clustering. When
        provided, labels for excluded ions are set to ``-1``.
    resolution : float
        Leiden resolution_parameter.
    edge_max_delta_cosine : float | None
        Minimum cosine similarity required to keep an edge. Edges with cosine
        smaller than this value are discarded.
    knn_k : int | None
        If provided, retain only the top-k neighbors per node using
        :func:`_prune_edges_knn_df`.
    knn_mode : {"union", "mutual"}
        k-NN pruning mode; see :func:`_prune_edges_knn_df`.
    return_graph : bool
        If True, return the igraph Graph object built on the filtered edges.

    Returns
    -------
    result : dict
        {
          "labels": np.ndarray of shape (n,), with singleton clusters set to -1
                       and excluded nodes (from ``keep_mask``) also set to -1,
          "n_clusters": int (includes -1 as one cluster),
          "n_minus1": int (count of nodes labeled -1),
          "compression": float (n_clusters / n_samples),
          "edges": pd.DataFrame with columns ["i", "j", "cosine", "weight"],
          "graph": igraph.Graph (only if return_graph=True)
        }
    """

    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if knn_k is not None and knn_k < 0:
        raise ValueError("knn_k cannot be negative.")
    if knn_mode not in ["union", "mutual"]:
        raise ValueError("knn_mode must be 'union' or 'mutual'.")

    if coloc_matrix.shape[0] != coloc_matrix.shape[1]:
        raise ValueError("coloc_matrix must be square.")

    n_total = coloc_matrix.shape[0]
    labels_full = np.full(n_total, -1, dtype=int)

    if keep_mask is None:
        keep_mask = np.ones(n_total, dtype=bool)
    else:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.shape[0] != n_total:
            raise ValueError(
                f"keep_mask must have length {n_total}, got {keep_mask.shape[0]}"
            )

    active_idx = np.flatnonzero(keep_mask)
    n_active = active_idx.size

    if n_active == 0:
        uniq = np.unique(labels_full)
        return {
            "labels": labels_full,
            "n_clusters": int(uniq.size),
            "n_minus1": int((labels_full == -1).sum()),
            "compression": float(uniq.size) / float(n_total),
            "edges": pd.DataFrame(columns=["i", "j", "cosine", "weight"]),
            **({"graph": None} if return_graph else {}),
        }

    if sp.issparse(coloc_matrix):
        S_active = coloc_matrix.tocsr()[active_idx][:, active_idx]
    else:
        S_active = np.asarray(coloc_matrix)[np.ix_(active_idx, active_idx)]

    # Build edges from the upper triangle
    rows = []
    threshold = edge_max_delta_cosine

    if sp.issparse(S_active):
        coo = S_active.tocoo()
        for i, j, val in zip(coo.row, coo.col, coo.data):
            if i >= j:
                continue
            if threshold is not None and float(val) < float(threshold):
                continue
            rows.append({"i": int(i), "j": int(j), "cosine": float(val), "weight": float(val)})
    else:
        tri = np.triu_indices(n_active, k=1)
        vals = S_active[tri]
        mask = np.ones(vals.shape, dtype=bool)
        if threshold is not None:
            mask &= vals >= float(threshold)
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
        labels_full[active_idx] = -1
        uniq = np.unique(labels_full)
        return {
            "labels": labels_full,
            "n_clusters": int(uniq.size),
            "n_minus1": int((labels_full == -1).sum()),
            "compression": float(uniq.size) / float(n_total),
            "edges": pd.DataFrame(columns=["i", "j", "cosine", "weight"]),
            **({"graph": None} if return_graph else {}),
        }

    if knn_k is not None and knn_k > 0:
        edges_df_local = _prune_edges_knn_df(
            n_nodes=n_active,
            edges_df=edges_df_local,
            k=int(knn_k),
            mode=str(knn_mode).lower(),
            weight_col="weight",
            err_col="err",  # columns missing -> default zeros in helper
            score_col="cand_score",
            dm_col="dm",
        )
        if edges_df_local.empty:
            labels_full[active_idx] = -1
            uniq = np.unique(labels_full)
            return {
                "labels": labels_full,
                "n_clusters": int(uniq.size),
                "n_minus1": int((labels_full == -1).sum()),
                "compression": float(uniq.size) / float(n_total),
                "edges": pd.DataFrame(columns=["i", "j", "cosine", "weight"]),
                **({"graph": None} if return_graph else {}),
            }

    try:
        import igraph as ig
        import leidenalg as la
    except Exception as e:
        raise ImportError(
            "Leiden requires 'python-igraph' and 'leidenalg', which are installed "
            "with pymsix. Ensure their system dependencies (e.g., igraph shared "
            "libraries) are available on your platform."
        ) from e

    g = ig.Graph(n=n_active, edges=list(zip(edges_df_local["i"].tolist(), edges_df_local["j"].tolist())))
    g.es["weight"] = edges_df_local["weight"].astype(float).tolist()

    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
    )
    labels_active = np.array(part.membership, dtype=int)
    labels_active = _singletons_to_minus1(labels_active)

    labels_full[active_idx] = labels_active
    n_clusters = int(np.unique(labels_full).size)
    n_minus1 = int((labels_full == -1).sum())
    compression = float(n_clusters) / float(n_total)

    edges_df = edges_df_local.copy()
    edges_df["i"] = active_idx[edges_df_local["i"].to_numpy(int)]
    edges_df["j"] = active_idx[edges_df_local["j"].to_numpy(int)]
    edges_df = edges_df[["i", "j", "cosine", "weight"]]

    result = {
        "labels": labels_full,
        "n_clusters": n_clusters,
        "n_minus1": n_minus1,
        "compression": compression,
        "edges": edges_df,
    }
    if return_graph:
        result["graph"] = g
    return result
