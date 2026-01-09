"""
Discriminant Analysis and Marker Discovery
==========================================

This module implements statistical methods to identify ions that significantly 
differ between groups of pixels or samples. It provides a Scanpy-like interface 
specifically optimized for Mass Spectrometry Imaging (MSI).

The analysis automatically adapts to your experimental design:
* **Single Sample**: Uses pixel-level effect sizes with optional spatial block bootstrapping.
* **Replicated Design**: Uses a pseudobulk-per-sample approach with statistical testing 
  (t-test or Wilcoxon) and FDR correction.


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

__all__ = ["RankIonsMSIParams", "rank_ions_groups_msi"]


# ---------------------------
# Helpers
# ---------------------------
def _get_X(adata: ad.AnnData, layer: Optional[str]):
    """
    Extract the relevant data matrix from an AnnData object.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object.
    layer : str, optional
        Name of the layer to extract. If None, returns ``adata.X``.

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        The requested data matrix.
    """
    return adata.layers[layer] if layer is not None else adata.X


def _mean_axis0(X) -> np.ndarray:
    """
    Compute the mean along axis 0 (column-wise).

    Parameters
    ----------
    X : array_like or sparse matrix
        Input data matrix of shape (n_observations, n_variables).

    Returns
    -------
    np.ndarray
        1D array of shape (n_variables,) containing the means.
    """
    if sp.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    return np.asarray(X.mean(axis=0)).ravel()


def _median_axis0_dense(X: np.ndarray) -> np.ndarray:
    """
    Compute the median along axis 0 for a dense array.

    Parameters
    ----------
    X : np.ndarray
        Dense input matrix.

    Returns
    -------
    np.ndarray
        1D array of shape (n_variables,) containing the medians.
    """
    return np.median(X, axis=0)


def _pct_detected(X, *, threshold: float = 0.0) -> np.ndarray:
    """
    Calculate the fraction of observations above a threshold for each variable.

    Parameters
    ----------
    X : array_like or sparse matrix
        Input data matrix.
    threshold : float, default 0.0
        The value above which a variable is considered "detected".

    Returns
    -------
    np.ndarray
        1D array of shape (n_variables,) with values in range [0, 1].
    """
    n = X.shape[0]
    if n == 0:
        return np.zeros(X.shape[1], dtype=float)

    if threshold == 0.0 and sp.issparse(X):
        Xc = X.tocsc(copy=False)
        nnz_col = np.diff(Xc.indptr).astype(float)
        return nnz_col / float(n)

    if sp.issparse(X):
        X = X.toarray()
    return (np.asarray(X) > threshold).mean(axis=0)


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """
    Apply Benjamini–Hochberg False Discovery Rate (FDR) adjustment.

    Parameters
    ----------
    p : np.ndarray
        1D array of p-values. Can contain ``np.nan`` values which will be ignored.

    Returns
    -------
    np.ndarray
        Adjusted p-values (q-values) of the same shape as input.

    Examples
    --------
    >>> pvals = np.array([0.01, 0.05, 0.5, np.nan])
    >>> _bh_fdr(pvals)
    array([0.04, 0.1 , 0.5 ,  nan])
    """
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    m = np.isfinite(p)
    if not np.any(m):
        return out
    pv = p[m]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[m] = tmp
    return out


def _log2fc(a: np.ndarray, b: np.ndarray, pseudocount: float) -> np.ndarray:
    """
    Compute the Log2 Fold Change between two arrays.

    Parameters
    ----------
    a : np.ndarray
        Numerator array (e.g., condition intensities).
    b : np.ndarray
        Denominator array (e.g., control intensities).
    pseudocount : float
        A value added to both numerator and denominator to prevent 
        division by zero and stabilize variance.

    Returns
    -------
    np.ndarray
        Calculated Log2 Fold Change.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.log2((a + pseudocount) / (b + pseudocount))


def _infer_blocks(
    adata: ad.AnnData,
    obs_idx: np.ndarray,
    *,
    block_size: int,
    x_key: str,
    y_key: str,
    spatial_key: str,
) -> np.ndarray:
    """
    Partition spatial observations into discrete block IDs.

    This function calculates which rectangular tile (block) each pixel belongs to 
    based on its spatial coordinates and a fixed block size.

    Parameters
    ----------
    adata : ad.AnnData
        The annotated data object.
    obs_idx : np.ndarray
        The indices of the observations to process.
    block_size : int
        The width/height of the square blocks in pixel units.
    x_key : str
        Column name for X coordinates in ``adata.obs``.
    y_key : str
        Column name for Y coordinates in ``adata.obs``.
    spatial_key : str
        Key name for coordinates in ``adata.obsm``. Only used if x_key/y_key 
        are not in ``adata.obs``.

    Returns
    -------
    np.ndarray
        1D array of integer block IDs.

    Notes
    -----
    The coordinates are normalized to start at 0 before block assignment.
    """

    if x_key in adata.obs.columns and y_key in adata.obs.columns:
        x = np.asarray(adata.obs.loc[adata.obs.index[obs_idx], x_key], dtype=int)
        y = np.asarray(adata.obs.loc[adata.obs.index[obs_idx], y_key], dtype=int)
    elif spatial_key in adata.obsm:
        xy = np.asarray(adata.obsm[spatial_key])[obs_idx, :2]
        x = xy[:, 0].astype(int)
        y = xy[:, 1].astype(int)
    else:
        raise KeyError(
            f"Need obs[{x_key},{y_key}] or obsm['{spatial_key}'] for block bootstrap."
        )

    x = x - int(x.min())
    y = y - int(y.min())
    bx = x // int(block_size)
    by = y // int(block_size)
    nbx = int(bx.max()) + 1
    return (by * nbx + bx).astype(np.int64)


# ---------------------------
# Main API
# ---------------------------
@dataclass
class RankIonsMSIParams:
    """
    Parameters for the ranking of ions between groups.

    Attributes
    ----------
    condition_key : str, default "condition"
        Column in ``adata.obs`` containing the experimental groups/conditions.
    sample_key : str, default "sample"
        Column in ``adata.obs`` identifying individual biological replicates.
    group : str, default "treated"
        The name of the condition to test (the "numerator").
    reference : str, default "control"
        The name of the condition to use as a baseline (the "denominator").
    layer : str, optional
        The AnnData layer to use for intensity values. If None, uses ``adata.X``.
    detection_threshold : float, default 0.0
        Intensity value above which an ion is considered "detected".
    pseudocount : float, default 1e-9
        Constant added to denominators to avoid division by zero in log2FC.
    agg : {"mean", "median"}, default "mean"
        Method to summarize pixels into sample-level pseudobulk values.
    method : {"auto", "ttest", "wilcoxon"}, default "auto"
        Statistical test to perform when replicates are available.
    direction : {"up", "abs"}, default "up"
        "up" focuses on ions overexpressed in `group`. "abs" ranks by absolute fold change.
    n_top : int, default 200
        Number of top ions to return in the summary table.
    compute_auc : bool, default True
        Whether to calculate the Area Under the Curve (Receiver Operating Characteristic).
    block_bootstrap : bool, default False
        Whether to use spatial block bootstrapping to estimate confidence 
        intervals for single-sample comparisons.
    block_size : int, default 25
        Side length of the square spatial blocks (in pixels) for bootstrapping.
    key_added : str, default "rank_ions_groups_msi"
        Key under which results are stored in ``adata.uns``.
    """
    condition_key: str = "condition"
    sample_key: str = "sample"
    group: str = "treated"  # user-selected
    reference: str = "control"  # user-selected

    layer: Optional[str] = None

    # effect sizes
    detection_threshold: float = 0.0
    pseudocount: float = 1e-9
    agg: Literal["mean", "median"] = "mean"  # for pseudobulk per sample; median requires dense

    # statistics (only when replicated)
    method: Literal["auto", "ttest", "wilcoxon"] = "auto"

    # ranking
    direction: Literal["up", "abs"] = "up"  # 'up' ranks overexpressed in group; 'abs' ranks by |logFC|
    n_top: int = 200

    # speed controls
    compute_auc: bool = True
    auc_on: Literal["auto", "samples", "pixels"] = "auto"
    auc_max_ions: int = 3000  # compute AUC only for top K by ranking score
    auc_max_pixels_per_group: int = 50000  # subsample pixels for AUC when using pixel-level

    # single-sample uncertainty (optional)
    block_bootstrap: bool = False
    block_size: int = 25
    n_boot: int = 200
    ci_alpha: float = 0.05
    ci_max_ions: int = 1000  # compute CI only for top K ions
    x_key: str = "x"
    y_key: str = "y"
    spatial_key: str = "spatial"

    # output
    key_added: str = "rank_ions_groups_msi"
    random_state: int = 0


def rank_ions_groups_msi(adata: ad.AnnData, *, params: RankIonsMSIParams) -> pd.DataFrame:
    """
    Rank ions by differential expression between two groups of MSI data.

    This function identifies marker ions by comparing a group condition against 
    a reference condition. It computes effect sizes (Log2 Fold Change, 
    Delta Detection Rate) and, if replicates are present, statistical 
    significance (p-values).

    Parameters
    ----------
    adata : ad.AnnData
        The annotated data matrix (pixels x ions).
    params : RankIonsMSIParams
        Configuration object specifying groups, stats, and output keys.

    Returns
    -------
    res_top : pd.DataFrame
        A table of the top `n_top` ranked ions with their associated statistics 
        (log2fc, pvals, AUC, etc.).

    Notes
    -----
    **Spatial Block Bootstrap**
    In MSI, pixels are not independent observations. If you have only one 
    sample per group, standard p-values are often artificially low. Enabling 
    `block_bootstrap` resamples square tiles of pixels to provide a more 
    realistic Confidence Interval (CI) for the Fold Change.

    

    **Storage**
    Results are saved in ``adata.uns[params.key_added]`` in a format 
    compatible with common visualization tools, including the ranking 
    scores and full statistical tables.

    Examples
    --------
    >>> from pymsix.processing.discriminant import rank_ions_groups_msi, RankIonsMSIParams
    >>> # Define comparison between Treated and Control
    >>> p = RankIonsMSIParams(
    ...     condition_key="group", 
    ...     group="Treated", 
    ...     reference="Control",
    ...     method="ttest"
    ... )
    >>> # Run analysis
    >>> top_ions = rank_ions_groups_msi(adata, params=p)
    >>> print(top_ions.head())
    """

    ck = params.condition_key
    sk = params.sample_key
    group = params.group
    ref = params.reference

    if ck not in adata.obs.columns:
        raise KeyError(f"adata.obs['{ck}'] not found.")
    if sk not in adata.obs.columns:
        raise KeyError(f"adata.obs['{sk}'] not found.")

    obs_cond = adata.obs[ck].astype(str)
    obs_samp = adata.obs[sk].astype(str)

    mask_group = (obs_cond == str(group)).to_numpy()
    mask_ref = (obs_cond == str(ref)).to_numpy()
    if not mask_group.any():
        raise ValueError(f"No obs with {ck} == {group!r}.")
    if not mask_ref.any():
        raise ValueError(f"No obs with {ck} == {ref!r}.")

    X = _get_X(adata, params.layer)

    # sample counts per condition
    group_samples = pd.unique(obs_samp[mask_group])
    ref_samples = pd.unique(obs_samp[mask_ref])
    n_g = len(group_samples)
    n_r = len(ref_samples)

    replicated = n_g >= 2 and n_r >= 2

    # --- Effect sizes on pooled pixels (always computed)
    Xg = X[mask_group, :]
    Xr = X[mask_ref, :]
    if params.agg == "median":
        if sp.issparse(Xg) or sp.issparse(Xr):
            raise ValueError("agg='median' requires dense matrix (or convert layer to dense).")

    mean_g_pix = _mean_axis0(Xg) if params.agg == "mean" else _median_axis0_dense(np.asarray(Xg))
    mean_r_pix = _mean_axis0(Xr) if params.agg == "mean" else _median_axis0_dense(np.asarray(Xr))

    pct_g_pix = _pct_detected(Xg, threshold=params.detection_threshold)
    pct_r_pix = _pct_detected(Xr, threshold=params.detection_threshold)

    logfc_pix = _log2fc(mean_g_pix, mean_r_pix, params.pseudocount)
    delta_pct_pix = pct_g_pix - pct_r_pix

    if params.direction == "up":
        base_score = logfc_pix
    else:
        base_score = np.abs(logfc_pix)

    order0 = np.argsort(-base_score)  # descending

    # --- Replicated mode: pseudobulk per sample + stats
    pvals = np.full(adata.n_vars, np.nan, dtype=float)
    pvals_adj = np.full(adata.n_vars, np.nan, dtype=float)
    stat_score = np.full(adata.n_vars, np.nan, dtype=float)

    mean_g_samp = mean_g_pix.copy()
    mean_r_samp = mean_r_pix.copy()
    pct_g_samp = pct_g_pix.copy()
    pct_r_samp = pct_r_pix.copy()
    logfc_samp = logfc_pix.copy()
    delta_pct_samp = delta_pct_pix.copy()

    analysis_mode = "single_sample_effects"

    if replicated:
        analysis_mode = "pseudobulk_samples"

        def _summarize_by_samples(samples: np.ndarray, cond_value: str) -> Tuple[np.ndarray, np.ndarray]:
            vals = []
            dets = []
            for s in samples:
                m = ((obs_samp == s) & (obs_cond == cond_value)).to_numpy()
                Xs = X[m, :]
                if Xs.shape[0] == 0:
                    continue
                if params.agg == "mean":
                    vals.append(_mean_axis0(Xs))
                else:
                    if sp.issparse(Xs):
                        raise ValueError("agg='median' requires dense matrix.")
                    vals.append(np.median(np.asarray(Xs), axis=0))
                dets.append(_pct_detected(Xs, threshold=params.detection_threshold))
            if not vals:
                return np.zeros((0, adata.n_vars), dtype=float), np.zeros((0, adata.n_vars), dtype=float)
            return np.vstack(vals), np.vstack(dets)

        G_vals, G_det = _summarize_by_samples(group_samples, str(group))
        R_vals, R_det = _summarize_by_samples(ref_samples, str(ref))

        if G_vals.shape[0] < 2 or R_vals.shape[0] < 2:
            analysis_mode = "single_sample_effects"
        else:
            mean_g_samp = G_vals.mean(axis=0)
            mean_r_samp = R_vals.mean(axis=0)
            pct_g_samp = G_det.mean(axis=0)
            pct_r_samp = R_det.mean(axis=0)
            logfc_samp = _log2fc(mean_g_samp, mean_r_samp, params.pseudocount)
            delta_pct_samp = pct_g_samp - pct_r_samp

            method = params.method
            if method == "auto":
                method = "ttest" if (G_vals.shape[0] >= 3 and R_vals.shape[0] >= 3) else "wilcoxon"

            if method == "ttest":
                t = stats.ttest_ind(G_vals, R_vals, axis=0, equal_var=False, nan_policy="omit")
                pvals = np.asarray(t.pvalue, dtype=float)
                stat_score = np.asarray(t.statistic, dtype=float)
            elif method == "wilcoxon":
                pv = np.full(adata.n_vars, np.nan, dtype=float)
                sc = np.full(adata.n_vars, np.nan, dtype=float)
                for j in range(adata.n_vars):
                    a = G_vals[:, j]
                    b = R_vals[:, j]
                    if np.all(~np.isfinite(a)) or np.all(~np.isfinite(b)):
                        continue
                    try:
                        res = stats.mannwhitneyu(a, b, alternative="two-sided")
                        pv[j] = float(res.pvalue)
                        sc[j] = float(res.statistic) / (a.size * b.size) - 0.5
                    except Exception:
                        continue
                pvals = pv
                stat_score = sc
            else:
                raise ValueError("method must be 'auto', 'ttest', or 'wilcoxon'.")

            pvals_adj = _bh_fdr(pvals)

            if params.direction == "up":
                base_score = logfc_samp
            else:
                base_score = np.abs(logfc_samp)
            order0 = np.argsort(-base_score)

    # --- AUC (optional; computed on top ions only)
    auc = np.full(adata.n_vars, np.nan, dtype=float)
    if params.compute_auc:
        rng = np.random.default_rng(params.random_state)
        top_auc_idx = order0[: min(params.auc_max_ions, adata.n_vars)]

        auc_on = params.auc_on
        if auc_on == "auto":
            auc_on = "samples" if replicated else "pixels"

        if auc_on == "samples" and replicated:

            def _sample_means(samples: np.ndarray, cond_value: str) -> np.ndarray:
                vals = []
                for s in samples:
                    m = ((obs_samp == s) & (obs_cond == cond_value)).to_numpy()
                    Xs = X[m, :]
                    if Xs.shape[0] == 0:
                        continue
                    vals.append(_mean_axis0(Xs))
                return np.vstack(vals) if vals else np.zeros((0, adata.n_vars), dtype=float)

            Gs = _sample_means(group_samples, str(group))
            Rs = _sample_means(ref_samples, str(ref))

            if Gs.shape[0] > 0 and Rs.shape[0] > 0:
                for j in top_auc_idx:
                    a = Gs[:, j]
                    b = Rs[:, j]
                    try:
                        res = stats.mannwhitneyu(a, b, alternative="two-sided")
                        auc[j] = float(res.statistic) / (a.size * b.size)
                    except Exception:
                        pass

        elif auc_on == "pixels":
            g_idx = np.where(mask_group)[0]
            r_idx = np.where(mask_ref)[0]
            if g_idx.size > params.auc_max_pixels_per_group:
                g_idx = rng.choice(g_idx, size=params.auc_max_pixels_per_group, replace=False)
            if r_idx.size > params.auc_max_pixels_per_group:
                r_idx = rng.choice(r_idx, size=params.auc_max_pixels_per_group, replace=False)

            Xg_s = X[g_idx, :]
            Xr_s = X[r_idx, :]

            for j in top_auc_idx:
                a = Xg_s[:, j]
                b = Xr_s[:, j]
                if sp.issparse(Xg_s):
                    a = a.toarray().ravel()
                    b = b.toarray().ravel()
                else:
                    a = np.asarray(a).ravel()
                    b = np.asarray(b).ravel()
                try:
                    res = stats.mannwhitneyu(a, b, alternative="two-sided")
                    auc[j] = float(res.statistic) / (a.size * b.size)
                except Exception:
                    pass

    # --- Optional block bootstrap CIs (single-sample mode only; top ions only)
    ci_low = np.full(adata.n_vars, np.nan, dtype=float)
    ci_high = np.full(adata.n_vars, np.nan, dtype=float)

    if params.block_bootstrap and (analysis_mode == "single_sample_effects"):
        rng = np.random.default_rng(params.random_state)
        top_ci_idx = order0[: min(params.ci_max_ions, adata.n_vars)]

        g_obs = np.where(mask_group)[0]
        r_obs = np.where(mask_ref)[0]

        g_blocks = _infer_blocks(
            adata,
            g_obs,
            block_size=params.block_size,
            x_key=params.x_key,
            y_key=params.y_key,
            spatial_key=params.spatial_key,
        )
        r_blocks = _infer_blocks(
            adata,
            r_obs,
            block_size=params.block_size,
            x_key=params.x_key,
            y_key=params.y_key,
            spatial_key=params.spatial_key,
        )

        def _block_means(obs_idx: np.ndarray, block_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            uniq = np.unique(block_ids)
            out = np.zeros((uniq.size, top_ci_idx.size), dtype=np.float32)
            for bi, b in enumerate(uniq):
                sel = obs_idx[block_ids == b]
                Xb = X[sel, :][:, top_ci_idx]
                out[bi, :] = _mean_axis0(Xb).astype(np.float32)
            return uniq, out

        _, Gbm = _block_means(g_obs, g_blocks)
        _, Rbm = _block_means(r_obs, r_blocks)

        if Gbm.shape[0] >= 2 and Rbm.shape[0] >= 2:
            boot = np.zeros((params.n_boot, top_ci_idx.size), dtype=np.float32)
            for t in range(params.n_boot):
                gi = rng.integers(0, Gbm.shape[0], size=Gbm.shape[0])
                ri = rng.integers(0, Rbm.shape[0], size=Rbm.shape[0])
                mg = Gbm[gi].mean(axis=0)
                mr = Rbm[ri].mean(axis=0)
                boot[t] = _log2fc(mg, mr, params.pseudocount).astype(np.float32)

            a = float(params.ci_alpha)
            lo = np.quantile(boot, a / 2.0, axis=0)
            hi = np.quantile(boot, 1.0 - a / 2.0, axis=0)
            ci_low[top_ci_idx] = lo
            ci_high[top_ci_idx] = hi

    # ---------------------------
    # Build result table
    # ---------------------------
    var_names = adata.var_names.astype(str).to_numpy()
    mz = (
        adata.var["mz"].to_numpy()
        if "mz" in adata.var.columns
        else np.full(adata.n_vars, np.nan)
    )

    res = pd.DataFrame(
        {
            "ion": var_names,
            "mz": mz,
            "log2fc_pixels": logfc_pix,
            "delta_detect_pixels": delta_pct_pix,
            "mean_pixels_group": mean_g_pix,
            "mean_pixels_ref": mean_r_pix,
            "pct_detect_pixels_group": pct_g_pix,
            "pct_detect_pixels_ref": pct_r_pix,
            "log2fc": logfc_samp,
            "delta_detect": delta_pct_samp,
            "mean_group": mean_g_samp,
            "mean_ref": mean_r_samp,
            "pct_detect_group": pct_g_samp,
            "pct_detect_ref": pct_r_samp,
            "auc": auc,
            "pval": pvals,
            "pval_adj": pvals_adj,
            "stat_score": stat_score,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )

    if params.direction == "up":
        res["score"] = res["log2fc"]
        res = res.sort_values("score", ascending=False)
    else:
        res["score"] = np.abs(res["log2fc"])
        res = res.sort_values("score", ascending=False)

    res_top = res.head(int(params.n_top)).reset_index(drop=True)

    # ---------------------------
    # Store in adata.uns (scanpy-ish)
    # ---------------------------
    key = params.key_added
    adata.uns[key] = {
        "params": {
            "condition_key": ck,
            "sample_key": sk,
            "group": group,
            "reference": ref,
            "layer": params.layer,
            "agg": params.agg,
            "method": params.method,
            "analysis_mode": analysis_mode,
            "n_samples_group": int(n_g),
            "n_samples_ref": int(n_r),
            "detection_threshold": float(params.detection_threshold),
            "pseudocount": float(params.pseudocount),
            "compute_auc": bool(params.compute_auc),
            "auc_on": params.auc_on,
            "block_bootstrap": bool(params.block_bootstrap),
            "block_size": int(params.block_size),
            "n_boot": int(params.n_boot),
            "ci_alpha": float(params.ci_alpha),
            "direction": params.direction,
        },
        "names": res["ion"].to_numpy(),
        "mz": res["mz"].to_numpy(),
        "scores": res["score"].to_numpy(),
        "log2fc": res["log2fc"].to_numpy(),
        "delta_detect": res["delta_detect"].to_numpy(),
        "auc": res["auc"].to_numpy(),
        "pvals": res["pval"].to_numpy(),
        "pvals_adj": res["pval_adj"].to_numpy(),
        "mean_group": res["mean_group"].to_numpy(),
        "mean_ref": res["mean_ref"].to_numpy(),
        "pct_detect_group": res["pct_detect_group"].to_numpy(),
        "pct_detect_ref": res["pct_detect_ref"].to_numpy(),
        "ci_low": res["ci_low"].to_numpy(),
        "ci_high": res["ci_high"].to_numpy(),
        "top_table": res_top,
    }

    return res_top
