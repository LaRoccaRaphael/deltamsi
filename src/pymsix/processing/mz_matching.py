"""
M/Z Matching Utilities
======================

Helper for matching query m/z values against the m/z axis stored in
``adata.var``. The function can optionally annotate matched variables
with user-provided labels.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from pymsix.core.msicube import MSICube


def match_mzs_to_var_simple(
    msicube: "MSICube",
    mzs: Iterable[float],
    *,
    mz_col: str = "mz",
    mode: Literal["closest", "tolerance"] = "closest",
    tol: float = 5.0,
    tol_unit: Literal["ppm", "da"] = "ppm",
    return_all_within_tol: bool = True,
    assume_sorted: bool = False,
    annotation: Optional[Union[str, Sequence[Optional[str]]]] = None,
    annotation_col: Optional[str] = None,
    multi_write: Literal["overwrite", "append"] = "append",
    sep: str = ";",
) -> pd.DataFrame:
    """
    Match query m/z values to ``msicube.adata.var`` rows.

    This function matches query m/z values against the metadata column
    ``mz_col`` in ``msicube.adata.var``. Matching can be performed by
    closest value or within a tolerance window. If ``annotation`` and
    ``annotation_col`` are provided, matched rows are annotated.

    Parameters
    ----------
    msicube : MSICube
        The MSICube instance containing the AnnData object to query.
    mzs : Iterable[float]
        Query m/z values to match against ``adata.var[mz_col]``.
    mz_col : str, default "mz"
        Column name in ``adata.var`` storing m/z values.
    mode : {"closest", "tolerance"}, default "closest"
        Matching strategy.
    tol : float, default 5.0
        Tolerance value when ``mode="tolerance"``.
    tol_unit : {"ppm", "da"}, default "ppm"
        Units for the tolerance value.
    return_all_within_tol : bool, default True
        When using tolerance mode, return all matches within tolerance.
    assume_sorted : bool, default False
        If True, assumes ``adata.var[mz_col]`` is already sorted.
    annotation : str or sequence, optional
        Annotation(s) to write to matched variables.
    annotation_col : str, optional
        Column name in ``adata.var`` to write annotations into.
    multi_write : {"overwrite", "append"}, default "append"
        Behavior when multiple annotations map to the same variable.
    sep : str, default ";"
        Separator used for appending annotations.

    Returns
    -------
    pd.DataFrame
        A DataFrame with match results (one row per query m/z).
    """

    if msicube.adata is None:
        raise ValueError("MSICube.adata is None. Load or compute data first.")

    adata = msicube.adata
    q = np.asarray(list(mzs), dtype=float)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("mzs must be a non-empty 1D iterable of floats.")
    if np.any(~np.isfinite(q)):
        raise ValueError("mzs contains NaN/inf.")

    if mz_col not in adata.var.columns:
        raise KeyError(f"adata.var does not contain mz_col={mz_col!r}.")
    mz_var = np.asarray(adata.var[mz_col], dtype=float)
    if mz_var.size != adata.n_vars or np.any(~np.isfinite(mz_var)):
        raise ValueError(f"adata.var[{mz_col!r}] must be finite and length n_vars.")

    if assume_sorted:
        order = np.arange(adata.n_vars, dtype=int)
        mz_sorted = mz_var
    else:
        order = np.argsort(mz_var)
        mz_sorted = mz_var[order]

    do_annotate = (annotation is not None) and (annotation_col is not None)
    if do_annotate:
        if annotation_col not in adata.var.columns:
            adata.var[annotation_col] = pd.Series(
                [pd.NA] * adata.n_vars,
                index=adata.var.index,
                dtype="string",
            )
        else:
            adata.var[annotation_col] = adata.var[annotation_col].astype("string")

        if isinstance(annotation, str):
            ann_list = [annotation] * q.size
        else:
            ann_list = list(annotation)
            if len(ann_list) != q.size:
                raise ValueError("If annotation is a sequence, it must have same length as mzs.")
    else:
        ann_list = [None] * q.size

    def _write(pos: int, ann: str) -> None:
        col_idx = adata.var.columns.get_loc(annotation_col)  # type: ignore[arg-type]
        cur = adata.var.iloc[pos, col_idx]
        if multi_write == "append" and pd.notna(cur) and str(cur) != "":
            parts = set(str(cur).split(sep))
            if ann not in parts:
                adata.var.iloc[pos, col_idx] = str(cur) + sep + ann
        else:
            adata.var.iloc[pos, col_idx] = ann

    def _ppm_to_da(mz0: float, ppm: float) -> float:
        return (ppm * 1e-6) * mz0

    rows = []
    for k, mzq in enumerate(q):
        if mode == "closest":
            idx = int(np.searchsorted(mz_sorted, mzq))
            cand = []
            if idx > 0:
                cand.append(idx - 1)
            if idx < mz_sorted.size:
                cand.append(idx)
            best_sidx = min(cand, key=lambda i: abs(mz_sorted[i] - mzq))
            pos = int(order[best_sidx])
            mz_match = float(mz_var[pos])
            err_da = float(mz_match - mzq)
            err_ppm = float(err_da / mzq * 1e6) if mzq != 0 else np.nan

            if do_annotate and ann_list[k] is not None:
                _write(pos, str(ann_list[k]))

            rows.append(
                dict(
                    query_mz=mzq,
                    n_matches=1,
                    match_pos=pos,
                    match_mz=mz_match,
                    err_da=err_da,
                    err_ppm=err_ppm,
                )
            )

        elif mode == "tolerance":
            tol_da = _ppm_to_da(mzq, tol) if tol_unit == "ppm" else float(tol)
            lo, hi = mzq - tol_da, mzq + tol_da
            left = int(np.searchsorted(mz_sorted, lo, side="left"))
            right = int(np.searchsorted(mz_sorted, hi, side="right"))
            sidxs = np.arange(left, right, dtype=int)

            if sidxs.size == 0:
                rows.append(
                    dict(
                        query_mz=mzq,
                        n_matches=0,
                        match_pos=[] if return_all_within_tol else np.nan,
                        match_mz=[] if return_all_within_tol else np.nan,
                        err_da=[] if return_all_within_tol else np.nan,
                        err_ppm=[] if return_all_within_tol else np.nan,
                        tol_da=tol_da,
                        tol_unit=tol_unit,
                        tol_value=float(tol),
                    )
                )
                continue

            poss = order[sidxs].astype(int)
            mzs_match = mz_var[poss].astype(float)
            err_da = mzs_match - mzq
            err_ppm = (err_da / mzq * 1e6) if mzq != 0 else np.full_like(err_da, np.nan)

            if return_all_within_tol:
                rows.append(
                    dict(
                        query_mz=mzq,
                        n_matches=int(poss.size),
                        match_pos=poss.tolist(),
                        match_mz=mzs_match.tolist(),
                        err_da=err_da.tolist(),
                        err_ppm=err_ppm.tolist(),
                        tol_da=tol_da,
                        tol_unit=tol_unit,
                        tol_value=float(tol),
                    )
                )
                write_pos = poss
            else:
                best = int(np.argmin(np.abs(err_da)))
                pos = int(poss[best])
                rows.append(
                    dict(
                        query_mz=mzq,
                        n_matches=int(poss.size),
                        match_pos=pos,
                        match_mz=float(mzs_match[best]),
                        err_da=float(err_da[best]),
                        err_ppm=float(err_ppm[best]),
                        tol_da=tol_da,
                        tol_unit=tol_unit,
                        tol_value=float(tol),
                    )
                )
                write_pos = np.array([pos], dtype=int)

            if do_annotate and ann_list[k] is not None:
                for pos in np.asarray(write_pos, dtype=int):
                    _write(int(pos), str(ann_list[k]))

        else:
            raise ValueError("mode must be 'closest' or 'tolerance'.")

    return pd.DataFrame(rows)
