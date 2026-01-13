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
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple, Union, Literal

import anndata as ad
import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:  # pragma: no cover
    from pymsix.core.msicube import MSICube


ScaleMode = Literal["all", "per_sample", "per_condition"]
ScaleStats = Dict[Any, Tuple[np.ndarray, np.ndarray]]
LowAction = Literal["keep", "nan", "zero", "clip"]
HighAction = Literal["keep", "nan", "clip"]


def _log1p_inplace_or_copy(X: Any, *, base: Optional[float] = None) -> Any:
    """
    Apply ``log1p`` to a dense or sparse matrix, mirroring Scanpy's behavior.

    Sparse matrices are copied before mutation to maintain integrity;
    dense inputs are modified in-place to optimize memory usage.

    Parameters
    ----------
    X : array_like or sparse matrix
        The input intensity matrix.
    base : float, optional
        If provided, the result is scaled to this logarithmic base
        (e.g., base=10 for log10).

    Returns
    -------
    X_out : same type as X
        The transformed matrix.
    """

    if sp.issparse(X):
        X = X.copy()
        X.data = np.log1p(X.data)
        if base is not None:
            X.data /= np.log(base)
        return X

    X_arr = np.asarray(X)
    if not np.issubdtype(X_arr.dtype, np.floating):
        X_arr = X_arr.astype(np.float32, copy=False)

    np.log1p(X_arr, out=X_arr)
    if base is not None:
        X_arr /= np.log(base)
    return X_arr


def log1p_intensity(
    msicube: "MSICube",
    *,
    base: Optional[float] = None,
    layer: Optional[str] = None,
    copy: bool = False,
) -> Optional["MSICube"]:
    """
    Apply a log(1 + x) transformation to the intensity matrix.

    This method calculates the natural logarithm of the intensity values plus one,
    which is a standard transformation in MSI to stabilize variance and reduce
    the skewness of high-intensity peaks. It can optionally transform to a
    specific base (e.g., base 2 or base 10).

    The operation follows the formula:
    $$f(x) = \frac{\ln(1 + x)}{\ln(base)}$$

    Parameters
    ----------
    base : float, optional
        The base of the logarithm. If None (default), the natural logarithm
        (base $e$) is used. Common values are 2 or 10.
    layer : str, optional
        The specific layer in ``adata.layers`` to transform. If None, the
        main matrix ``adata.X`` is used.
    copy : bool, default False
        Whether to return a new MSICube instance.

        * If True, a deep copy of the instance is created and returned.
        * If False, the transformation is applied in-place and None is returned.

    Returns
    -------
    MSICube or None
        The transformed MSICube instance if ``copy=True``, otherwise None.

    Raises
    ------
    ValueError
        If ``msicube.adata`` is None or if the targeted data matrix is empty.
    KeyError
        If the specified `layer` does not exist in ``adata.layers``.

    Notes
    -----
    The transformation is applied to both dense and sparse matrices efficiently.
    For sparse data, only non-zero entries are affected, preserving the
    sparsity structure of the matrix.
    """

    if msicube.adata is None:
        raise ValueError("MSICube.adata is None. Run data extraction first.")

    if copy:
        from pymsix.core.msicube import MSICube

        target_cube = MSICube(msicube.data_directory)
        target_cube.org_imzml_path_dict = msicube.org_imzml_path_dict.copy()
        target_cube.adata = msicube.adata.copy()
    else:
        target_cube = msicube

    adata_obj = target_cube.adata

    if layer is None:
        if adata_obj.X is None:
            raise ValueError("MSICube.adata.X is None.")
        adata_obj.X = _log1p_inplace_or_copy(adata_obj.X, base=base)
    else:
        if layer not in adata_obj.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        adata_obj.layers[layer] = _log1p_inplace_or_copy(
            adata_obj.layers[layer], base=base
        )

    adata_obj.uns.setdefault("log1p", {})
    adata_obj.uns["log1p"]["base"] = base

    return target_cube if copy else None


def clip_or_mask_intensities(
    msicube: "MSICube",
    *,
    low: Optional[float] = None,
    high: Optional[float] = None,
    low_action: LowAction = "nan",
    high_action: HighAction = "clip",
    layer: Optional[str] = None,
    result_layer: Optional[str] = None,
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """
    Clip or mask intensity values within the AnnData object.

    This method processes pixel intensities by applying thresholds to the
    specified data layer. It is commonly used for background subtraction,
    removing extreme outliers, or normalizing noise floor levels.

    Parameters
    ----------
    low : float, optional
        Lower threshold. Values below this threshold are processed according
        to `low_action`.
    high : float, optional
        Upper threshold. Values above this threshold are processed according
        to `high_action`.
    low_action : {'nan', 'zero', 'clip', 'keep', 'move'}, default 'nan'
        Action to perform on values less than `low`:

        * 'nan': Set values to ``np.nan``.
        * 'zero': Set values to ``0.0``.
        * 'clip': Set values to exactly `low`.
        * 'move': Subtract `low` from all values and floor results at ``0.0``.
        * 'keep': Do nothing.
    high_action : {'clip', 'nan', 'keep'}, default 'clip'
        Action to perform on values greater than `high`:

        * 'clip': Set values to exactly `high`.
        * 'nan': Set values to ``np.nan``.
        * 'keep': Do nothing.
    layer : str, optional
        The key in ``adata.layers`` to process. If None, operates on the
        main data matrix ``adata.X``.
    result_layer : str, optional
        The key in ``adata.layers`` where the processed matrix will be stored.
        If None, the input `layer` (or `X`) is overwritten in-place.
    copy : bool, default False
        If True, returns a new AnnData object with the modified data.
        If False, modifies the current :attr:`adata` instance and returns None.

    Returns
    -------
    anndata.AnnData or None
        If ``copy=True``, returns the modified AnnData object.
        Otherwise, returns None after in-place modification.

    Raises
    ------
    ValueError
        If ``msicube.adata`` is None or if 'move' action is used without a `low` value.
    KeyError
        If the specified `layer` is not found in the AnnData object.

    Notes
    -----
    **Sparse Matrix Handling:**
    For sparse matrices (CSR/CSC), only explicitly stored non-zero entries
    are modified. Implicit zeros remain zero. Using ``low_action="nan"``
    on sparse data may significantly increase memory usage or break
    downstream sparse-compatible algorithms.

    **Metadata:**
    A record of the clipping operation is appended to ``adata.uns["intensity_clipping"]``.
    """

    if msicube.adata is None:
        raise ValueError("MSICube.adata is None. Run data extraction first.")

    obj = msicube.adata.copy() if copy else msicube.adata

    if layer is None:
        X = obj.X
    else:
        if layer not in obj.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        X = obj.layers[layer]

    if low is None and high is None:
        return obj if copy else None

    if sp.issparse(X):
        X_out = X.astype(np.float32, copy=True)
        data = X_out.data

        if low_action == "move":
            if low is None:
                raise ValueError("low must be set when low_action='move'")
            np.subtract(data, low, out=data)
            data[data < 0] = 0.0
        elif low is not None and low_action != "keep":
            mask = data < low
            if low_action == "nan":
                data[mask] = np.nan
            elif low_action == "zero":
                data[mask] = 0.0
            elif low_action == "clip":
                data[mask] = low

        if high is not None and high_action != "keep":
            mask = data > high
            if high_action == "nan":
                data[mask] = np.nan
            elif high_action == "clip":
                data[mask] = high

        X_out.data = data
        if low_action in {"zero", "move"}:
            X_out.eliminate_zeros()

    else:
        X_arr = np.asarray(X, dtype=np.float32).copy()

        if low_action == "move":
            if low is None:
                raise ValueError("low must be set when low_action='move'")
            np.subtract(X_arr, low, out=X_arr)
            X_arr[X_arr < 0] = 0.0
        elif low is not None and low_action != "keep":
            if low_action == "nan":
                X_arr[X_arr < low] = np.nan
            elif low_action == "zero":
                X_arr[X_arr < low] = 0.0
            elif low_action == "clip":
                X_arr[X_arr < low] = low

        if high is not None and high_action != "keep":
            if high_action == "nan":
                X_arr[X_arr > high] = np.nan
            elif high_action == "clip":
                X_arr[X_arr > high] = high

        X_out = X_arr

    if result_layer is None and layer is None:
        obj.X = X_out
    elif result_layer is None and layer is not None:
        obj.layers[layer] = X_out
    else:
        obj.layers[result_layer] = X_out

    obj.uns.setdefault("intensity_clipping", [])
    obj.uns["intensity_clipping"].append(
        {
            "layer": layer,
            "result_layer": result_layer,
            "low": None if low is None else float(low),
            "high": None if high is None else float(high),
            "low_action": low_action,
            "high_action": high_action,
        }
    )

    return obj if copy else None


def scale_ion_images_zscore(
    msicube: "MSICube",
    *,
    mode: ScaleMode = "all",
    layer: Optional[str] = None,
    output_layer: Optional[str] = None,
    with_mean: bool = True,
    with_std: bool = True,
    ddof: int = 0,
    eps: float = 1e-8,
    max_value: Optional[float] = None,
    return_stats: bool = False,
    copy: bool = False,
) -> Union[ad.AnnData, ScaleStats, Tuple[ad.AnnData, ScaleStats], None]:
    """
    Apply Z-score scaling to ion images (m/z features) in the AnnData object.

    This method standardizes the intensity of each ion image by subtracting the
    mean and dividing by the standard deviation. Scaling is essential for
    downstream multivariate analysis (like PCA or clustering) to ensure that
    high-intensity peaks do not dominate the results simply due to their scale.

    Parameters
    ----------
    mode : {'all', 'per_sample', 'per_condition'}, default 'all'
        The grouping strategy for calculating statistics:

        * 'all': Scales using the mean/std calculated across every pixel in the
        entire dataset.
        * 'per_sample': Scales each sample independently based on
        ``adata.obs['sample']``. Useful for correcting batch effects.
        * 'per_condition': Scales independently for each group in
        ``adata.obs['condition']``.
    layer : str, optional
        The source layer in ``adata.layers`` to scale. If None, uses ``adata.X``.
    output_layer : str, optional
        The destination layer name for the scaled data. If None, the source data
        is overwritten in-place.
    with_mean : bool, default True
        If True, center the data by subtracting the mean.
    with_std : bool, default True
        If True, scale the data by dividing by the standard deviation.
    ddof : int, default 0
        Delta Degrees of Freedom. The divisor used in calculations is ``N - ddof``,
        where ``N`` is the number of pixels.
    eps : float, default 1e-8
        A small constant (epsilon) added to the standard deviation to prevent
        division by zero in constant ion images.
    max_value : float, optional
        Clip scaled values to this absolute maximum (e.g., ``[-max_value, max_value]``).
        Helpful for reducing the influence of extreme outliers.
    return_stats : bool, default False
        If True, returns the calculated mean and standard deviation arrays
        per group.
    copy : bool, default False
        If True, returns a modified copy of the AnnData object. If False,
        modifies the internal :attr:`adata` instance.

    Returns
    -------
    Union[ad.AnnData, dict, tuple, None]
        * If ``copy=True`` and ``return_stats=True``: returns ``(adata_copy, stats_dict)``.
        * If ``copy=True``: returns the new AnnData object.
        * If ``return_stats=True``: returns the dictionary of scaling statistics.
        * Otherwise: returns None (in-place modification).

    Raises
    ------
    ValueError
        If ``msicube.adata`` is None or if data is missing.
    KeyError
        If the specified `layer` or the grouping column (sample/condition)
        is missing from AnnData.

    Notes
    -----
    Standardizing "per_sample" is a common strategy in MSI to mitigate
    instrumental drift across multiple sections or slides.
    """

    if msicube.adata is None:
        raise ValueError(
            "MSICube.adata is None. Run data extraction before scaling ion images."
        )

    obj = msicube.adata.copy() if copy else msicube.adata

    if layer is None:
        X = obj.X
    else:
        if layer not in obj.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        X = obj.layers[layer]

    if X is None:
        raise ValueError("No ion image data found to scale (adata.X is None).")

    if mode == "all":
        groups = {None: np.ones(obj.n_obs, dtype=bool)}
        group_key: Optional[str] = None
    elif mode == "per_sample":
        group_key = "sample"
        if group_key not in obj.obs:
            raise KeyError("mode='per_sample' requires adata.obs['sample']")
        vals = obj.obs[group_key].astype("category")
        groups = {k: (vals == k).to_numpy() for k in vals.cat.categories}
    elif mode == "per_condition":
        group_key = "condition"
        if group_key not in obj.obs:
            raise KeyError("mode='per_condition' requires adata.obs['condition']")
        vals = obj.obs[group_key].astype("category")
        groups = {k: (vals == k).to_numpy() for k in vals.cat.categories}
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    def _scale_dense_block(Xsub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.isnan(Xsub).any():
            mean = (
                np.nanmean(Xsub, axis=0)
                if with_mean
                else np.zeros(Xsub.shape[1], dtype=np.float64)
            )
            std = (
                np.nanstd(Xsub, axis=0, ddof=ddof)
                if with_std
                else np.ones(Xsub.shape[1], dtype=np.float64)
            )
        else:
            mean = (
                Xsub.mean(axis=0)
                if with_mean
                else np.zeros(Xsub.shape[1], dtype=np.float64)
            )
            std = (
                Xsub.std(axis=0, ddof=ddof)
                if with_std
                else np.ones(Xsub.shape[1], dtype=np.float64)
            )

        std = np.asarray(std, dtype=np.float64)
        std[std < eps] = 1.0

        if with_mean:
            Xsub -= mean
        if with_std:
            Xsub /= std

        if max_value is not None:
            np.clip(Xsub, -max_value, max_value, out=Xsub)

        return np.asarray(mean, dtype=np.float64), std

    stats: ScaleStats = {}

    if sp.issparse(X):
        X_lil = X.tolil(copy=True)
        for g, m in groups.items():
            idx = np.where(m)[0]
            if idx.size == 0:
                continue
            block = X[idx, :].toarray().astype(np.float32, copy=False)
            mean, std = _scale_dense_block(block)
            stats[g] = (mean, std)
            X_lil[idx, :] = block

        X_out = X_lil.tocsr()
    else:
        X_out = np.asarray(X).astype(np.float32, copy=False)
        if not X_out.flags.writeable:
            X_out = X_out.copy()

        for g, m in groups.items():
            idx = np.where(m)[0]
            if idx.size == 0:
                continue
            block = np.array(X_out[idx, :], dtype=np.float32, copy=False)
            mean, std = _scale_dense_block(block)
            stats[g] = (mean, std)
            X_out[idx, :] = block

    if output_layer is None:
        if layer is None:
            obj.X = X_out
        else:
            obj.layers[layer] = X_out
    else:
        obj.layers[output_layer] = X_out

    obj.uns.setdefault("scale_ion_images_zscore", {})
    obj.uns["scale_ion_images_zscore"].update(
        {
            "mode": mode,
            "group_key": group_key,
            "layer": layer,
            "output_layer": output_layer,
            "with_mean": with_mean,
            "with_std": with_std,
            "ddof": int(ddof),
            "eps": float(eps),
            "max_value": None if max_value is None else float(max_value),
        }
    )

    if copy:
        if return_stats:
            return obj, stats
        return obj

    return stats if return_stats else None

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
