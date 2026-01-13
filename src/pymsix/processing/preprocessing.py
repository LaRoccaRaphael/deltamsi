"""
MSI Image-Space Preprocessing
=============================

This module provides spatial-aware filtering and intensity adjustment tools 
for MSI data. It allows for:
1. **Hotspot Removal**: Capping extreme intensities that bias visualization.
2. **Thresholding**: Removing background noise based on intensity distributions.
3. **Spatial Filtering**: Smoothing ion images using 2D median filters while 
   preserving edges.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy import ndimage

from pymsix.params.options import (
    MSIHotspotCapParams,
    MSIMedianFilterParams,
    MSIThresholdParams,
)

if TYPE_CHECKING:  # pragma: no cover
    from pymsix.core.msicube import MSICube


def _get_matrix(msicube: "MSICube", layer: Optional[str]):
    """
    Internal helper to retrieve the intensity matrix from AnnData.

    Parameters
    ----------
    msicube : MSICube
        The MSICube instance containing the annotated data.
    layer : str, optional
        Name of the layer to retrieve. If None, the main matrix ``adata.X`` 
        is returned.

    Returns
    -------
    Union[np.ndarray, sp.spmatrix]
        The intensity matrix (dense or sparse).

    Raises
    ------
    ValueError
        If ``msicube.adata`` is not yet initialized.
    KeyError
        If the specified ``layer`` does not exist in the AnnData object.
    """
    adata = msicube.adata
    if adata is None:
        raise ValueError("MSICube.adata is None. Run data extraction first.")

    if layer is None:
        return adata.X

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers")

    return adata.layers[layer]


def _set_matrix(msicube: "MSICube", layer: Optional[str], X):
    """
    Internal helper to store the intensity matrix back into AnnData.

    Parameters
    ----------
    msicube : MSICube
        The MSICube instance where data will be stored.
    layer : str, optional
        Name of the layer to update. If None, updates ``adata.X``.
    X : array_like or sparse matrix
        The new intensity data to be stored.

    Raises
    ------
    ValueError
        If ``msicube.adata`` is not yet initialized.
    """
    adata = msicube.adata
    if adata is None:
        raise ValueError("MSICube.adata is None. Run data extraction first.")

    if layer is None:
        adata.X = X
    else:
        adata.layers[layer] = X


def _ensure_dense_float(X, dtype=np.float32) -> np.ndarray:
    """
    Internal helper to convert matrices to dense arrays for image processing.

    This function is critical for algorithms that do not support sparse 
    inputs (e.g., convolution, morphology, or certain scikit-image functions).

    Parameters
    ----------
    X : array_like or sparse matrix
        Input intensity data.
    dtype : np.dtype, default np.float32
        The desired numerical precision for the output array.

    Returns
    -------
    np.ndarray
        A dense NumPy array of the specified dtype.
    """
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=dtype)


def _infer_grid_index(
    msicube: "MSICube",
    *,
    x_key: str = "x",
    y_key: str = "y",
    spatial_key: str = "spatial",
    shape: Optional[Tuple[int, int]] = None,
    origin: Literal["min", "zero"] = "min",
) -> Tuple[int, int, np.ndarray]:
    """
    Map observations to a dense (H, W) grid index based on pixel coordinates.

    This is used to transform the flattened MSI data (N pixels) into a 2D 
    structure (Height x Width) required for spatial filters.

    Returns
    -------
    tuple
        ``(H, W, flat_index_for_obs)`` where ``flat_index_for_obs`` has length
        ``n_obs``.
    """

    adata = msicube.adata
    if adata is None:
        raise ValueError("MSICube.adata is None. Run data extraction first.")

    if x_key in adata.obs.columns and y_key in adata.obs.columns:
        x = np.asarray(adata.obs[x_key], dtype=int)
        y = np.asarray(adata.obs[y_key], dtype=int)
    elif spatial_key in adata.obsm:
        xy = np.asarray(adata.obsm[spatial_key])
        if xy.shape[1] < 2:
            raise ValueError(f"adata.obsm['{spatial_key}'] must have at least 2 columns.")
        x = xy[:, 0].astype(int)
        y = xy[:, 1].astype(int)
    else:
        raise KeyError(
            f"Need pixel coordinates in adata.obs[['{x_key}','{y_key}']] or adata.obsm['{spatial_key}']."
        )

    if origin == "min":
        x0, y0 = int(x.min()), int(y.min())
    else:
        x0, y0 = 0, 0

    xg = x - x0
    yg = y - y0

    if shape is None:
        W = int(xg.max()) + 1
        H = int(yg.max()) + 1
    else:
        H, W = int(shape[0]), int(shape[1])
        if xg.max() >= W or yg.max() >= H:
            raise ValueError(
                f"Provided shape={shape} is too small for coordinates (need at least H>{yg.max()}, W>{xg.max()})."
            )

    flat = yg * W + xg
    return H, W, flat.astype(np.int64)


def msi_cap_hotspots(
    msicube: "MSICube", *, params: MSIHotspotCapParams = MSIHotspotCapParams()
) -> None:
    """
    Cap ion images at a specific quantile to eliminate pixel hotspots.

    In MSI, single pixels often show extreme intensities due to matrix 
    crystals or electronic noise. This function caps intensities at the 
    `q`-th quantile (default 99%), ensuring that visualizations are not 
    dominated by a few outlier pixels.

    
    """

    X = _ensure_dense_float(_get_matrix(msicube, params.layer_in), dtype=np.dtype(params.dtype))
    _, n_vars = X.shape

    Xo = X if params.layer_out == params.layer_in else X.copy()

    for start in range(0, n_vars, params.chunk_size):
        end = min(n_vars, start + params.chunk_size)
        block = Xo[:, start:end]
        caps = np.quantile(block, params.q, axis=0)
        np.minimum(block, caps[None, :], out=block)

    _set_matrix(msicube, params.layer_out, Xo)


def msi_threshold_quantile(
    msicube: "MSICube", *, params: MSIThresholdParams = MSIThresholdParams()
) -> None:
    """
    Threshold ion images at the per-variable quantile ``q``.

    This is a data-driven way to perform background subtraction. For each 
    ion image, a threshold is calculated based on its own intensity 
    distribution. Intensities below this limit are masked.
    """

    X = _ensure_dense_float(_get_matrix(msicube, params.layer_in), dtype=np.dtype(params.dtype))
    _, n_vars = X.shape

    Xo = X if params.layer_out == params.layer_in else X.copy()

    for start in range(0, n_vars, params.chunk_size):
        end = min(n_vars, start + params.chunk_size)
        block = Xo[:, start:end]
        thr = np.quantile(block, params.q, axis=0)
        mask = block < thr[None, :]
        if params.mode == "zero":
            block[mask] = 0.0
        else:
            block[mask] = np.nan

    _set_matrix(msicube, params.layer_out, Xo)


def msi_median_filter_2d(
    msicube: "MSICube", *, params: MSIMedianFilterParams = MSIMedianFilterParams()
) -> None:
    """
    Apply a 2D median filter to each ion image.

    This function reconstructs a 2D spatial grid from the flattened MSI data 
    and applies a median filter. It is highly effective at removing "salt-and-pepper" 
    noise while preserving the sharp edges of biological structures.

    Parameters
    ----------
    msicube : MSICube
        The data cube to filter.
    params : MSIMedianFilterParams
        Configuration including:
        * **size**: The kernel size (e.g., 3 for a 3x3 window).
        * **x_key / y_key**: Coordinates used to reconstruct the 2D image.

    Notes
    -----
    The function automatically handles non-rectangular ablation areas by 
    mapping pixels to a dense grid and using a `fill_value` for empty regions.

    
    """

    X = _ensure_dense_float(_get_matrix(msicube, params.layer_in), dtype=np.dtype(params.dtype))
    _, n_vars = X.shape

    H, W, flat = _infer_grid_index(
        msicube,
        x_key=params.x_key,
        y_key=params.y_key,
        spatial_key=params.spatial_key,
        shape=params.shape,
        origin=params.origin,
    )
    HW = H * W

    Xo = X if params.layer_out == params.layer_in else X.copy()

    for start in range(0, n_vars, params.chunk_size):
        end = min(n_vars, start + params.chunk_size)
        block = Xo[:, start:end]

        if params.nan_to_num_before:
            block = np.nan_to_num(
                block,
                nan=params.fill_value,
                posinf=params.fill_value,
                neginf=params.fill_value,
            )

        cube = np.full((end - start, HW), params.fill_value, dtype=block.dtype)
        cube[:, flat] = block.T
        cube = cube.reshape((end - start, H, W))

        cube_f = ndimage.median_filter(cube, size=(1, params.size, params.size), mode="nearest")

        block_out = cube_f.reshape((end - start, HW))[:, flat]
        Xo[:, start:end] = block_out.T

    _set_matrix(msicube, params.layer_out, Xo)
