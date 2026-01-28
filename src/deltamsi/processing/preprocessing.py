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

if TYPE_CHECKING:  # pragma: no cover
    from deltamsi.core.msicube import MSICube


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
    msicube: "MSICube",
    *,
    q: float = 0.99,
    layer: Optional[str] = None,
    output_layer: Optional[str] = None,
    chunk_size: int = 256,
    dtype: Union[str, np.dtype] = "float32",
) -> None:
    """
    Cap ion images at a specific quantile to eliminate pixel hotspots.

    In MSI, single pixels often show extreme intensities due to matrix 
    crystals or electronic noise. This function caps intensities at the 
    `q`-th quantile (default 99%), ensuring that visualizations are not 
    dominated by a few outlier pixels.

    Parameters
    ----------
    q : float, default 0.99
        Quantile threshold (0.0 to 1.0). Intensities above this value
        will be clipped to the quantile value.
    layer : str, optional
        The AnnData layer to process. If None, uses ``adata.X``.
    output_layer : str, optional
        The layer to store results. If None, overwrites the input.
    chunk_size : int, default 256
        Number of ion images to process simultaneously to optimize memory.
    dtype : str or np.dtype, default "float32"
        The numerical precision for processing.
    """

    X = _ensure_dense_float(_get_matrix(msicube, layer), dtype=np.dtype(dtype))
    _, n_vars = X.shape

    Xo = X.copy()

    for start in range(0, n_vars, chunk_size):
        end = min(n_vars, start + chunk_size)
        block = Xo[:, start:end]
        caps = np.quantile(block, q, axis=0)
        np.minimum(block, caps[None, :], out=block)

    _set_matrix(msicube, output_layer, Xo)


def msi_threshold_quantile(
    msicube: "MSICube",
    *,
    q: float = 0.5,
    mode: Literal["zero", "nan"] = "zero",
    layer: Optional[str] = None,
    output_layer: Optional[str] = None,
    chunk_size: int = 256,
    dtype: Union[str, np.dtype] = "float32",
) -> None:
    """
    Threshold ion images at the per-variable quantile ``q``.

    This is a data-driven way to perform background subtraction. For each 
    ion image, a threshold is calculated based on its own intensity 
    distribution. Intensities below this limit are masked.

    Parameters
    ----------
    q : float, default 0.5
        The quantile below which intensities are removed.
    mode : {"zero", "nan"}, default "zero"
        Whether to set values below threshold to 0.0 or NaN.
    layer : str, optional
        Input layer in AnnData. If None, uses ``adata.X``.
    output_layer : str, optional
        Output layer in AnnData. If None, overwrites the input.
    chunk_size : int, default 256
        Number of variables processed in a single block.
    dtype : str or np.dtype, default "float32"
        The numerical precision for processing.
    """

    X = _ensure_dense_float(_get_matrix(msicube, layer), dtype=np.dtype(dtype))
    _, n_vars = X.shape

    Xo = X.copy()

    for start in range(0, n_vars, chunk_size):
        end = min(n_vars, start + chunk_size)
        block = Xo[:, start:end]
        thr = np.quantile(block, q, axis=0)
        mask = block < thr[None, :]
        if mode == "zero":
            block[mask] = 0.0
        else:
            block[mask] = np.nan

    _set_matrix(msicube, output_layer, Xo)


def msi_median_filter_2d(
    msicube: "MSICube",
    *,
    size: int = 3,
    layer: Optional[str] = None,
    output_layer: Optional[str] = None,
    dtype: Union[str, np.dtype] = "float32",
    x_key: str = "x",
    y_key: str = "y",
    spatial_key: str = "spatial",
    shape: Optional[Tuple[int, int]] = None,
    origin: Literal["min", "zero"] = "min",
    fill_value: float = 0.0,
    nan_to_num_before: bool = True,
    chunk_size: int = 64,
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
    size : int, default 3
        The side length of the square median window (e.g., 3 for 3x3).
    layer : str, optional
        Source layer name. If None, uses ``adata.X``.
    output_layer : str, optional
        Destination layer name. If None, overwrites the input.
    dtype : str or np.dtype, default "float32"
        The numerical precision for processing.
    x_key, y_key : str
        Metadata keys for pixel coordinates.
    spatial_key : str
        Key for the spatial coordinate matrix in ``obsm``.
    shape : tuple, optional
        Explicit (Height, Width) for the spatial grid.
    origin : {"min", "zero"}
        Whether to offset coordinates to (0,0).
    fill_value : float, default 0.0
        Value used for pixels with no data in a non-rectangular grid.
    nan_to_num_before : bool, default True
        If True, replaces NaN values with `fill_value` before filtering.
    chunk_size : int, default 64
        Number of images processed per block. Keep low for large spatial grids.

    Notes
    -----
    The function automatically handles non-rectangular ablation areas by 
    mapping pixels to a dense grid and using a `fill_value` for empty regions.

    
    """

    X = _ensure_dense_float(_get_matrix(msicube, layer), dtype=np.dtype(dtype))
    _, n_vars = X.shape

    H, W, flat = _infer_grid_index(
        msicube,
        x_key=x_key,
        y_key=y_key,
        spatial_key=spatial_key,
        shape=shape,
        origin=origin,
    )
    HW = H * W

    Xo = X.copy()

    for start in range(0, n_vars, chunk_size):
        end = min(n_vars, start + chunk_size)
        block = Xo[:, start:end]

        if nan_to_num_before:
            block = np.nan_to_num(
                block,
                nan=fill_value,
                posinf=fill_value,
                neginf=fill_value,
            )

        cube = np.full((end - start, HW), fill_value, dtype=block.dtype)
        cube[:, flat] = block.T
        cube = cube.reshape((end - start, H, W))

        cube_f = ndimage.median_filter(cube, size=(1, size, size), mode="nearest")

        block_out = cube_f.reshape((end - start, HW))[:, flat]
        Xo[:, start:end] = block_out.T

    _set_matrix(msicube, output_layer, Xo)
