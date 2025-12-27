"""Preprocessing utilities for :class:`~pymsix.core.msicube.MSICube`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy import ndimage

if TYPE_CHECKING:  # pragma: no cover
    from pymsix.core.msicube import MSICube


def _get_matrix(msicube: "MSICube", layer: Optional[str]):
    adata = msicube.adata
    if adata is None:
        raise ValueError("MSICube.adata is None. Run data extraction first.")

    if layer is None:
        return adata.X

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers")

    return adata.layers[layer]


def _set_matrix(msicube: "MSICube", layer: Optional[str], X):
    adata = msicube.adata
    if adata is None:
        raise ValueError("MSICube.adata is None. Run data extraction first.")

    if layer is None:
        adata.X = X
    else:
        adata.layers[layer] = X


def _ensure_dense_float(X, dtype=np.float32) -> np.ndarray:
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


@dataclass
class MSIHotspotCapParams:
    """Parameters for hotspot capping preprocessing."""

    q: float = 0.99
    layer_in: Optional[str] = None
    layer_out: Optional[str] = None
    chunk_size: int = 256
    dtype: Union[str, np.dtype] = "float32"


def msi_cap_hotspots(
    msicube: "MSICube", *, params: MSIHotspotCapParams = MSIHotspotCapParams()
) -> None:
    """Cap each ion image at its ``q``-quantile to remove hotspots."""

    X = _ensure_dense_float(_get_matrix(msicube, params.layer_in), dtype=np.dtype(params.dtype))
    _, n_vars = X.shape

    Xo = X if params.layer_out == params.layer_in else X.copy()

    for start in range(0, n_vars, params.chunk_size):
        end = min(n_vars, start + params.chunk_size)
        block = Xo[:, start:end]
        caps = np.quantile(block, params.q, axis=0)
        np.minimum(block, caps[None, :], out=block)

    _set_matrix(msicube, params.layer_out, Xo)


@dataclass
class MSIThresholdParams:
    """Parameters for per-ion quantile thresholding."""

    q: float = 0.5
    mode: Literal["zero", "nan"] = "zero"
    layer_in: Optional[str] = None
    layer_out: Optional[str] = None
    chunk_size: int = 256
    dtype: Union[str, np.dtype] = "float32"


def msi_threshold_quantile(
    msicube: "MSICube", *, params: MSIThresholdParams = MSIThresholdParams()
) -> None:
    """
    Threshold ion images at the per-variable quantile ``q``.

    Intensities below the threshold are set to ``0`` or ``NaN`` depending on
    ``params.mode``.
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


@dataclass
class MSIMedianFilterParams:
    """Parameters for applying a 2D median filter to ion images."""

    size: int = 3
    layer_in: Optional[str] = None
    layer_out: Optional[str] = None
    dtype: Union[str, np.dtype] = "float32"

    x_key: str = "x"
    y_key: str = "y"
    spatial_key: str = "spatial"
    shape: Optional[Tuple[int, int]] = None
    origin: Literal["min", "zero"] = "min"

    fill_value: float = 0.0
    nan_to_num_before: bool = True
    chunk_size: int = 64


def msi_median_filter_2d(
    msicube: "MSICube", *, params: MSIMedianFilterParams = MSIMedianFilterParams()
) -> None:
    """Apply a 2D median filter to each ion image using pixel coordinates."""

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

