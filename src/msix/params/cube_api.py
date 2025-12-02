# msix/core/spec/cube_api.py
"""
Public contract for the read-only Cube API.

A Cube is a light, Scanpy-like handle over a Zarr-backed MSI datacube built by
Phase 4/5, exposing:
  - Shapes & geometry: n_obs (pixels), n_vars (m/z), height, width
  - Lazy reads of /X and /var/mz
  - Obs/var tables and obsm["spatial"]
  - Ion image helpers
  - Subsetting that returns cheap views (same store, narrowed indexers)

Conventions
-----------
- All coordinates are 0-based pixel indices (x, y).
- /X is (n_obs, n_vars), row-major chunks are preferred.
- /var/mz is strictly increasing float64.
- Reads are non-blocking; nothing loads until you call a read method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Protocol

import numpy as np
import pandas as pd


__all__ = ["IonImageParams", "CubeLike"]


@dataclass(frozen=True)
class IonImageParams:
    """
    Parameters for ion image rendering from a column X[:, j].

    Attributes
    ----------
    fill_value : float
        Intensity to assign to pixels without data (e.g., missing rows).
    origin : {"upper","lower"}
        Matplotlib-compatible origin; does not affect data, only display.
    """

    fill_value: float = 0.0
    origin: Literal["upper", "lower"] = "upper"


class CubeLike(Protocol):
    """
    Read-only Cube contract.

    Properties
    ----------
    store_uri : str
    n_obs, n_vars : int
    height, width : int
    shape : tuple[int, int]              # (n_obs, n_vars)
    dtype : np.dtype                     # dtype of /X
    chunks : tuple[int, int]             # (obs_chunk, var_chunk), if available

    Tables & matrices
    -----------------
    var_mz() -> np.ndarray               # float64 (n_vars,)
    obs() -> pd.DataFrame                # /tables/obs
    var() -> pd.DataFrame                # /tables/var
    obsm(name: str) -> np.ndarray        # typically "spatial" -> float32 (n_obs, 2)

    Matrix reads
    ------------
    read_X(rows, cols) -> np.ndarray     # dense ndarray; prefer chunk-aligned slices for large reads

    Convenience
    -----------
    ion_image(var_index: int, *, params: IonImageParams = IonImageParams()) -> np.ndarray
        Returns float image (height, width) by scattering X[:, j] to (x,y).
    stack_images(var_indices: Sequence[int], *, params: IonImageParams = IonImageParams()) -> np.ndarray
        Returns (len(var_indices), height, width).

    Subsetting
    ----------
    subset(obs_idx: Optional[Sequence[int]] = None,
           var_idx: Optional[Sequence[int]] = None) -> "CubeLike"
        Returns a cheap view sharing the same store, narrowing indexers used by reads.

    Lifecycle
    ---------
    close() -> None
    """

    # Identity
    @property
    def store_uri(self) -> str: ...

    # Shapes & geometry
    @property
    def n_obs(self) -> int: ...
    @property
    def n_vars(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def chunks(self) -> Tuple[int, int]: ...

    # Tables & matrices
    def var_mz(self) -> np.ndarray: ...
    def obs(self) -> pd.DataFrame: ...
    def var(self) -> pd.DataFrame: ...
    def obsm(self, name: str) -> np.ndarray: ...

    # Matrix reads
    def read_X(self, rows: Sequence[int], cols: Sequence[int]) -> np.ndarray: ...

    # Convenience
    def ion_image(
        self, var_index: int, *, params: IonImageParams = IonImageParams()
    ) -> np.ndarray: ...
    def stack_images(
        self, var_indices: Sequence[int], *, params: IonImageParams = IonImageParams()
    ) -> np.ndarray: ...

    # Subsetting
    def subset(
        self,
        obs_idx: Optional[Sequence[int]] = None,
        var_idx: Optional[Sequence[int]] = None,
    ) -> "CubeLike": ...

    # Lifecycle
    def close(self) -> None: ...
