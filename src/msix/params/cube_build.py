# msix/core/spec/cube_build.py
"""
Contracts for building /X (datacube) against an existing /var/mz axis
using the *targeted_window* projector.

This phase performs **no peak picking and no resampling**.
Each spectrum is projected onto /var/mz with a tolerance around each
experimental peak, in either ppm or Da, and aggregated per bin using
'sum' or 'max'.

Fields are intentionally minimal and storage-oriented (dtype, chunks, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from ..utils.validate import _assert  # simple boolean guard -> raises ValueError


__all__ = ["CubeBuildParams", "validate_cube_build"]


@dataclass(frozen=True)
class CubeBuildParams:
    """
    Parameters controlling /X construction via the *targeted_window* projector.

    Semantics
    ---------
    For each experimental peak (m, I):
      - if target_tolerance_mode == "ppm":
          half_width_Da = m * target_tolerance_value * 1e-6
        else ("da"):
          half_width_Da = target_tolerance_value

      - bins in /var/mz whose centers fall within [m - half_width_Da, m + half_width_Da]
        receive the peak's intensity according to `target_aggregator`:
          * 'sum' : add I to all covered bins
          * 'max' : set bin := max(bin, I)

    Notes
    -----
    - Choose ppm unless your targets live in a narrow m/z band.
    - With 'sum', wide tolerances can spread one peak into multiple bins.
      Keep tolerance tight (e.g., 3–10 ppm) for crisp assignment.
    """

    # Projector is fixed in this phase
    projector: Literal["targeted_window"] = "targeted_window"

    # Tolerance policy
    target_tolerance_mode: Literal["ppm", "da"] = "ppm"
    target_tolerance_value: float = 5.0  # ppm or Da depending on mode (>0)

    # Aggregation inside the matched window
    target_aggregator: Literal["sum", "max"] = "sum"

    # Per-spectrum preprocessing and post-projection transform
    normalize: Literal["none", "tic"] = "none"  # pre-projection
    transform: Literal["none", "sqrt", "log1p"] = "none"  # post-projection

    # Storage policy for /X
    dtype: Literal["float32", "float64"] = "float32"
    fill_value: float = 0.0
    chunks_obs: Optional[int] = None  # if None, builder chooses a heuristic
    chunks_var: Optional[int] = None
    compressor: Optional[str] = "zstd"  # e.g., 'zstd', 'blosc:zstd', or None
    write_order: Literal["C", "F"] = "C"

    # Tables
    store_obs_table: bool = True
    store_obsm_spatial: bool = True


def validate_cube_build(p: CubeBuildParams) -> None:
    """Validate a CubeBuildParams instance; raises ValueError on problems."""
    _assert(
        p.projector == "targeted_window",
        "Only 'targeted_window' is supported in Phase 5.",
    )

    _assert(
        p.target_tolerance_mode in {"ppm", "da"},
        "target_tolerance_mode must be 'ppm' or 'da'.",
    )
    _assert(p.target_tolerance_value > 0, "target_tolerance_value must be > 0.")
    _assert(
        p.target_aggregator in {"sum", "max"},
        "target_aggregator must be 'sum' or 'max'.",
    )

    _assert(p.normalize in {"none", "tic"}, "normalize must be 'none' or 'tic'.")
    _assert(
        p.transform in {"none", "sqrt", "log1p"},
        "transform must be 'none', 'sqrt', or 'log1p'.",
    )

    _assert(p.dtype in {"float32", "float64"}, "dtype must be 'float32' or 'float64'.")
    _assert(p.write_order in {"C", "F"}, "write_order must be 'C' or 'F'.")

    if p.chunks_obs is not None:
        _assert(p.chunks_obs > 0, "chunks_obs must be a positive integer or None.")
    if p.chunks_var is not None:
        _assert(p.chunks_var > 0, "chunks_var must be a positive integer or None.")
    if p.compressor is not None:
        _assert(
            isinstance(p.compressor, str) and len(p.compressor) > 0,
            "compressor must be a non-empty str or None.",
        )

    # fill_value can be any finite float; leave as-is
