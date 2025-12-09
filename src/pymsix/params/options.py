"""
Phase 4 option specs (frozen dataclasses) for mean spectrum, peak picking,
binning policy, and axis construction.

Design goals
------------
- Explicit, minimal, JSON-safe (use `asdict` for provenance).
- Immutable (frozen=True) so runs are reproducible once options are created.
- Light validation helpers (raise ValueError with human-friendly messages).
- Separate concerns:
  * BinningParams          — how to discretize m/z (ppm or Da) for profile data
  * MeanSpectrumOptions    — how to aggregate per-spectrum into a mean spectrum (and common axis)
  * CentroidingParams      — how to centroid profile spectra on-the-fly (optional)
  * PeakPickParams         — how to pick peaks on the mean spectrum
  * AxisPolicy             — how to form /var/mz (from peaks or a grid)

These structs specify *what* to do; implementation lives in:
- msix/core/spectrum/mean.py
- msix/core/spectrum/peaks.py
- msix/core/spectrum/axis.py
- msix/core/builders/build_var_axis.py
"""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

__all__ = [
    "MeanSpectrumOptions",
    "GlobalMeanSpectrumOptions",
    "PeakPickingOptions",
    "PeakMatrixOptions",
]


# ----------------------------- Option dataclasses -----------------------------


@dataclass(frozen=True)
class MeanSpectrumOptions:
    """
    Options for computing the mean spectrum of an MSI cube.

    These parameters define the common m/z axis and, optionally, the smoothing
    or smearing required for centroid data.

    Attributes:
        mode: The aggregation method: "profile" (simple binning) or "centroid" (smoothed binning).
        min_mz: Minimum m/z value for the common axis.
        max_mz: Maximum m/z value for the common axis.
        binning_p: Bin width in Da (e.g., 0.0001).
        tolerance_da: Constant Gaussian width in Da (1σ) for centroid mode, if mass_accuracy_ppm is None.
        mass_accuracy_ppm: Instrument accuracy in ppm (1σ). If provided, this overrides tolerance_da.
        n_sigma: Radius of the Gaussian window in σ units (e.g., 3.0 for ±3σ window).
    """

    mode: Literal["profile", "centroid"]
    min_mz: float = 0.0
    max_mz: float = 2000.0
    binning_p: float = 0.001
    tolerance_da: Optional[float] = None
    mass_accuracy_ppm: Optional[float] = 3.0
    n_sigma: float = 3.0

    def validate(self) -> None:
        """
        Validates the coherence of the mean spectrum options.

        Raises:
            ValueError: If any option combination is invalid.
        """
        if self.mode not in {"profile", "centroid"}:
            raise ValueError(
                f"Mode must be 'profile' or 'centroid', not '{self.mode}'."
            )

        if self.min_mz >= self.max_mz:
            raise ValueError("min_mz must be strictly less than max_mz.")

        if self.binning_p <= 0.0:
            raise ValueError("binning_p (bin width) must be positive.")

        if self.mode == "centroid":
            has_da = self.tolerance_da is not None and self.tolerance_da > 0
            has_ppm = self.mass_accuracy_ppm is not None and self.mass_accuracy_ppm > 0

            if not (has_da ^ has_ppm):  # XOR check
                raise ValueError(
                    "For centroid mode, provide exactly one of "
                    "'tolerance_da' (Da) or 'mass_accuracy_ppm' (ppm), and it must be positive."
                )

            if self.n_sigma <= 0.0:
                raise ValueError("n_sigma must be positive for centroid smoothing.")


@dataclass(frozen=True)
class GlobalMeanSpectrumOptions:
    """
    Options for combining multiple mean spectra into a single global spectrum.

    Attributes:
        binning_p: Bin width in Da for the common m/z axis during combination.
        use_intersection: If True, the common axis is built only on the overlapping m/z range.
        tic_normalize: If True, each mean spectrum is TIC-normalized before averaging.
        compress_axis: If True, drop bins where the final mean intensity is zero.
    """

    binning_p: float = 0.0001
    use_intersection: bool = True
    tic_normalize: bool = True
    compress_axis: bool = False

    def validate(self) -> None:
        if self.binning_p <= 0:
            raise ValueError("binning_p must be strictly positive.")


@dataclass(frozen=True)
class PeakPickingOptions:
    """Options for peak picking on a spectrum."""

    topn: int = 10000
    binning_p: float = 1e-4
    distance_da: Optional[float] = None
    distance_ppm: Optional[float] = None

    def validate(self) -> None:
        """Validate the peak picking parameters."""
        if self.topn <= 0:
            raise ValueError("topn must be a positive integer.")
        if self.binning_p <= 0:
            raise ValueError("binning_p must be strictly positive.")

        has_da = self.distance_da is not None and self.distance_da > 0
        has_ppm = self.distance_ppm is not None and self.distance_ppm > 0

        if has_da and has_ppm:
            raise ValueError(
                "Provide exactly one of 'distance_da' (Da) or 'distance_ppm' (ppm), not both."
            )

        if self.distance_da is not None and self.distance_da < 0:
            raise ValueError("distance_da must be non-negative.")
        if self.distance_ppm is not None and self.distance_ppm < 0:
            raise ValueError("distance_ppm must be non-negative.")

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the options."""
        return {
            "topn": self.topn,
            "binning_p": self.binning_p,
            "distance_da": self.distance_da,
            "distance_ppm": self.distance_ppm,
        }


@dataclass(frozen=True)
class PeakMatrixOptions:
    """Options for extracting the peak intensity matrix (X)."""

    tol_da: Optional[float] = None
    tol_ppm: Optional[float] = None

    def validate(self) -> None:
        """Validate peak matrix extraction parameters."""
        has_da = self.tol_da is not None and self.tol_da > 0
        has_ppm = self.tol_ppm is not None and self.tol_ppm > 0

        if not (has_da ^ has_ppm):  # XOR check
            raise ValueError(
                "For peak matrix extraction, provide exactly one of "
                "'tol_da' (Da) or 'tol_ppm' (ppm), and it must be positive."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tol_da": self.tol_da,
            "tol_ppm": self.tol_ppm,
        }
