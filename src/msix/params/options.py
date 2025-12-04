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

from dataclasses import dataclass, asdict
from typing import Any, Literal, Optional


__all__ = [
    "BinningParams",
    "MeanSpectrumOptions",
    "GlobalMeanSpectrumOptions",
    "CentroidingParams",
    "PeakPickParams",
    "AxisPolicy",
    "asdict_safe",
    "validate_binning",
    # "validate_mean"
    "validate_centroiding",
    "validate_peakpick",
    "validate_axis_policy",
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
    binning_p: float = 0.0001
    tolerance_da: Optional[float] = None
    mass_accuracy_ppm: Optional[float] = None
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
            if self.tolerance_da is None and self.mass_accuracy_ppm is None:
                raise ValueError(
                    "For centroid mode, provide either 'tolerance_da' (Da) "
                    "or 'mass_accuracy_ppm' (ppm)."
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
class BinningParams:
    """
    Binning strategy for profile spectra.

    mode
        "ppm": bin width is specified in parts-per-million at each m/z.
        "da" : bin width is specified in Daltons (uniform Δm).
    bin_width
        Positive bin width; interpreted according to `mode`.
    mz_min, mz_max
        Optional explicit span. If None, use probed span (with small padding).
    """

    mode: Literal["ppm", "da"] = "ppm"
    bin_width: float = 5.0
    mz_min: Optional[float] = None
    mz_max: Optional[float] = None


# @dataclass(frozen=True)
# class MeanSpecParams:
#     """
#     Parameters controlling how we form a mean spectrum across pixels.
#
#     normalize
#         "tic": divide each spectrum by its TIC before accumulation.
#         "none": raw intensities.
#     sample
#         If set, randomly subsample up to `sample` spectra for pilot steps (deterministic RNG).
#     agg
#         "mean" or "median_mean" (median across spectra, then mean smoothing).
#     winsor_pct
#         Optional winsorization percentage (0–5) applied per m/z bin before aggregation.
#     kde_sigma_ppm
#         For centroid inputs: Gaussian kernel σ (ppm) around each peak when accumulating.
#     kde_half_width_sigma
#         Truncate the Gaussian at ±k·σ to bound compute.
#     """
#
#     normalize: Literal["tic", "none"] = "tic"
#     sample: Optional[int] = None
#     agg: Literal["mean", "median_mean"] = "median_mean"
#     winsor_pct: float = 0.0
#     kde_sigma_ppm: float = 5.0
#     kde_half_width_sigma: float = 3.0


@dataclass(frozen=True)
class CentroidingParams:
    """
    On-the-fly centroiding for profile spectra (used in recalibration or when
    a centroid view is needed without writing a new imzML).

    smooth_window
        Savitzky–Golay window length (odd, >=3).
    smooth_poly
        Savitzky–Golay polynomial order (< smooth_window).
    baseline_lambda
        Asymmetric least-squares baseline strength (lambda > 0).
    baseline_p
        Asymmetry parameter p in (0,1); smaller favors positive residuals as peaks.
    min_prominence
        Minimum prominence passed to peak finder (non-negative).
    min_snr
        Minimum signal-to-noise (SNR) after baseline; non-negative.
    min_distance_ppm
        Minimum separation between peaks when de-duplicating (ppm).
    max_peaks
        Optional cap per spectrum (None disables).
    """

    smooth_window: int = 11
    smooth_poly: int = 3
    baseline_lambda: float = 1e5
    baseline_p: float = 1e-3
    min_prominence: float = 0.0
    min_snr: float = 3.0
    min_distance_ppm: float = 5.0
    max_peaks: Optional[int] = None


@dataclass(frozen=True)
class PeakPickParams:
    """
    Peak picking on the MEAN spectrum (after mandatory resampling).
    """

    # NEW: mandatory resampling precision (Δm = 10^{-resample_digits} Da)
    resample_digits: int = 3  # e.g., 3 -> 0.001 Da grid

    mode: Literal["topn", "preprocessed"] = "preprocessed"
    topn: int = 2000
    smooth_window: int = 11
    smooth_poly: int = 3
    baseline_lambda: float = 1e5
    baseline_p: float = 1e-3
    min_prominence: float = 0.0
    min_snr: float = 3.0
    merge_ppm: float = 5.0


@dataclass(frozen=True)
class AxisPolicy:
    """
    Policy for constructing the /var/mz axis.

    kind
        "peaklist": derive from picked peaks on the mean spectrum.
        "grid"    : construct directly from a grid spec (ppm or Da).
    grid_mode, bin_width, mz_min, mz_max
        Used only when kind="grid".
    """

    kind: Literal["peaklist", "grid"] = "peaklist"
    grid_mode: Literal["ppm", "da"] = "ppm"
    bin_width: float = 5.0
    mz_min: Optional[float] = None
    mz_max: Optional[float] = None


# ----------------------------- Helper utilities -----------------------------


def asdict_safe(obj: Any) -> dict[str, Any]:
    """JSON-safe serialization for dataclasses (shallow)."""
    try:
        return asdict(obj)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Object not serializable via asdict(): {e}") from e


# ------------------------------- Validators ---------------------------------


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_binning(o: BinningParams) -> None:
    _assert(o.mode in {"ppm", "da"}, "BinningParams.mode must be 'ppm' or 'da'")
    _assert(o.bin_width > 0, "BinningParams.bin_width must be > 0")
    if o.mz_min is not None and o.mz_max is not None:
        _assert(o.mz_max > o.mz_min, "BinningParams.mz_max must be > mz_min")


# def validate_mean(o: MeanSpecParams) -> None:
#     _assert(
#         o.normalize in {"tic", "none"},
#         "MeanSpecParams.normalize must be 'tic' or 'none'",
#     )
#     if o.sample is not None:
#         _assert(o.sample > 0, "MeanSpecParams.sample must be > 0")
#     _assert(
#         o.agg in {"mean", "median_mean"},
#         "MeanSpecParams.agg must be 'mean' or 'median_mean'",
#     )
#     _assert(0.0 <= o.winsor_pct <= 5.0, "MeanSpecParams.winsor_pct must be in [0, 5]")
#     _assert(o.kde_sigma_ppm > 0, "MeanSpecParams.kde_sigma_ppm must be > 0")
#     _assert(
#         o.kde_half_width_sigma >= 1.0,
#         "MeanSpecParams.kde_half_width_sigma must be >= 1",
#    )


def validate_centroiding(o: CentroidingParams) -> None:
    _assert(
        o.smooth_window >= 3 and o.smooth_window % 2 == 1,
        "CentroidingParams.smooth_window must be odd and >= 3",
    )
    _assert(
        0 <= o.smooth_poly < o.smooth_window,
        "CentroidingParams.smooth_poly must be >=0 and < smooth_window",
    )
    _assert(o.baseline_lambda > 0, "CentroidingParams.baseline_lambda must be > 0")
    _assert(0 < o.baseline_p < 1, "CentroidingParams.baseline_p must be in (0,1)")
    _assert(o.min_prominence >= 0, "CentroidingParams.min_prominence must be >= 0")
    _assert(o.min_snr >= 0, "CentroidingParams.min_snr must be >= 0")
    _assert(o.min_distance_ppm > 0, "CentroidingParams.min_distance_ppm must be > 0")
    if o.max_peaks is not None:
        _assert(o.max_peaks > 0, "CentroidingParams.max_peaks must be > 0 if set")


def validate_peakpick(o: PeakPickParams) -> None:
    _assert(
        1 <= o.resample_digits <= 6, "PeakPickParams.resample_digits must be in [1, 6]"
    )
    _assert(
        o.mode in {"topn", "preprocessed"},
        "PeakPickParams.mode must be 'topn' or 'preprocessed'",
    )
    if o.mode == "topn":
        _assert(o.topn > 0, "PeakPickParams.topn must be > 0")
    else:
        _assert(
            o.smooth_window >= 3 and o.smooth_window % 2 == 1,
            "PeakPickParams.smooth_window must be odd and >= 3",
        )
        _assert(
            0 <= o.smooth_poly < o.smooth_window,
            "PeakPickParams.smooth_poly must be >=0 and < smooth_window",
        )
        _assert(o.baseline_lambda > 0, "PeakPickParams.baseline_lambda must be > 0")
        _assert(0 < o.baseline_p < 1, "PeakPickParams.baseline_p must be in (0,1)")
        _assert(o.min_prominence >= 0, "PeakPickParams.min_prominence must be >= 0")
        _assert(o.min_snr >= 0, "PeakPickParams.min_snr must be >= 0")
    _assert(o.merge_ppm > 0, "PeakPickParams.merge_ppm must be > 0")


def validate_axis_policy(o: AxisPolicy) -> None:
    _assert(
        o.kind in {"peaklist", "grid"}, "AxisPolicy.kind must be 'peaklist' or 'grid'"
    )
    if o.kind == "grid":
        _assert(
            o.grid_mode in {"ppm", "da"}, "AxisPolicy.grid_mode must be 'ppm' or 'da'"
        )
        _assert(o.bin_width > 0, "AxisPolicy.bin_width must be > 0")
        if o.mz_min is not None and o.mz_max is not None:
            _assert(o.mz_max > o.mz_min, "AxisPolicy.mz_max must be > mz_min")
