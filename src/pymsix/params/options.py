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

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, Tuple, Union, Callable, List

__all__ = [
    "MeanSpectrumOptions",
    "GlobalMeanSpectrumOptions",
    "PeakPickingOptions",
    "PeakMatrixOptions",
    "RecalibrationOptions",
    "MassClusteringOptions",
    "KendrickPlotOptions",
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
    __module__ = "pymsix.params"

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
    __module__ = "pymsix.params"

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
    __module__ = "pymsix.params"

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
    __module__ = "pymsix.params"

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


@dataclass(frozen=True)
class RecalibrationOptions:
    """
    Options for performing mass spectrometry imaging (MSI) recalibration
    using a mass database and the RANSAC algorithm.

    Attributes (matching recalibration_core.RecalParams):
        tol_da: Dalton tolerance for identifying calibration hits (e.g., 0.03 Da).
        tol_ppm: Matching tolerance in ppm. If provided, overrides tol_da for per-peak tolerance calculation.
        kde_bw_da: Bandwidth for the Kernel Density Estimation (KDE) function (e.g., 0.002 Da).
        roi_halfwidth_da: ROI half-width around the error mode (e.g., 0.02 Da).
        n_peaks: Number of top intense peaks per spectrum to use for hit search (e.g., 1000).
        min_hits_for_fit: Minimum number of hits (after ROI) needed to fit the RANSAC model (e.g., 20).
    """
    __module__ = "pymsix.params"

    # Matching tolerance (one must be provided)
    tol_da: float = 0.03
    tol_ppm: Optional[float] = None

    # KDE / ROI (in Da)
    kde_bw_da: float = 0.002
    roi_halfwidth_da: float = 0.02

    # Peak selection + fit
    n_peaks: int = 1000
    min_hits_for_fit: int = 20

    def validate(self) -> None:
        """Validate the recalibration parameters."""
        has_da = self.tol_da is not None and self.tol_da > 0
        has_ppm = self.tol_ppm is not None and self.tol_ppm > 0

        # Check that at least one tolerance is valid
        if not (has_da or has_ppm):
            raise ValueError(
                "Provide at least one positive tolerance: 'tol_da' (Da) or 'tol_ppm' (ppm)."
            )

        if self.kde_bw_da <= 0:
            raise ValueError("kde_bw_da (KDE bandwidth) must be strictly positive.")
        if self.roi_halfwidth_da <= 0:
            raise ValueError(
                "roi_halfwidth_da (ROI half-width) must be strictly positive."
            )
        if self.n_peaks <= 0:
            raise ValueError("n_peaks (number of peaks) must be a positive integer.")
        if self.min_hits_for_fit <= 0:
            raise ValueError("min_hits_for_fit must be a positive integer.")


@dataclass(frozen=True)
class MassClusteringOptions:
    """
    Options for clustering m/z values.

    The ``method`` parameter selects between candidate-based clustering
    (``"candidates"``) and colocalization-based clustering (``"colocalization"``).
    """
    __module__ = "pymsix.params"

    method: Literal["candidates", "colocalization"] = "candidates"

    # Matching params (candidates mode)
    tol_da: float = 0.005
    tol_ppm: Optional[float] = None
    edge_max_delta_m: Optional[float] = None

    # Column mapping
    delta_col: str = "delta_da"
    score_col: str = "score"
    label_col: Optional[str] = "label"

    # Graph & Clustering
    resolution: float = 1.0
    weight_transform: Union[str, Callable[[float], float]] = "inv1p"
    weight_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Pruning (k-NN)
    knn_k: Optional[int] = None
    knn_mode: str = "union"

    # Colocalization mode options
    coloc_varp_key: Optional[str] = "ion_cosine"
    edge_max_delta_cosine: Optional[float] = None

    # Output options
    return_graph: bool = False

    def validate(self) -> None:
        if self.method not in {"candidates", "colocalization"}:
            raise ValueError("method must be 'candidates' or 'colocalization'.")
        if self.tol_da <= 0:
            raise ValueError("tol_da must be positive.")
        if self.tol_ppm is not None and self.tol_ppm <= 0:
            raise ValueError("tol_ppm must be positive.")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive.")
        if self.knn_k is not None and self.knn_k < 0:
            raise ValueError("knn_k cannot be negative.")
        if self.knn_mode not in ["union", "mutual"]:
            raise ValueError("knn_mode must be 'union' or 'mutual'.")
        if self.method == "colocalization" and self.coloc_varp_key is None:
            raise ValueError("coloc_varp_key must be provided for colocalization mode.")

    def get_tol_param(self) -> Union[float, Tuple[str, float]]:
        """Returns the format expected by the 'tol' argument."""
        if self.tol_ppm is not None:
            return ("ppm", self.tol_ppm)
        return ("da", self.tol_da)


@dataclass
class KendrickPlotOptions:
    """
    Paramètres de configuration pour la génération de diagrammes de Kendrick.
    """
    __module__ = "pymsix.params"

    # Paramètres de base Kendrick
    base: Union[str, float, Tuple[float, float]] = "CH2"
    mass_col: str = "mz"
    kendrick_varm_key: Optional[str] = None

    # Axes et Style
    x_axis: str = "kendrick_mass"  # 'kendrick_mass' ou 'm_over_z'
    kmd_mode: str = "fraction"  # 'fraction' ou 'defect'
    point_size: float = 24.0
    alpha: float = 0.9
    hgrid_step: float = (
        1.0  # Pas de la grille KMD (1.0 pour fraction, ~0.1 pour defect)
    )
    jitter: float = 0.0  # Bruit horizontal pour éviter les superpositions

    # Annotations
    annotate: bool = False
    max_ann_per_group: int = 0  # Nombre max de points annotés par groupe (index)

    # Filtres de Clusters
    top_k_clusters: Optional[int] = 20  # Garder seulement les K plus grands clusters
    selected_clusters: Optional[List[int]] = (
        None  # Liste explicite d'IDs de clusters (ex: [0, 1, 5])
    )
    include_minus1_in_top: bool = (
        True  # Inclure le cluster -1 (non-assignés) dans le top K
    )
    min_cluster_size: int = 1  # Taille minimale d'un cluster pour être affiché

    # Mise en page
    two_panels: bool = True  # Second panneau coloré par 'family' si disponible
    figsize: Tuple[float, float] = (9.0, 4.5)
    dpi: int = 140

    def validate(self) -> None:
        """Valide la cohérence des options de plot."""
        if self.x_axis not in ["kendrick_mass", "m_over_z"]:
            raise ValueError(f"x_axis invalide: {self.x_axis}")
        if self.kmd_mode not in ["fraction", "defect"]:
            raise ValueError(f"kmd_mode invalide: {self.kmd_mode}")
        if self.top_k_clusters is not None and self.top_k_clusters < 1:
            self.top_k_clusters = None
