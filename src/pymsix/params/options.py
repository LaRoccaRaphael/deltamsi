"""
MSI Processing Option Specifications
====================================

This module defines frozen dataclasses for configuring mean spectrum calculation, 
peak picking, binning policies, and mass axis construction. 

Design Goals
------------

* **Explicit & Minimal**: JSON-safe structures that can be easily serialized 
  (using ``asdict``) for metadata provenance and reproducibility.
* **Immutable**: Classes use ``frozen=True`` to ensure that once a processing 
  configuration is created, it cannot be modified during the run.
* **Validation**: Integrated ``validate()`` methods provide human-friendly 
  error messages for invalid parameter combinations.
* **Separation of Concerns**: Decouples the *specification* of processing 
  steps from their *implementation*.

Functional Categories
---------------------

* **Binning & Aggregation**: How to discretize and merge m/z data (e.g., 
  :class:`MeanSpectrumOptions`).
* **Feature Detection**: Parameters for finding peaks and building intensity 
  matrices (e.g., :class:`PeakPickingOptions`, :class:`PeakMatrixOptions`).
* **Post-Processing**: Advanced options for recalibration, mass clustering, 
  and Kendrick visualization.

Notes
-----
The implementation of the logic defined by these options resides in 
dedicated submodules:

    * ``pymsix.core.msicube``
    * ``pymsix.processing.combine_mean_spectra``
    * ``pymsix.processing.mean_spectrum``
    * ``pymsix.processing.peak_picking``
    * ``pymsix.processing.recalibration``

Examples
--------
>>> from pymsix.params import MeanSpectrumOptions, PeakPickingOptions
>>> # Create a pipeline configuration
>>> ms_opts = MeanSpectrumOptions(mode="centroid", binning_p=0.001)
>>> pp_opts = PeakPickingOptions(topn=1000, distance_ppm=10.0)
>>> # Validate before use
>>> ms_opts.validate()
>>> pp_opts.validate()
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, Tuple, Union, Callable, List

import numpy as np

__all__ = [
    "MeanSpectrumOptions",
    "GlobalMeanSpectrumOptions",
    "PeakPickingOptions",
    "PeakMatrixOptions",
    "RecalibrationOptions",
    "MassClusteringOptions",
    "KendrickPlotOptions",
    "RecalParams",
    "CosineColocParams",
    "RankIonsMSIParams",
]


@dataclass(frozen=True)
class MeanSpectrumOptions:
    """
    Options for computing the mean spectrum of an MSI cube.

    These parameters define the common m/z axis and, optionally, the smoothing
    or smearing required for centroid data.

    Parameters
    ----------
    mode : {"profile", "centroid"}
        The aggregation method. Use "profile" for simple binning or "centroid"
        for smoothed binning using Gaussian kernels.
    min_mz : float, default 0.0
        Minimum m/z value for the common axis.
    max_mz : float, default 2000.0
        Maximum m/z value for the common axis.
    binning_p : float, default 0.001
        Bin width in Da (e.g., 0.0001).
    tolerance_da : float, optional
        Constant Gaussian width in Da (:math:`1\sigma`) for centroid mode.
        Ignored if `mass_accuracy_ppm` is provided.
    mass_accuracy_ppm : float, optional, default 3.0
        Instrument accuracy in ppm (:math:`1\sigma`). If provided, this 
        overrides `tolerance_da`.
    n_sigma : float, default 3.0
        Radius of the Gaussian window in :math:`\sigma` units 
        (e.g., 3.0 for a :math:`\pm 3\sigma` window).

    Examples
    --------
    >>> from pymsix.params import MeanSpectrumOptions
    >>> opts = MeanSpectrumOptions(mode="centroid", binning_p=0.005, mass_accuracy_ppm=5.0)
    >>> opts.validate()
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

        Raises
        ------
        ValueError
            If any option combination is invalid.
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

    Parameters
    ----------
    binning_p : float, default 0.0001
        Bin width in Da for the common m/z axis during combination.
    use_intersection : bool, default True
        If True, the common axis is built only on the overlapping m/z range.
    tic_normalize : bool, default True
        If True, each mean spectrum is TIC-normalized before averaging.
    compress_axis : bool, default False
        If True, drop bins where the final mean intensity is zero.

    Examples
    --------
    >>> from pymsix.params import GlobalMeanSpectrumOptions
    >>> opts = GlobalMeanSpectrumOptions(binning_p=0.0005, tic_normalize=False)
    >>> opts.validate()
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
    """
    Options for peak picking on a spectrum.

    Parameters
    ----------
    topn : int, default 10000
        Maximum number of peaks to retain.
    binning_p : float, default 1e-4
        Bin width in Da used during the peak finding process.
    distance_da : float, optional
        Minimum distance between peaks in Daltons.
    distance_ppm : float, optional
        Minimum distance between peaks in ppm.

    Examples
    --------
    >>> from pymsix.params import PeakPickingOptions
    >>> opts = PeakPickingOptions(topn=500, distance_ppm=10.0)
    >>> opts.validate()
    """
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
    """
    Options for extracting the peak intensity matrix (X).

    Parameters
    ----------
    tol_da : float, optional
        Matching tolerance in Daltons.
    tol_ppm : float, optional
        Matching tolerance in ppm.

    Examples
    --------
    >>> from pymsix.params import PeakMatrixOptions
    >>> opts = PeakMatrixOptions(tol_ppm=5.0)
    >>> opts.validate()
    """
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
    Options for performing MSI recalibration using RANSAC.

    Parameters
    ----------
    tol_da : float, default 0.03
        Dalton tolerance for identifying calibration hits.
    tol_ppm : float, optional
        Matching tolerance in ppm. Overrides `tol_da` if provided.
    kde_bw_da : float, default 0.002
        Bandwidth for the Kernel Density Estimation (KDE) function.
    roi_halfwidth_da : float, default 0.02
        ROI half-width around the error mode in Da.
    n_peaks : int, default 1000
        Number of top intense peaks per spectrum used for hit search.
    min_hits_for_fit : int, default 20
        Minimum number of hits needed to fit the RANSAC model.

    Examples
    --------
    >>> from pymsix.params import RecalibrationOptions
    >>> opts = RecalibrationOptions(tol_ppm=20.0, n_peaks=500)
    >>> opts.validate()
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
    Options for clustering m/z values based on matching or colocalization.

    This class configures the construction of a graph where nodes are m/z peaks 
    and edges represent relationships (mass shifts or spatial similarity). 
    Community detection (Leiden/Louvain) is then applied to find clusters.

    Parameters
    ----------
    method : {"candidates", "colocalization"}, default "candidates"
        Clustering strategy. "candidates" uses theoretical mass shifts (e.g., isotopes, 
        adducts). "colocalization" uses spatial correlation between ion images.
    tol_da : float, default 0.005
        Absolute tolerance in Daltons for matching mass differences between peaks.
    tol_ppm : float, optional
        Relative tolerance in ppm for matching mass differences. If set, it 
        usually overrides or complements `tol_da`.
    edge_max_delta_m : float, optional
        Maximum mass difference (in Da) allowed to create an edge between two peaks.
    delta_col : str, default "delta_da"
        Name of the column containing the mass difference values in the input data.
    score_col : str, default "score"
        Name of the column used to weight edges (e.g., matching score or correlation).
    label_col : str, optional, default "label"
        Column name for annotation labels used during the clustering process.
    output_col : str, default "mass_cluster"
        Column name in ``adata.var`` where clustering labels are stored.
    resolution : float, default 1.0
        Resolution parameter for the Louvain/Leiden algorithm. Higher values 
        lead to more (smaller) clusters.
    weight_transform : str or callable, default "inv1p"
        Transformation to apply to `score_col` to compute edge weights. 
        "inv1p" compute :math:`1 / (1 + x)`.
    weight_kwargs : dict
        Additional keyword arguments passed to the `weight_transform` function.
    knn_k : int, optional
        If provided, prunes the graph to keep only the top `k` neighbors per node.
    knn_mode : {"union", "mutual"}, default "union"
        Logic for k-NN pruning. "union" keeps an edge if either node is in 
        the other's k-NN; "mutual" requires both.
    coloc_varp_key : str, optional, default "ion_cosine"
        Key in the `varp` (variable properties) used for colocalization 
        similarity (e.g., Cosine or Pearson).
    edge_max_delta_cosine : float, optional
        Minimum similarity threshold for creating an edge in colocalization mode.
    return_graph : bool, default False
        If True, the underlying NetworkX/igraph object is returned with the results.

    Examples
    --------
    >>> from pymsix.params import MassClusteringOptions
    >>> # Configure colocalization clustering with k-NN pruning
    >>> opts = MassClusteringOptions(
    ...     method="colocalization", 
    ...     knn_k=5, 
    ...     resolution=0.8
    ... )
    >>> opts.validate()
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
    output_col: str = "mass_cluster"

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
        if self.output_col is None:
            self.output_col = "mass_cluster"
        if not self.output_col:
            raise ValueError("output_col must be a non-empty string.")

    def get_tol_param(self) -> Union[float, Tuple[str, float]]:
        """Returns the format expected by the 'tol' argument."""
        if self.tol_ppm is not None:
            return ("ppm", self.tol_ppm)
        return ("da", self.tol_da)


@dataclass
class KendrickPlotOptions:
    """
    Configuration parameters for generating Kendrick Mass Plots.

    The Kendrick Mass Defect (KMD) is used to identify homologous series 
    (e.g., :math:`CH_2` chains). This class controls both the calculation 
    and the aesthetic rendering of the plots.

    Parameters
    ----------
    base : str, float or tuple, default "CH2"
        The formula (e.g., "CH2", "H2O") or exact mass used as the Kendrick base.
    mass_col : str, default "mz"
        The column in the data containing the m/z values to transform.
    label_col : str, optional
        Column in ``adata.var`` used to color the family/label panel. If None,
        falls back to the default lookup (``family`` then ``label``).
    kendrick_varm_key : str, optional
        Key to store/retrieve the calculated Kendrick values in the object's metadata.
    x_axis : {"kendrick_mass", "m_over_z"}, default "kendrick_mass"
        Values to display on the horizontal axis.
    kmd_mode : {"fraction", "defect"}, default "fraction"
        Type of KMD to plot. "fraction" is the classic Kendrick mass defect; 
        "defect" refers to the remainder after rounding.
    point_size : float, default 24.0
        Size of the markers in the scatter plot.
    alpha : float, default 0.9
        Transparency of the points (0 to 1).
    hgrid_step : float, default 1.0
        The interval between horizontal grid lines.
    jitter : float, default 0.0
        Amount of random horizontal noise added to points to prevent overlap 
        in dense series.
    annotate : bool, default False
        If True, adds text labels to specific points in the plot.
    max_ann_per_group : int, default 0
        Maximum number of annotations allowed per cluster/group.
    top_k_clusters : int, optional, default 20
        Only display the top `k` largest clusters. Set to None for all.
    selected_clusters : list of int, optional
        Explicit list of cluster IDs to display.
    include_minus1_in_top : bool, default True
        Whether to include the unassigned cluster (ID -1) when calculating the top `k`.
    min_cluster_size : int, default 1
        Clusters smaller than this value will be hidden.
    two_panels : bool, default True
        If True, splits the plot into two panels (e.g., All ions vs. Clustered ions).
    figsize : tuple of float, default (9.0, 4.5)
        Width and height of the figure in inches.
    dpi : int, default 140
        Resolution of the figure (Dots Per Inch).

    Examples
    --------
    >>> from pymsix.params import KendrickPlotOptions
    >>> # Plot using H2O as base for lipid analysis
    >>> opts = KendrickPlotOptions(base="H2O", point_size=40, annotate=True)
    >>> opts.validate()
    """
    __module__ = "pymsix.params"

    base: Union[str, float, Tuple[float, float]] = "CH2"
    mass_col: str = "mz"
    label_col: Optional[str] = None
    kendrick_varm_key: Optional[str] = None


    x_axis: str = "kendrick_mass"
    kmd_mode: str = "fraction"
    point_size: float = 24.0
    alpha: float = 0.9
    hgrid_step: float = (
        1.0
    )
    jitter: float = 0.0

    annotate: bool = False
    max_ann_per_group: int = 0 

    top_k_clusters: Optional[int] = 20
    selected_clusters: Optional[List[int]] = (
        None
    )
    include_minus1_in_top: bool = (
        True
    )
    min_cluster_size: int = 1

    two_panels: bool = True
    figsize: Tuple[float, float] = (9.0, 4.5)
    dpi: int = 140

    def validate(self) -> None:
        """Validates the coherence of the options."""
        if self.x_axis not in ["kendrick_mass", "m_over_z"]:
            raise ValueError(f"x_axis invalide: {self.x_axis}")
        if self.kmd_mode not in ["fraction", "defect"]:
            raise ValueError(f"kmd_mode invalide: {self.kmd_mode}")
        if self.top_k_clusters is not None and self.top_k_clusters < 1:
            self.top_k_clusters = None


@dataclass(frozen=True)
class RecalParams:
    """
    Parameter container for recalibration logic.

    Attributes
    ----------
    tol_da : float
        Search tolerance in Daltons.
    tol_ppm : float, optional
        Search tolerance in ppm. If set, it overrides `tol_da`.
    kde_bw_da : float
        Bandwidth for the KDE (smoothing factor).
    roi_halfwidth_da : float
        The window size around the detected mode to keep hits for RANSAC.
    n_peaks : int
        Number of highest-intensity peaks to consider per pixel.
    min_hits_for_fit : int
        Minimum number of database matches required to attempt a fit.
    """

    __module__ = "pymsix.params"

    # Matching tolerance (choose one)
    tol_da: float = 0.03  # used if tol_ppm is None
    tol_ppm: Optional[float] = None  # if set, per-peak tol_da = mz * tol_ppm * 1e-6

    # KDE / ROI (in Da)
    kde_bw_da: float = 0.002
    roi_halfwidth_da: float = 0.02
    kde_grid_step_da: float = 1e-4

    # Peak selection + fit
    n_peaks: int = 1000
    min_hits_for_fit: int = 20
    ransac_max_trials: int = 300
    ransac_min_samples: int = 10


@dataclass
class CosineColocParams:
    """
    Parameters controlling cosine-based ion colocalization computation.

    Attributes
    ----------
    layer : str, optional
        Name of the ``adata.layers`` entry to use. If None, uses ``adata.X``.
    dtype : Union[np.dtype, str], default "float32"
        Numerical precision for the computation and the resulting matrix.
    mode : {"dense", "topk_sparse"}, default "topk_sparse"
        Computation and storage strategy. Use ``"topk_sparse"`` for large
        datasets to avoid memory overflow.
    topk : int, default 50
        Only keep the top K most similar ions for each variable in
        sparse mode.
    min_sim : float, default 0.2
        Similarity threshold. Values below this are treated as zero in
        sparse mode.
    chunk_size : int, default 256
        Number of variables to process at once during block-wise sparse
        computation.
    symmetrize : bool, default True
        Ensure the output matrix is symmetric ($S_{ij} = S_{ji}$).
    include_self : bool, default False
        Whether to keep the diagonal (self-similarity of 1.0) in the matrix.
    store_varp_key : str, optional, default "ion_cosine"
        Key used to store the resulting matrix in ``adata.varp``.
    """

    __module__ = "pymsix.params"

    layer: Optional[str] = None
    dtype: Union[np.dtype, str] = "float32"
    mode: Literal["dense", "topk_sparse"] = "topk_sparse"
    topk: int = 50
    min_sim: float = 0.2
    chunk_size: int = 256
    symmetrize: bool = True
    include_self: bool = False
    store_varp_key: Optional[str] = "ion_cosine"


@dataclass
class RankIonsMSIParams:
    """
    Parameters for the ranking of ions between groups.

    Attributes
    ----------
    condition_key : str, default "condition"
        Column in ``adata.obs`` containing the experimental groups/conditions.
    sample_key : str, default "sample"
        Column in ``adata.obs`` identifying individual biological replicates.
    group : str, default "treated"
        The name of the condition to test (the "numerator").
    reference : str, default "control"
        The name of the condition to use as a baseline (the "denominator").
    layer : str, optional
        The AnnData layer to use for intensity values. If None, uses ``adata.X``.
    detection_threshold : float, default 0.0
        Intensity value above which an ion is considered "detected".
    pseudocount : float, default 1e-9
        Constant added to denominators to avoid division by zero in log2FC.
    agg : {"mean", "median"}, default "mean"
        Method to summarize pixels into sample-level pseudobulk values.
    method : {"auto", "ttest", "wilcoxon"}, default "auto"
        Statistical test to perform when replicates are available.
    direction : {"up", "abs"}, default "up"
        "up" focuses on ions overexpressed in `group`. "abs" ranks by absolute fold change.
    n_top : int, default 200
        Number of top ions to return in the summary table.
    compute_auc : bool, default True
        Whether to calculate the Area Under the Curve (Receiver Operating Characteristic).
    block_bootstrap : bool, default False
        Whether to use spatial block bootstrapping to estimate confidence
        intervals for single-sample comparisons.
    block_size : int, default 25
        Side length of the square spatial blocks (in pixels) for bootstrapping.
    key_added : str, default "rank_ions_groups_msi"
        Key under which results are stored in ``adata.uns``.
    """

    __module__ = "pymsix.params"

    condition_key: str = "condition"
    sample_key: str = "sample"
    group: str = "treated"  # user-selected
    reference: str = "control"  # user-selected

    layer: Optional[str] = None

    # effect sizes
    detection_threshold: float = 0.0
    pseudocount: float = 1e-9
    agg: Literal["mean", "median"] = "mean"  # for pseudobulk per sample; median requires dense

    # statistics (only when replicated)
    method: Literal["auto", "ttest", "wilcoxon"] = "auto"

    # ranking
    direction: Literal["up", "abs"] = "up"  # 'up' ranks overexpressed in group; 'abs' ranks by |logFC|
    n_top: int = 200

    # speed controls
    compute_auc: bool = True
    auc_on: Literal["auto", "samples", "pixels"] = "auto"
    auc_max_ions: int = 3000  # compute AUC only for top K by ranking score
    auc_max_pixels_per_group: int = 50000  # subsample pixels for AUC when using pixel-level

    # single-sample uncertainty (optional)
    block_bootstrap: bool = False
    block_size: int = 25
    n_boot: int = 200
    ci_alpha: float = 0.05
    ci_max_ions: int = 1000  # compute CI only for top K ions
    x_key: str = "x"
    y_key: str = "y"
    spatial_key: str = "spatial"

    # output
    key_added: str = "rank_ions_groups_msi"
    random_state: int = 0
