"""
MSICube Framework for Mass Spectrometry Imaging (MSI)
=====================================================

The `MSICube` class is the central orchestrator for the `deltamsi` library. It 
combines raw `imzML` data management with the `AnnData` structure to provide 
a comprehensive analysis environment for spatial metabolomics and proteomics.

Key Capabilities:
    * **Data Integration**: Connects raw mass spectrometry files to a structured 
        `AnnData` backend.
    * **Preprocessing**: Normalization, log-transformation, and intensity 
        clipping/masking.
    * **Spectral Processing**: Generation of mean spectra (individual and global), 
        peak picking, and intensity matrix extraction.
    * **Spatial Analysis**: Cosine colocalization and variable aggregation 
        by cluster or annotation labels.
    * **Persistence**: Robust loading and saving of the analysis state 
        using `h5ad` or `zarr` formats.



The framework leverages the standard `AnnData` slots:
    * **X**: The intensity matrix (n_pixels x n_m/z).
    * **obs**: Metadata for pixels (coordinates, sample ID, condition).
    * **var**: Metadata for features (m/z value, chemical annotation).
    * **uns**: Unstructured metadata (processing options, mean spectra).
    * **obsm**: Multidimensional pixel annotations (spatial coordinates).

Classes
-------
Logger
    Internal utility for standardized console output.
MSICube
    Primary interface for MSI data analysis and pipeline management.

"""

from concurrent.futures import ProcessPoolExecutor
import json
import os
import re
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Optional, Dict, Any, List, Literal, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from pyimzml.ImzMLParser import ImzMLParser

from deltamsi.plotting.ion_images import plot_ion_images
from deltamsi.plotting.spectrum import plot_mean_spectrum_windows
from deltamsi.processing.mean_spectrum import compute_mean_spectrum
from deltamsi.processing.combine_mean_spectra import combine_mean_spectra, Spectrum
from deltamsi.processing.peak_picking import peak_picking, extract_peak_matrix
from deltamsi.processing.aggregation import aggregate_vars_by_label, Agg
from deltamsi.processing.normalization import (
    HighAction,
    LowAction,
    ScaleMode,
    ScaleStats,
    clip_or_mask_intensities as clip_or_mask_intensities_processing,
    log1p_intensity as log1p_intensity_processing,
    scale_ion_images_zscore as scale_ion_images_zscore_processing,
    tic_normalize_msicube,
)
from deltamsi.processing.colocalization import compute_mz_cosine_colocalization
from deltamsi.processing.spatial_chaos import (
    compute_spatial_chaos_matrix,
    spatial_chaos_fold_change_from_adata,
)

from deltamsi.processing.recalibration_core import load_database_masses

from deltamsi.processing.recalibration_cli_clean import write_corrected_msi
from deltamsi.processing.recal_visu_clean import diagnostics_for_pixel, select_pixels

from deltamsi.processing.mass_clustering import (
    cluster_masses_from_colocalization,
    cluster_masses_with_candidates,
)
from deltamsi.processing.kendrick import compute_kendrick_varm
from deltamsi.processing.mass_neighbors import direct_mass_neighbors
from deltamsi.processing.discriminant_analysis import rank_ions_groups_msi
from deltamsi.processing.preprocessing import (
    msi_cap_hotspots as msi_cap_hotspots_processing,
    msi_median_filter_2d as msi_median_filter_2d_processing,
    msi_threshold_quantile as msi_threshold_quantile_processing,
)
from deltamsi.processing.mz_matching import (
    match_mzs_to_var_simple as match_mzs_to_var_simple_processing,
)
from deltamsi.plotting.plot_kendrick_cluster_mz import plot_kendrick_from_clustering

from deltamsi.params.options import (
    MeanSpectrumOptions,
    GlobalMeanSpectrumOptions,
    PeakPickingOptions,
    PeakMatrixOptions,
    RecalibrationOptions,
    MassClusteringOptions,
    KendrickPlotOptions,
    CosineColocParams,
    RecalParams,
    RankIonsMSIParams,
)


class Logger:
    def info(self, msg: str) -> None:
        print(f"INFO: {msg}")

    def warning(self, msg: str) -> None:
        print(f"WARNING: {msg}")

    def error(self, msg: str) -> None:
        print(f"ERROR: {msg}")


logger = Logger()


class MSICube:
    """
    Main object for Mass Spectrometry Imaging (MSI) analysis.

    Encapsulates the entire workflow from raw imzML files to a fully 
    processed AnnData object.

    Attributes
    ----------
    data_directory : str
        Path to the directory containing raw imzML/ibd files.
    adata : ad.AnnData or None
        The annotated data matrix containing intensities and metadata.
    org_imzml_path_dict : dict
        Mapping of sample names to their corresponding file paths.
    """

    data_directory: str
    adata: Optional[ad.AnnData]

    def __init__(self, data_directory: str) -> None:
        """
        Initialize the MSICube by scanning for available imzML files.

        This constructor verifies the existence of the provided directory, 
        populates a dictionary of available mass spectrometry imaging (MSI) 
        datasets, and prepares a placeholder for the future AnnData object.

        Parameters
        ----------
        data_directory : str
            The file system path to the directory containing `.imzML` and `.ibd` files.

        Attributes
        ----------
        data_directory : str
            The validated path to the data source.
        org_imzml_path_dict : dict of {str : str}
            A mapping where keys are dataset names (usually filenames) and 
            values are the absolute paths to the `.imzML` files.
        adata : ad.AnnData or None
            The container for multi-sample MSI data, initialized as None. 
            Populated after calling the loading methods.

        Raises
        ------
        FileNotFoundError
            If the provided `data_directory` is not a valid directory.

        Examples
        --------
        >>> cube = MSICube("./data/my_msi_experiment")
        INFO: MSICube initialized with 3 samples found.
        >>> print(cube.org_imzml_path_dict.keys())
        dict_keys(['sample_A', 'sample_B', 'sample_C'])
        """
        if not os.path.isdir(data_directory):
            raise FileNotFoundError(f"The directory does not exist: {data_directory}")

        self.data_directory = data_directory

        self.org_imzml_path_dict: Dict[str, str] = {}
        self._scan_imzml_files(data_directory)

        # 2. Initialization of an empty anndata object
        # The anndata object will be filled later; we initialize the slot now.
        self.adata = None

        logger.info(
            f"MSICube initialized with {len(self.org_imzml_path_dict)} samples found."
        )

    def _default_adata_path(self, file_format: Literal["h5ad", "zarr"]) -> str:
        """
        Construct the default file path for saving the AnnData object.

        The path is generated by appending the appropriate file extension 
        to a default filename ("adata") located within the instance's 
        data directory.

        Parameters
        ----------
        file_format : {'h5ad', 'zarr'}
            The desired storage format for the AnnData object. 
            Determines whether the extension will be '.h5ad' (HDF5) 
            or '.zarr' (Zarr directory).

        Returns
        -------
        str
            The absolute or relative path where the AnnData object should 
            be persisted, following the pattern: 
            ``{self.data_directory}/adata.{file_format}``.

        See Also
        --------
        save : The public method used to persist data to this path.

        Examples
        --------
        Assuming ``self.data_directory`` is set to ``"/path/to/project"``:

        >>> cube._default_adata_path("h5ad")
        '/path/to/project/adata.h5ad'
        
        >>> cube._default_adata_path("zarr")
        '/path/to/project/adata.zarr'
        """

        extension = "h5ad" if file_format == "h5ad" else "zarr"
        return os.path.join(self.data_directory, f"adata.{extension}")

    def save(
        self,
        adata_path: Optional[str] = None,
        file_format: Literal["h5ad", "zarr"] = "h5ad",
        **kwargs: Any,
    ) -> str:
        """
        Persist the internal AnnData object to disk.

        Saves the processed MSI data stored in ``self.adata`` to either HDF5 
        (.h5ad) or Zarr format. If no path is provided, it uses a default 
        naming convention within the project's data directory.

        Parameters
        ----------
        adata_path : str, optional
            The specific file path where the data should be saved. If None, 
            defaults to ``{data_directory}/adata.{file_format}``.
        file_format : {'h5ad', 'zarr'}, default 'h5ad'
            The serialization format to use:
            
            * 'h5ad': Standard HDF5-based format for AnnData.
            * 'zarr': Chunked, compressed, binary format (better for cloud 
            storage or very large datasets).
        **kwargs : dict
            Additional keyword arguments passed directly to 
            :meth:`anndata.AnnData.write_h5ad` or :meth:`anndata.AnnData.write_zarr`. 
            Useful for specifying compression settings or chunk sizes.

        Returns
        -------
        str
            The absolute path to the saved file or directory.

        Raises
        ------
        ValueError
            If ``self.adata`` is None (i.e., no data has been loaded or processed).
        FileNotFoundError
            If the directory specified in `adata_path` does not exist.

        Notes
        -----
        The Zarr format is often preferred for multi-scale data or when 
        parallel read/write access is required.

        Examples
        --------
        Save the cube data using the default settings:

        >>> cube = MSICube("path/to/data")
        >>> # ... after loading data ...
        >>> cube.save()
        'path/to/data/adata.h5ad'

        Save to a custom location using Zarr format with specific compression:

        >>> cube.save(
        ...     adata_path="/exports/results/experiment_1.zarr",
        ...     file_format="zarr",
        ...     storage_transformer=my_transformer
        ... )
        '/exports/results/experiment_1.zarr'
        """

        if self.adata is None:
            raise ValueError("No AnnData object to save. Run analysis steps first.")

        save_path = adata_path or self._default_adata_path(file_format)
        parent_dir = os.path.dirname(save_path)

        if parent_dir and not os.path.isdir(parent_dir):
            raise FileNotFoundError(
                f"The directory does not exist for saving AnnData: {parent_dir}"
            )

        if file_format == "h5ad":
            restore_intensity_clipping = None
            intensity_clipping = self.adata.uns.get("intensity_clipping")
            if (
                isinstance(intensity_clipping, list)
                and intensity_clipping
                and all(isinstance(item, dict) for item in intensity_clipping)
            ):
                restore_intensity_clipping = intensity_clipping
                self.adata.uns["intensity_clipping"] = [
                    json.dumps(item, default=str) for item in intensity_clipping
                ]
            try:
                try:
                    self.adata.write_h5ad(save_path, **kwargs)
                except RuntimeError as exc:
                    if "allow_write_nullable_strings" not in str(exc):
                        raise
                    previous_setting = ad.settings.allow_write_nullable_strings
                    ad.settings.allow_write_nullable_strings = True
                    try:
                        self.adata.write_h5ad(save_path, **kwargs)
                    finally:
                        ad.settings.allow_write_nullable_strings = previous_setting
            finally:
                if restore_intensity_clipping is not None:
                    self.adata.uns["intensity_clipping"] = restore_intensity_clipping
        else:
            self.adata.write_zarr(save_path, **kwargs)

        logger.info(f"AnnData saved to {save_path} (format={file_format}).")
        return save_path

    def load(
        self,
        adata_path: Optional[str] = None,
        file_format: Literal["h5ad", "zarr"] = "h5ad",
        **kwargs: Any,
    ) -> ad.AnnData:
        """
        Load an AnnData object from disk and attach it to the MSICube instance.

        This method reads a previously saved MSI dataset in either HDF5 or Zarr 
        format. Once loaded, the data is stored in the ``self.adata`` attribute 
        and returned to the caller.

        Parameters
        ----------
        adata_path : str, optional
            The file path to the AnnData object. If None, the method looks for 
            a file named ``adata.<ext>`` within the instance's ``data_directory``.
        file_format : {'h5ad', 'zarr'}, default 'h5ad'
            The format of the file to be loaded:
            
            * 'h5ad': Standard HDF5-based AnnData file.
            * 'zarr': Zarr directory format, suitable for large-scale data.
        **kwargs : dict
            Additional keyword arguments passed to the underlying reading 
            functions: :func:`anndata.read_h5ad` or :func:`anndata.read_zarr`. 
            This can include arguments like ``backed='r'`` for lazy loading.

        Returns
        -------
        ad.AnnData
            The loaded AnnData object containing the MSI data and metadata.

        Raises
        ------
        FileNotFoundError
            If the specified `adata_path` or the default path does not exist 
            on the file system.

        See Also
        --------
        save : Persist the current AnnData object to disk.

        Examples
        --------
        Load the default 'h5ad' file from the data directory:

        >>> cube = MSICube("./experiment_data")
        >>> cube.load()
        INFO: AnnData loaded from ./experiment_data/adata.h5ad (format=h5ad).
        AnnData object with n_obs × n_vars = 1500 × 20000

        Load a specific Zarr file with memory-mapping (backed mode):

        >>> cube.load(
        ...     adata_path="results/processed_data.zarr",
        ...     file_format="zarr",
        ...     chunks="auto"
        ... )
        INFO: AnnData loaded from results/processed_data.zarr (format=zarr).
        """

        load_path = adata_path or self._default_adata_path(file_format)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"AnnData file not found: {load_path}")

        if file_format == "h5ad":
            loaded_adata = ad.read_h5ad(load_path, **kwargs)
        else:
            loaded_adata = ad.read_zarr(load_path, **kwargs)

        self.adata = loaded_adata
        logger.info(f"AnnData loaded from {load_path} (format={file_format}).")
        return loaded_adata

    def clip_or_mask_intensities(
        self,
        *,
        low: Optional[float] = None,
        high: Optional[float] = None,
        low_action: LowAction = "nan",
        high_action: HighAction = "clip",
        layer: Optional[str] = None,
        output_layer: Optional[str] = None,
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
        output_layer : str, optional
            The key in ``adata.layers`` where the processed matrix will be stored. 
            If None, the processed data are written to ``adata.X`` (the source
            layer is never modified).
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
            If ``self.adata`` is None or if 'move' action is used without a `low` value.
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

        Examples
        --------
        Perform a simple high-end clip on the main matrix:

        >>> cube.clip_or_mask_intensities(high=5000.0, high_action="clip")

        Create a new layer for "denoised" data by zeroing values below a threshold:

        >>> cube.clip_or_mask_intensities(
        ...     low=10.0, 
        ...     low_action="zero", 
        ...     output_layer="denoised"
        ... )

        Use the 'move' action to subtract background and return a copy:

        >>> new_adata = cube.clip_or_mask_intensities(
        ...     low=5.0, 
        ...     low_action="move", 
        ...     copy=True
        ... )
        """
        return clip_or_mask_intensities_processing(
            self,
            low=low,
            high=high,
            low_action=low_action,
            high_action=high_action,
            layer=layer,
            output_layer=output_layer,
            copy=copy,
        )

    @classmethod
    def from_file(
        cls,
        data_directory: str,
        adata_path: Optional[str] = None,
        file_format: Literal["h5ad", "zarr"] = "h5ad",
        **kwargs: Any,
    ) -> "MSICube":
        """
        Instantiate an MSICube and immediately load a persisted AnnData object.

        This factory method simplifies the workflow of resuming an existing project 
        by combining class initialization with the loading of processed MSI data 
        from a file or directory.

        Parameters
        ----------
        data_directory : str
            The root directory for the project. This path is used to validate 
            the data location and to derive the default `adata_path` if none 
            is provided.
        adata_path : str, optional
            The specific file path to the saved AnnData object. If None, it 
            defaults to ``{data_directory}/adata.{file_format}``.
        file_format : {'h5ad', 'zarr'}, default 'h5ad'
            The storage format of the file to be loaded.
        **kwargs : dict
            Additional keyword arguments passed to :meth:`load`, which in 
            turn are passed to :func:`anndata.read_h5ad` or :func:`anndata.read_zarr`. 
            Common arguments include ``backed='r'`` for memory-mapping.

        Returns
        -------
        MSICube
            An initialized instance of MSICube with the `adata` attribute 
            successfully populated from disk.

        Raises
        ------
        FileNotFoundError
            If the `data_directory` does not exist or if the AnnData file 
            cannot be found at the specified path.

        See Also
        --------
        load : The underlying method used to load the data.
        __init__ : The standard constructor for MSICube.

        Examples
        --------
        Load a project using default naming in the current directory:

        >>> cube = MSICube.from_file("./my_project")
        INFO: MSICube initialized with 5 samples found.
        INFO: AnnData loaded from ./my_project/adata.h5ad (format=h5ad).

        Load a project from a specific Zarr archive with lazy loading:

        >>> cube = MSICube.from_file(
        ...     data_directory="./data",
        ...     adata_path="./results/final_cube.zarr",
        ...     file_format="zarr",
        ...     backed="r"
        ... )
        """

        cube = cls(data_directory=data_directory)
        cube.load(adata_path=adata_path, file_format=file_format, **kwargs)
        return cube

    def log1p_intensity(
        self,
        *,
        base: Optional[float] = None,
        layer: Optional[str] = None,
        output_layer: Optional[str] = None,
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
        output_layer : str, optional
            The destination layer name for the transformed data. If None, the
            transformed data are written to ``adata.X`` (the source layer is never
            modified).
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
            If ``self.adata`` is None or if the targeted data matrix is empty.
        KeyError
            If the specified `layer` does not exist in ``adata.layers``.

        Notes
        -----
        The transformation is applied to both dense and sparse matrices efficiently.
        For sparse data, only non-zero entries are affected, preserving the 
        sparsity structure of the matrix.

        Examples
        --------
        Apply natural log1p transformation in-place:

        >>> cube.log1p_intensity()
        >>> print(cube.adata.uns["log1p"])
        {'base': None}

        Return a new cube with log2 transformation for a specific layer:

        >>> new_cube = cube.log1p_intensity(
        ...     base=2.0,
        ...     layer="raw",
        ...     output_layer="log2",
        ...     copy=True,
        ... )
        >>> # The original 'cube' remains unchanged.
        """
        return log1p_intensity_processing(
            self, base=base, layer=layer, output_layer=output_layer, copy=copy
        )

    def msi_cap_hotspots(
        self,
        *,
        q: float = 0.99,
        layer: Optional[str] = None,
        output_layer: Optional[str] = None,
        chunk_size: int = 256,
        dtype: Union[str, np.dtype] = "float32",
    ) -> None:
        """
        Cap ion images at a specific quantile to eliminate pixel hotspots.

        This is a convenience wrapper around
        :func:`deltamsi.processing.preprocessing.msi_cap_hotspots` that operates
        on ``self.adata``.
        """

        msi_cap_hotspots_processing(
            self,
            q=q,
            layer=layer,
            output_layer=output_layer,
            chunk_size=chunk_size,
            dtype=dtype,
        )

    def msi_threshold_quantile(
        self,
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

        This is a convenience wrapper around
        :func:`deltamsi.processing.preprocessing.msi_threshold_quantile` that
        operates on ``self.adata``.
        """

        msi_threshold_quantile_processing(
            self,
            q=q,
            mode=mode,
            layer=layer,
            output_layer=output_layer,
            chunk_size=chunk_size,
            dtype=dtype,
        )

    def msi_median_filter_2d(
        self,
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

        This is a convenience wrapper around
        :func:`deltamsi.processing.preprocessing.msi_median_filter_2d` that
        operates on ``self.adata``.
        """

        msi_median_filter_2d_processing(
            self,
            size=size,
            layer=layer,
            output_layer=output_layer,
            dtype=dtype,
            x_key=x_key,
            y_key=y_key,
            spatial_key=spatial_key,
            shape=shape,
            origin=origin,
            fill_value=fill_value,
            nan_to_num_before=nan_to_num_before,
            chunk_size=chunk_size,
        )

    def compute_cosine_colocalization(
        self, *, params: CosineColocParams = CosineColocParams()
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Compute the spatial similarity (colocalization) matrix between m/z images.

        This method quantifies the spatial overlap between different ions by 
        calculating the cosine similarity of their intensity distributions. It 
        identifies ions that co-localize within the same histological regions.

        Parameters
        ----------
        params : CosineColocParams, optional
            A dataclass instance defining the computation settings. 
            Available attributes are:

            * **layer** : *str, default None*
            The ``adata.layers`` key to use (uses ``adata.X`` if None).
            * **mode** : *{'topk_sparse', 'dense'}, default 'topk_sparse'*
            'topk_sparse' for large memory-efficient matrices, 
            'dense' for a full NumPy array.
            * **topk** : *int, default 50*
            Number of top neighbors to keep (for sparse mode).
            * **min_sim** : *float, default 0.2*
            Minimum similarity threshold to store a value.
            * **chunk_size** : *int, default 256*
            Number of m/z features processed per block (CPU optimization).
            * **symmetrize** : *bool, default True*
            Ensures the matrix is symmetric ($S_{ij} = S_{ji}$).
            * **include_self** : *bool, default False*
            Whether to keep the diagonal (similarity of 1.0).
            * **store_varp_key** : *str, default "ion_cosine"*
            Key used to save the result in ``adata.varp``.

        Returns
        -------
        Union[numpy.ndarray, scipy.sparse.csr_matrix]
            Similarity matrix of shape ``(n_features, n_features)``.

        Raises
        ------
        ValueError
            If ``self.adata`` is None.

        Examples
        --------
        **1. Quick start with defaults**
        
        >>> cube.compute_cosine_colocalization()

        **2. Full customization via CosineColocParams**
        
        >>> from deltamsi.params import CosineColocParams
        >>> # Define specific parameters for a large dataset
        >>> custom_config = CosineColocParams(
        ...     mode="topk_sparse",
        ...     topk=100,
        ...     min_sim=0.4,
        ...     layer="normalized",
        ...     store_varp_key="high_res_coloc"
        ... )
        >>> # Execute computation
        >>> sim_matrix = cube.compute_cosine_colocalization(params=custom_config)
        
        **3. Accessing the results in AnnData**
        
        >>> print(cube.adata.varp["high_res_coloc"].shape)
        """

        if self.adata is None:
            raise ValueError("MSICube.adata is None. Run data extraction first.")

        return compute_mz_cosine_colocalization(self, params=params)

    def aggregate_vars_by_label(
        self,
        label_col: str,
        *,
        layer: Optional[str] = None,
        agg: Agg = "mean",
        obsm_key: str = "X_by_label",
        dropna: bool = True,
        keep_order: bool = True,
        as_df: bool = False,
        dtype: Union[np.dtype, type] = np.float32,
    ) -> pd.Index:
        """
        Aggregate variables (m/z features) that share the same label in ``adata.var``.

        This method groups features based on a metadata column (e.g., metabolite 
        assignments or lipid classes) and computes a representative intensity 
        profile for each group across all pixels. The results are stored in 
        the observation-wise multidimensional arrays (``obsm``).

        Parameters
        ----------
        label_col : str
            The name of the column in ``self.adata.var`` used for grouping. 
            Each unique value in this column will become a feature in the 
            resulting aggregated matrix.
        layer : str, optional
            Target a specific entry in ``adata.layers`` for aggregation. If None, 
            the main intensity matrix ``adata.X`` is used.
        agg : {'mean', 'median', 'max'}, default 'mean'
            The mathematical strategy to combine intensities of variables 
            sharing the same label.
        obsm_key : str, default 'X_by_label'
            The key under which the resulting aggregated matrix will be stored 
            in ``self.adata.obsm``.
        dropna : bool, default True
            If True, features with ``NaN`` or missing values in `label_col` are 
            excluded. If False, these are grouped under the label "NA".
        keep_order : bool, default True
            If True, the output columns follow the order in which labels first 
            appear in the variable index. If False, labels are sorted 
            alphabetically.
        as_df : bool, default False
            If True, stores the aggregated results as a :class:`pandas.DataFrame` 
            inside ``obsm``. If False, stores them as a :class:`numpy.ndarray`.
        dtype : numpy.dtype or type, default numpy.float32
            The numerical precision of the resulting aggregated matrix.

        Returns
        -------
        pd.Index
            The index of unique labels created by the aggregation, in the order 
            they appear in the output matrix.

        Notes
        -----
        The list of labels is also permanently stored in 
        ``adata.uns[f"{obsm_key}_labels"]`` to ensure the column order of 
        the aggregated matrix is always recoverable.

        Examples
        --------
        Aggregate ion intensities by their assigned metabolite names:

        >>> # Assume adata.var['metabolite'] contains ['A', 'A', 'B', 'C', 'B']
        >>> labels = cube.aggregate_vars_by_label(
        ...     label_col="metabolite",
        ...     agg="max",
        ...     obsm_key="metabolite_intensities"
        ... )
        >>> print(labels)
        Index(['A', 'B', 'C'], name='metabolite')
        
        Access the aggregated spatial image for the first label ('A'):
        
        >>> agg_matrix = cube.adata.obsm["metabolite_intensities"]
        >>> image_A = agg_matrix[:, 0]
        """

        return aggregate_vars_by_label(
            self,
            label_col,
            layer=layer,
            agg=agg,
            obsm_key=obsm_key,
            dropna=dropna,
            keep_order=keep_order,
            as_df=as_df,
            dtype=dtype,
        )

    def rank_ions_groups_msi(self, *, params: RankIonsMSIParams) -> pd.DataFrame:
        """
        Rank ions by differential expression between two MSI groups.

        This is a convenience wrapper around
        :func:`deltamsi.processing.discriminant_analysis.rank_ions_groups_msi`
        that operates on ``self.adata`` and returns the ranked ions table.

        Parameters
        ----------
        params : RankIonsMSIParams
            Configuration object defining group labels, statistics, and output
            settings. Results are stored in ``adata.uns[params.key_added]``.

        Returns
        -------
        pandas.DataFrame
            The top-ranked ions table.

        Raises
        ------
        ValueError
            If ``self.adata`` is None.
        """

        if self.adata is None:
            raise ValueError("MSICube.adata is None. Load or compute data first.")

        return rank_ions_groups_msi(self.adata, params=params)

    def _scan_imzml_files(self, directory: str) -> None:
        """
        Scan the specified directory for valid imzML datasets.

        This internal helper method identifies files with the ``.imzML`` (or ``.imzml``) 
        extension and verifies the presence of their mandatory sibling ``.ibd`` 
        data files. Successfully validated datasets are stored in 
        ``self.org_imzml_path_dict``.

        Parameters
        ----------
        directory : str
            The file system path to scan for MSI datasets.

        Notes
        -----
        Mass Spectrometry Imaging (MSI) data in the imzML format consists of two 
        coupled files:
        
        1. **.imzML**: An XML file containing metadata and the spatial structure.
        2. **.ibd**: A binary file containing the actual spectral intensities.
        
        This method ensures that for every metadata file found, the corresponding 
        binary data file exists before adding it to the project.

        

        Attributes Modified
        -------------------
        org_imzml_path_dict : dict
            Updated with entries where the key is the filename (without extension) 
            and the value is the absolute path to the ``.imzML`` file.

        Examples
        --------
        If a directory contains:
        - ``sample1.imzML``
        - ``sample1.ibd``
        - ``sample2.imzml`` (missing .ibd)

        >>> cube = MSICube("path/to/data")
        >>> # _scan_imzml_files is called during __init__
        >>> print(cube.org_imzml_path_dict)
        {'sample1': 'path/to/data/sample1.imzML'}
        >>> # sample2 is ignored and a warning is logged due to the missing .ibd file.
        """
        # Regex to match the file name and capture the sample name, case-insensitive
        imzml_pattern = re.compile(r"(.+)\.imzml$", re.IGNORECASE)

        for filename in os.listdir(directory):
            match = imzml_pattern.match(filename)
            if match:
                sample_name = match.group(1)  # The sample name without the extension
                full_path = os.path.join(directory, filename)

                # Check for the associated .ibd file
                # The replacement uses 'imzML' to handle both cases from the full_path
                ibd_path = full_path.replace(".imzML", ".ibd").replace(".imzml", ".ibd")
                if not os.path.exists(ibd_path):
                    logger.warning(f"Missing .ibd file for {filename}")
                    continue

                # Store: sample_name -> full_path
                self.org_imzml_path_dict[sample_name] = full_path

    def _compute_all_mean_spectra(
        self, mode: Literal["profile", "centroid"], **kwargs: Any
    ) -> None:
        """
        Calculate the average mass spectrum for every imzML file in the directory.

        This method iterates through all discovered datasets, computes a 
        representative mean spectrum for each, and stores the results in the 
        AnnData object. The mean spectrum is useful for identifying global 
        ion peaks and determining the m/z range for subsequent data extraction.

        

        Parameters
        ----------
        mode : {'profile', 'centroid'}
            The acquisition mode of the input MSI data. 
            
            * 'profile': Continuous data where peaks have a Gaussian-like shape.
            * 'centroid': Discrete data where peaks are represented as single sticks.
        **kwargs : Any
            Parameters used to configure the m/z axis and peak detection accuracy. 
            Supported arguments include:

            * **min_mz** : *float, default 0.0*
            The minimum m/z value to consider.
            * **max_mz** : *float, default 2000.0*
            The maximum m/z value to consider.
            * **binning_p** : *float, default 0.001*
            The precision of the m/z bins (step size) used for accumulation.
            * **mass_accuracy_ppm** : *int, default 3*
            The mass accuracy in parts-per-million, used for peak alignment.
            * **tolerance_da** : *float, optional*
            Absolute tolerance in Daltons. If provided, overrides ``mass_accuracy_ppm``.
            * **n_sigma** : *float, default 3.0*
            The number of standard deviations for peak width calculation.

        Returns
        -------
        None
            Results are stored internally in ``self.adata.uns``.

        Attributes Modified
        -------------------
        adata.uns['mean_spectra'] : dict
            A dictionary mapping sample names to their respective ``mz`` 
            and ``intensity`` arrays.
        adata.uns['mean_spectra_options'] : dict
            The configuration settings used for this computation.
        adata.uns['mean_spectra_samples'] : list
            List of samples successfully processed.

        Raises
        ------
        ValueError
            If invalid options are provided or if the mode is not supported.

        Examples
        --------
        **1. Standard usage with default parameters**
        
        >>> cube._compute_all_mean_spectra(mode="centroid")

        **2. Customizing the m/z range and precision**
        
        If your data targets small metabolites (e.g., m/z 50 to 500) with 
        high-resolution binning:

        >>> cube._compute_all_mean_spectra(
        ...     mode="profile",
        ...     min_mz=50.0,
        ...     max_mz=500.0,
        ...     binning_p=0.0005,
        ...     mass_accuracy_ppm=5
        ... )

        **3. Accessing and plotting results**
        
        >>> sample_data = cube.adata.uns["mean_spectra"]["sample_01"]
        >>> mz, intensity = sample_data["mz"], sample_data["intensity"]
        """
        if self.adata is None:
            self.adata = ad.AnnData()

        mean_spectra_dict = {}

        # 1. Options Object Creation and Validation
        options_kwargs = {
            "mode": mode,
            "min_mz": kwargs.pop("min_mz", 0.0),
            "max_mz": kwargs.pop("max_mz", 2000.0),
            "binning_p": kwargs.pop("binning_p", 0.001),
            "tolerance_da": kwargs.pop("tolerance_da", None),
            "mass_accuracy_ppm": kwargs.pop("mass_accuracy_ppm", 3),
            "n_sigma": kwargs.pop("n_sigma", 3.0),
        }

        if kwargs:
            logger.warning(
                f"Ignoring unknown arguments passed to mean spectra computation: {list(kwargs.keys())}"
            )

        try:
            options = MeanSpectrumOptions(**options_kwargs)
            options.validate()
        except ValueError as e:
            raise ValueError(
                f"Invalid Mean Spectrum Options provided for mode '{mode}': {e}"
            ) from e

        # 2. Iteration and Calculation
        for sample_name, imzml_path in self.org_imzml_path_dict.items():
            logger.info(
                f"Calculating mean spectrum for: {sample_name} (Mode: {options.mode})"
            )

            try:
                mean_mz, mean_y = compute_mean_spectrum(imzml_path, options=options)

                mean_spectra_dict[sample_name] = {"mz": mean_mz, "intensity": mean_y}
            except Exception as e:
                logger.error(f"error during calculation for {sample_name}: {e}")
                continue

        # 3. Storing in AnnData
        self.adata.uns["mean_spectra"] = mean_spectra_dict
        self.adata.uns["mean_spectra_options"] = options.__dict__
        self.adata.uns["mean_spectra_samples"] = list(self.org_imzml_path_dict.keys())
        logger.info("Mean spectra calculated and stored.")

    def _compute_global_mean_spectrum(self, **kwargs: Any) -> None:
        """
        Compute a single global mean spectrum by combining all individual mean spectra.

        This method aggregates the mean spectra previously calculated for each sample 
        into a single consensus spectrum. This is a critical step for defining a 
        common m/z axis across multiple MSI datasets, enabling cross-sample 
        comparisons and peak picking on the entire project.

        Parameters
        ----------
        **kwargs : Any
            Configuration parameters for the spectral combination process. 
            Supported arguments include:

            * **binning_p** : *float, default 0.0001*
            The precision (step size) of the global m/z axis. A smaller value 
            preserves more spectral resolution but increases memory usage.
            * **use_intersection** : *bool, default True*
            If True, the global m/z range will be the intersection (overlap) 
            of all samples. If False, the union (full range) is used.
            * **tic_normalize** : *bool, default True*
            Whether to normalize individual mean spectra by their Total Ion 
            Current (TIC) before averaging to account for intensity variations 
            between runs.
            * **compress_axis** : *bool, default False*
            If True, removes empty m/z bins to reduce the size of the 
            resulting arrays.

        Returns
        -------
        None
            The global spectrum is stored internally in ``self.adata.uns``.

        Attributes Modified
        -------------------
        adata.uns['mean_spectrum_global'] : dict
            A dictionary containing the consensus ``mz`` and ``intensity`` arrays.
        adata.uns['mean_spectrum_global_options'] : dict
            The configuration settings used for the combination.

        Raises
        ------
        ValueError
            If invalid options are provided or if required spectra are missing.
        RuntimeError
            If ``compute_mean_spectra(scope="samples")`` has not been executed previously.

        Notes
        -----
        The global mean spectrum serves as the reference for the peak picking 
        algorithms. If samples were acquired with significantly different 
        m/z ranges, ensure ``use_intersection`` is set according to your 
        biological questions.

        Examples
        --------
        **1. Standard global combination**
        
        >>> cube._compute_all_mean_spectra(mode="centroid")
        >>> cube._compute_global_mean_spectrum()

        **2. High-resolution combination using the full m/z range**
        
        If you want to keep data from all samples even if they don't overlap 
        perfectly:

        >>> cube._compute_global_mean_spectrum(
        ...     binning_p=0.00005,
        ...     use_intersection=False,
        ...     tic_normalize=True
        ... )

        **3. Visualizing the consensus**
        
        >>> global_spec = cube.adata.uns["mean_spectrum_global"]
        >>> print(f"Global bins: {len(global_spec['mz'])}")
        """
        if self.adata is None or "mean_spectra" not in self.adata.uns:
            logger.error(
                "Individual mean spectra not calculated yet. Run compute_mean_spectra(scope='samples') first."
            )
            return

        logger.info("Computing global mean spectrum...")

        # 1. Options Object Creation and Validation
        options_kwargs = {
            "binning_p": kwargs.pop("binning_p", 0.0001),
            "use_intersection": kwargs.pop("use_intersection", True),
            "tic_normalize": kwargs.pop("tic_normalize", True),
            "compress_axis": kwargs.pop("compress_axis", False),
        }

        if kwargs:
            logger.warning(
                f"Ignoring unknown arguments passed to global mean spectra computation: {list(kwargs.keys())}"
            )

        try:
            options = GlobalMeanSpectrumOptions(**options_kwargs)
            options.validate()
        except ValueError as e:
            raise ValueError(
                f"Invalid Global Mean Spectrum Options provided: {e}"
            ) from e

        # 2. Build the list of spectra from adata.uns
        mean_spectra_data: Dict[str, Dict[str, np.ndarray]] = self.adata.uns[
            "mean_spectra"
        ]

        list_of_spectra: List[Spectrum] = [
            (data["mz"], data["intensity"]) for data in mean_spectra_data.values()
        ]

        # 3. Call combine_mean_spectra
        try:
            mzs_combined, ints_combined = combine_mean_spectra(
                list_of_spectra,
                options,  # Use __dict__ to expose options as kwargs
            )
        except Exception as e:
            logger.error(f"error during global spectrum combination: {e}")
            return

        # 4. Store the global mean spectrum in adata.uns
        self.adata.uns["mean_spectrum_global"] = {
            "mz": mzs_combined,
            "intensity": ints_combined,
        }
        self.adata.uns["mean_spectrum_global_options"] = options.__dict__
        logger.info(
            "Global mean spectrum calculated and stored in adata.uns['mean_spectrum_global']."
        )

    def compute_mean_spectra(
        self,
        scope: Literal["samples", "global"] = "samples",
        mode: Optional[Literal["profile", "centroid"]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Compute mean spectra for the project.

        Depending on the 'scope', this method either calculates an average spectrum 
        for each individual sample or combines existing sample spectra into a single 
        consensus (global) spectrum.

        Parameters
        ----------
        scope : {'samples', 'global'}, default 'samples'
            * 'samples': Calculates the mean spectrum for every imzML file.
            * 'global': Aggregates all sample spectra into one reference spectrum.
        mode : {'profile', 'centroid'}, optional
            Acquisition mode. Required if scope is 'samples'.
        **kwargs : Any
            If scope='samples': 
                Parameters used to configure the m/z axis and peak detection accuracy. 
                Supported arguments include:

                * **min_mz** : *float, default 0.0*
                The minimum m/z value to consider.
                * **max_mz** : *float, default 2000.0*
                The maximum m/z value to consider.
                * **binning_p** : *float, default 0.001*
                The precision of the m/z bins (step size) used for accumulation.
                * **mass_accuracy_ppm** : *int, default 3*
                The mass accuracy in parts-per-million, used for peak alignment.
                * **tolerance_da** : *float, optional*
                Absolute tolerance in Daltons. If provided, overrides ``mass_accuracy_ppm``.
                * **n_sigma** : *float, default 3.0*
                The number of standard deviations for peak width calculation.

            If scope='global': 
                Configuration parameters for the spectral combination process. 
                * **binning_p** : *float, default 0.0001*
                The precision (step size) of the global m/z axis. A smaller value 
                preserves more spectral resolution but increases memory usage.
                * **use_intersection** : *bool, default True*
                If True, the global m/z range will be the intersection (overlap) 
                of all samples. If False, the union (full range) is used.
                * **tic_normalize** : *bool, default True*
                Whether to normalize individual mean spectra by their Total Ion 
                Current (TIC) before averaging to account for intensity variations 
                between runs.
                * **compress_axis** : *bool, default False*
                If True, removes empty m/z bins to reduce the size of the 
                resulting arrays.


        See Also
        --------
        clear_mean_spectra : To remove results from memory.
        """
        if scope == "samples":
            if mode is None:
                raise ValueError("Argument 'mode' is required when scope='samples'.")
            self._compute_all_mean_spectra(mode=mode, **kwargs)
        
        elif scope == "global":
            self._compute_global_mean_spectrum(**kwargs)
        
        else:
            raise ValueError(f"Invalid scope '{scope}'. Choose 'samples' or 'global'.")


    def clear_mean_spectra(self) -> None:
        """
        Remove all mean spectra-related data from the AnnData object.

        Mass spectrometry imaging (MSI) datasets can be very large. The mean 
        spectra (per-sample and global) and their high-resolution m/z axes 
        can consume significant RAM or disk space when saving to ``.h5ad``. 
        This method provides a clean way to purge these metadata entries 
        from ``self.adata.uns`` once they are no longer needed (e.g., after 
        peak picking is completed).

        Returns
        -------
        None
            The entries are removed in-place from the ``self.adata.uns`` 
            dictionary.

        See Also
        --------
        compute_all_mean_spectra : Method that generates per-sample spectra.
        compute_global_mean_spectrum : Method that generates the consensus spectrum.

        Notes
        -----
        The following keys are targeted for removal from ``adata.uns``:
        
        * ``mean_spectra``
        * ``mean_spectra_options``
        * ``mean_spectra_samples``
        * ``mean_spectrum_global``
        * ``mean_spectrum_global_options``

        Examples
        --------
        Use this method to free up memory before saving a large cube to disk:

        >>> # After running extraction and analysis
        >>> cube.clear_mean_spectra()
        INFO: Removed mean spectra entries from adata.uns: mean_spectra, mean_spectra_options, mean_spectra_samples.
        
        >>> # Attempting to clear an already empty object
        >>> cube.clear_mean_spectra()
        INFO: No mean spectra entries found in adata.uns to remove.
        """

        if self.adata is None:
            logger.warning(
                "AnnData object (adata) is not initialized. Nothing to clear."
            )
            return

        keys_to_remove = [
            "mean_spectra",
            "mean_spectra_options",
            "mean_spectra_samples",
            "mean_spectrum_global",
            "mean_spectrum_global_options",
        ]

        removed_keys = []
        for key in keys_to_remove:
            if key in self.adata.uns:
                self.adata.uns.pop(key, None)
                removed_keys.append(key)

        if removed_keys:
            logger.info(
                "Removed mean spectra entries from adata.uns: "
                + ", ".join(removed_keys)
                + "."
            )
        else:
            logger.info("No mean spectra entries found in adata.uns to remove.")

    def pick_peaks(self, **kwargs: Any) -> None:
        """
        Detect local maxima in the global mean spectrum to define study-wide features.

        This method identifies significant ion peaks from the consensus mean spectrum. 
        The detected m/z values are used to define the feature variables (columns) 
        of the AnnData object. If the number of detected peaks differs from the 
        current feature count, the AnnData object is restructured with a 
        placeholder matrix.

        Parameters
        ----------
        **kwargs : Any
            Parameters used to configure the peak detection algorithm. 
            Supported arguments include:

            * **topn** : *int, default 10000*
            The maximum number of peaks to retain. The algorithm will select 
            the `topn` most intense peaks that satisfy the distance constraints.
            * **distance_da** : *float, optional*
            The minimum required distance between two peaks in absolute Daltons (Da). 
            This prevents selecting multiple points from the same isotopic envelope.
            * **distance_ppm** : *float, optional*
            The minimum required distance between peaks in parts-per-million (ppm). 
            This is often preferred for high-resolution data as it scales with 
            the m/z value.
            * **binning_p** : *float, default 0.0001*
            The m/z precision used to evaluate peak local maxima. Should ideally 
            match the `binning_p` used in ``compute_mean_spectra(scope="global")``.

        Returns
        -------
        None
            Updates ``self.adata.var`` with the new peak m/z values and stores 
            metadata in ``self.adata.uns['peak_picking_options']``.

        Raises
        ------
        ValueError
            If invalid options are provided or if the global mean spectrum 
            is missing.

        Notes
        -----
        **Restructuring Warning:**
        If peak picking results in a different number of features than currently 
        exists in ``self.adata``, the data matrix (``self.adata.X``) will be 
        reset to a zero-matrix placeholder. This is necessary to maintain 
        consistent dimensions for AnnData. You must call ``extract_peak_matrix()`` 
        afterwards to populate the matrix with actual pixel intensities.

        Examples
        --------
        **1. Standard peak picking**
        Detect up to 5000 peaks using default distance constraints:

        >>> cube.pick_peaks(topn=5000)

        **2. High-resolution distance constraints**
        Use a 15 ppm window to ensure peaks are well-separated in high-res data:

        >>> cube.pick_peaks(
        ...     topn=2000,
        ...     distance_ppm=15.0,
        ...     binning_p=0.00005
        ... )

        **3. Checking selected features**
        
        >>> print(cube.adata.var.head())
                          mz
        feature_id            
        mz_0_104.1234  104.1234
        mz_1_150.0891  150.0891
        """
        logger.info("Starting peak picking on the global mean spectrum.")

        # Check required input data
        if self.adata is None:
            logger.error(
                "AnnData object (adata) is not initialized. Run compute_all_mean_spectra first."
            )
            return

        if "mean_spectrum_global" not in self.adata.uns:
            logger.error(
                "Global mean spectrum not found in adata.uns['mean_spectrum_global']. "
                "Run compute_global_mean_spectrum first."
            )
            return

        # Options parsing and validation
        # Extract relevant kwargs for PeakPickingOptions
        options_kwargs = {
            "topn": kwargs.pop("topn", 10000),
            "binning_p": kwargs.pop("binning_p", 0.0001),
            "distance_da": kwargs.pop("distance_da", None),
            "distance_ppm": kwargs.pop("distance_ppm", None),
        }
        if kwargs:
            logger.warning(
                f"Ignoring unknown arguments passed to peak picking: {list(kwargs.keys())}"
            )

        try:
            options = PeakPickingOptions(**options_kwargs)
            options.validate()
        except Exception as e:
            logger.error(f"Invalid Peak Picking Options: {e}")
            return

        # Retrieve data and call the processing function
        global_spectrum = self.adata.uns["mean_spectrum_global"]
        mzs_combined = global_spectrum["mz"]
        ints_combined = global_spectrum["intensity"]

        # Call the peak_picking function with the options object
        selected_mzs = peak_picking(
            mzs=mzs_combined,
            intensities=ints_combined,
            options=options,
        )

        n_vars_new = len(selected_mzs)
        logger.info(f"Peak picking finished. {len(selected_mzs)} mz selected.")

        # Store results in adata.var

        # Create a new DataFrame for var based on the selected mz values.
        # AnnData format prefers unique identifiers for the .var index.
        # We use the formatted mz as a unique identifier for AnnData.var.
        new_var_data = pd.DataFrame(
            data={"mz": selected_mzs},
            # Index formatted for a readable identifier (e.g., 'mz_0_200.0000')
            index=[f"mz_{idx}_{m:.4f}" for idx, m in enumerate(selected_mzs)],
        )
        new_var_data.index.name = "feature_id"

        n_obs = self.adata.n_obs
        needs_resize = self.adata.n_vars != n_vars_new

        if needs_resize:
            logger.warning(
                f"Feature count mismatch: {self.adata.n_vars} old features vs. {n_vars_new} new features. "
                "Recreating AnnData structure to align dimensions."
            )

            # Save non-feature-dimension data before reconstruction
            # We assume only 'spatial' might be in obsm before extract_peak_matrix
            current_obs = self.adata.obs.copy()
            current_uns = self.adata.uns.copy()
            current_obsm = self.adata.obsm.copy()

            # Create a placeholder X (zero matrix) if there are observations (pixels)
            # This is necessary because the AnnData object in the fixture starts with X.
            new_X = np.zeros((n_obs, n_vars_new)) if n_obs > 0 else None

            # Reconstruct the AnnData object with the new var, obs, and placeholder X
            self.adata = ad.AnnData(
                X=new_X,
                obs=current_obs,
                var=new_var_data,  # This sets the new feature dimension
                uns=current_uns,
                obsm=current_obsm,
            )

            if n_obs > 0:
                logger.warning(
                    "adata.X structure reset to zero matrix placeholder to match new feature count. "
                    "Run extract_peak_matrix() next to populate real data."
                )

        else:
            # If n_vars did not change, simply replace var
            self.adata.var = new_var_data

        # Store options in adata.uns for provenance
        self.adata.uns["peak_picking_options"] = options.to_dict()

        logger.info(
            "Selected mz values stored in adata.var and adata.uns['peak_picking_options']."
        )

    def extract_matrix(self, **kwargs: Any) -> None:
        """
        Extract peak intensities for all pixels across all samples.

        This method populates the main data matrix (``X``) by integrating ion 
        signals around the m/z values defined in ``adata.var``. It scans the 
        raw binary data (``.ibd``) for every sample, aligns spatial coordinates, 
        and builds a unified multi-sample AnnData object.

        Parameters
        ----------
        **kwargs : Any
            Parameters used to define the integration window for each peak. 
            Supported arguments include:

            * **tol_da** : *float, optional*
            The absolute extraction window (tolerance) in Daltons. Intensities 
            within ``[target_mz - tol_da, target_mz + tol_da]`` are summed.
            * **tol_ppm** : *float, optional*
            The relative extraction window in parts-per-million. If provided, 
            the window scales with the m/z value. This is the standard choice 
            for high-resolution instruments (Orbitrap, FT-ICR).

        Returns
        -------
        None
            The ``self.adata`` object is fully populated and restructured.

        Attributes Modified
        -------------------
        adata.X : numpy.ndarray
            The intensity matrix of shape ``(n_pixels, n_peaks)``.
        adata.layers['RAW'] : numpy.ndarray
            A backup copy of the original extracted intensities.
        adata.obsm['spatial'] : numpy.ndarray
            The 2D/3D coordinates (x, y, [z]) for every pixel.
        adata.obs['sample'] : pandas.Series (Categorical)
            Provenance labels identifying which sample each pixel belongs to.
        adata.uns['matrix_extraction_options'] : dict
            Metadata tracking the tolerances used for extraction.

        Raises
        ------
        ValueError
            If no peaks are found in ``adata.var`` or if extraction options 
            fail validation.
        RuntimeError
            If a sample cannot be read or if data concatenation fails.

        Notes
        -----
        This method effectively "finalizes" the MSICube construction. It moves 
        the object from a metadata-only state to a data-rich state. For large 
        projects (thousands of pixels and peaks), this step may take several 
        minutes and requires significant RAM.

        Examples
        --------
        **1. Extract using a fixed PPM tolerance (Recommended)**
        
        >>> # Extract with a 10 ppm window around each peak
        >>> cube.extract_matrix(tol_ppm=10.0)
        INFO: Extraction complete. Final shape: (15000, 2000)

        **2. Extract using a fixed Dalton window**
        
        >>> # Extract with +/- 0.01 Da around each peak
        >>> cube.extract_matrix(tol_da=0.01)

        **3. Post-extraction inspection**
        
        >>> # Check coordinates of the first 5 pixels
        >>> print(cube.adata.obsm['spatial'][:5])
        >>> # See sample distribution
        >>> print(cube.adata.obs['sample'].value_counts())
        """
        logger.info("Starting extraction of peak intensity matrix (X) for all samples.")

        # 1. Prerequisites Check
        if self.adata is None or self.adata.var is None or "mz" not in self.adata.var:
            logger.error(
                "Target mz list not found in adata.var['mz']. "
                "Run pick_peaks first."
            )
            return

        # 2. Options Parsing
        options_kwargs = {
            "tol_da": kwargs.pop("tol_da", None),
            "tol_ppm": kwargs.pop("tol_ppm", None),
        }

        try:
            options = PeakMatrixOptions(**options_kwargs)
            options.validate()
        except ValueError as e:
            logger.error(f"Invalid Peak Matrix Options: {e}")
            return

        # 3. Preparation
        target_mzs = self.adata.var["mz"].values

        # Containers for batch accumulation
        X_blocks = []
        coords_blocks = []
        sample_labels = []
        obs_names = []

        # 4. Processing Loop
        # Sort keys to ensure reproducible order of samples
        sorted_samples = sorted(self.org_imzml_path_dict.keys())

        for sample_name in sorted_samples:
            imzml_path = str(self.org_imzml_path_dict[sample_name])
            logger.info(f"Extracting matrix for sample: {sample_name}")

            try:
                X_sample, coords_sample = extract_peak_matrix(
                    imzml_path, target_mzs, options=options
                )

                n_pixels = X_sample.shape[0]

                X_blocks.append(X_sample)
                coords_blocks.append(coords_sample)

                # Create labels for obs
                sample_labels.extend([sample_name] * n_pixels)
                # Unique index: sample_name + pixel_index
                obs_names.extend([f"{sample_name}_p{i}" for i in range(n_pixels)])

            except Exception as e:
                logger.error(f"Failed to extract matrix for {sample_name}: {e}")
                # We stop here to avoid creating a corrupted AnnData object with missing samples
                return

        # 5. Concatenation and Assembly
        logger.info("Concatenating data from all samples...")

        try:
            X_all = np.vstack(X_blocks)
            coords_all = np.vstack(coords_blocks)
        except ValueError as e:
            logger.error(
                f"Error during concatenation (possibly no data extracted): {e}"
            )
            return

        # 6. Final AnnData Population

        # We recreate the AnnData object to resize dimensions (n_obs changed from 0 to N_total)
        # while keeping var and uns.
        new_obs = pd.DataFrame({"sample": sample_labels}, index=obs_names)

        # Make 'sample' a categorical column (efficient for large datasets)
        new_obs["sample"] = new_obs["sample"].astype("category")

        self.adata = ad.AnnData(
            X=X_all,
            obs=new_obs,
            var=self.adata.var,  # Keep the selected peaks
            uns=self.adata.uns,  # Keep provenance options
            obsm={"spatial": coords_all},
        )

        self.adata.layers["RAW"] = self.adata.X.copy()

        # Store options used for this step
        self.adata.uns["matrix_extraction_options"] = options.to_dict()

        logger.info(
            f"Extraction complete. "
            f"Final shape: {self.adata.shape} (Pixels x Peaks). "
            f"Data stored in adata.X, adata.obsm['spatial'], adata.obs['sample']."
        )

    def create_feature_matrix(
        self,
        peak_picking_kwargs: dict | None = None,
        extraction_kwargs: dict | None = None,
    ) -> None:
        """
        Create a full pixel × feature matrix from raw MSI data.

        This is a high-level convenience method that combines:

        1. Peak detection on the global mean spectrum (``pick_peaks``)
        2. Extraction of pixel-wise intensities for the detected peaks
        (``extract_matrix``)

        It is the recommended entry point for building ``adata.X``.

        Parameters
        ----------
        peak_picking_kwargs : dict, optional
            Keyword arguments forwarded to ``pick_peaks``.
            If ``None``, default peak picking parameters are used.

            Supported options include:

            * **topn** : int, default 10000  
            Maximum number of peaks to retain.
            * **distance_da** : float, optional  
            Minimum absolute distance between peaks (Da).
            * **distance_ppm** : float, optional  
            Minimum relative distance between peaks (ppm).
            * **binning_p** : float, default 0.0001  
            m/z precision used for peak detection.

            Example::
            
                peak_picking_kwargs={
                    "topn": 3000,
                    "distance_ppm": 15.0,
                    "binning_p": 0.00005,
                }

        extraction_kwargs : dict, optional
            Keyword arguments forwarded to ``extract_matrix``.
            If ``None``, default extraction parameters are used.

            Supported options include:

            * **tol_da** : float, optional  
            Absolute extraction tolerance in Daltons.
            * **tol_ppm** : float, optional  
            Relative extraction tolerance in ppm (recommended).

            Example::
            
                extraction_kwargs={
                    "tol_ppm": 10.0
                }

        Returns
        -------
        None
            Populates ``self.adata`` with:

            * ``adata.X`` : intensity matrix (pixels × peaks)
            * ``adata.var`` : detected peak m/z values
            * ``adata.obs`` : pixel metadata (sample labels)
            * ``adata.obsm['spatial']`` : spatial coordinates

        Notes
        -----
        - This method may restructure the AnnData object.
        - All parameters used during peak picking and extraction are
        stored in ``adata.uns`` for full provenance tracking.
        - For large datasets, this operation may be time- and memory-intensive.

        See Also
        --------
        pick_peaks : Detect study-wide m/z features.
        extract_matrix : Extract pixel-wise intensities for selected peaks.
        """

        self.pick_peaks(**(peak_picking_kwargs or {}))
        self.extract_matrix(**(extraction_kwargs or {}))

    def scale_zscore(
        self,
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
            The destination layer name for the scaled data. If None, the scaled data
            are written to ``adata.X`` (the source layer is never modified).
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
            If ``self.adata`` is None or if data is missing.
        KeyError
            If the specified `layer` or the grouping column (sample/condition) 
            is missing from AnnData.

        Notes
        -----
        Standardizing "per_sample" is a common strategy in MSI to mitigate 
        instrumental drift across multiple sections or slides.

        Examples
        --------
        **1. Global Z-score scaling in-place**
        
        >>> cube.scale_zscore(mode="all")

        **2. Per-sample scaling with outlier clipping**
        
        >>> cube.scale_zscore(
        ...     mode="per_sample", 
        ...     max_value=3.0, 
        ...     output_layer="scaled"
        ... )

        **3. Scale a specific layer and retrieve statistics**
        
        >>> stats = cube.scale_zscore(
        ...     layer="log1p", 
        ...     return_stats=True
        ... )
        >>> # Get mean of the first m/z for 'sample_A'
        >>> mean_val = stats['sample_A'][0][0]
        """
        return scale_ion_images_zscore_processing(
            self,
            mode=mode,
            layer=layer,
            output_layer=output_layer,
            with_mean=with_mean,
            with_std=with_std,
            ddof=ddof,
            eps=eps,
            max_value=max_value,
            return_stats=return_stats,
            copy=copy,
        )

    def normalize_tic(
        self,
        *,
        target_sum: float = 1e6,
        layer: Optional[str] = None,
        output_layer: Optional[str] = None,
        store_tic_in_obs: Optional[str] = "tic",
        copy: bool = False,
    ) -> Optional["MSICube"]:
        """
        Apply Total Ion Current (TIC) normalization to the intensity matrix.

        TIC normalization scales each spectrum (pixel) such that the sum of all 
        intensities equals a constant target value. This is a standard 
        preprocessing step in MSI to correct for variations in ion source 
        efficiency, matrix application heterogeneity, or tissue thickness.

        The operation follows the formula:
        $$I_{norm} = \frac{I_{raw}}{\sum I_{raw}} \times target\_sum$$

        Parameters
        ----------
        target_sum : float, default 1e6
            The fixed value that the sum of each spectrum will equal after 
            normalization. Standard values are often $10^4$ or $10^6$.
        layer : str, optional
            The specific layer in ``adata.layers`` to normalize. If None, the 
            main matrix ``adata.X`` is used.
        output_layer : str, optional
            The destination layer name for the normalized data. If None, the
            normalized data are written to ``adata.X`` (the source layer is never
            modified).
        store_tic_in_obs : str, optional, default 'tic'
            Key in ``adata.obs`` used to store the original TIC values (the sum 
            of intensities per pixel before normalization). This is useful 
            for quality control. Set to None to skip storage.
        copy : bool, default False
            Whether to return a new MSICube instance.

            * If True, a deep copy of the instance is created and returned.
            * If False, the transformation is applied in-place and None is returned.

        Returns
        -------
        MSICube or None
            The normalized MSICube instance if ``copy=True``, otherwise None.

        Raises
        ------
        ValueError
            If ``self.adata`` is None (no data loaded).

        Notes
        -----
        TIC normalization assumes that the majority of the signal variation 
        between pixels is technical rather than biological. It may not be 
        suitable if a few very high-intensity peaks dominate the spectrum 
        differently across the tissue.

        Examples
        --------
        **1. Standard in-place normalization**
        
        >>> cube.normalize_tic(target_sum=1e4)
        >>> # The pre-normalization TIC is now saved in observations
        >>> print(cube.adata.obs["tic"].head())

        **2. Normalizing a specific layer and returning a copy**
        
        >>> normalized_cube = cube.normalize_tic(
        ...     layer="raw", 
        ...     target_sum=1e6, 
        ...     output_layer="tic_normalized",
        ...     copy=True
        ... )
        >>> # The original cube.X remains unchanged.
        """

        return tic_normalize_msicube(
            self,
            target_sum=target_sum,
            layer=layer,
            output_layer=output_layer,
            store_tic_in_obs=store_tic_in_obs,
            copy=copy,
        )

    def match_mzs_to_var_simple(
        self,
        mzs: Sequence[float],
        *,
        mz_col: str = "mz",
        mode: Literal["closest", "tolerance"] = "closest",
        tol: float = 5.0,
        tol_unit: Literal["ppm", "da"] = "ppm",
        return_all_within_tol: bool = True,
        assume_sorted: bool = False,
        annotation: Optional[Union[str, Sequence[Optional[str]]]] = None,
        annotation_col: Optional[str] = None,
        multi_write: Literal["overwrite", "append"] = "append",
        sep: str = ";",
    ) -> pd.DataFrame:
        """
        Match query m/z values to ``self.adata.var`` rows.

        This is a convenience wrapper around
        :func:`deltamsi.processing.mz_matching.match_mzs_to_var_simple` that
        operates directly on the MSICube instance.

        Parameters
        ----------
        mzs : sequence of float
            Query m/z values to match against ``adata.var[mz_col]``.
        mz_col : str, default "mz"
            Column name in ``adata.var`` storing m/z values.
        mode : {"closest", "tolerance"}, default "closest"
            Matching strategy.
        tol : float, default 5.0
            Tolerance value when ``mode="tolerance"``.
        tol_unit : {"ppm", "da"}, default "ppm"
            Units for the tolerance value.
        return_all_within_tol : bool, default True
            When using tolerance mode, return all matches within tolerance.
        assume_sorted : bool, default False
            If True, assumes ``adata.var[mz_col]`` is already sorted.
        annotation : str or sequence, optional
            Annotation(s) to write to matched variables.
        annotation_col : str, optional
            Column name in ``adata.var`` to write annotations into.
        multi_write : {"overwrite", "append"}, default "append"
            Behavior when multiple annotations map to the same variable.
        sep : str, default ";"
            Separator used for appending annotations.

        Returns
        -------
        pd.DataFrame
            A DataFrame with match results (one row per query m/z).
        """

        return match_mzs_to_var_simple_processing(
            self,
            mzs,
            mz_col=mz_col,
            mode=mode,
            tol=tol,
            tol_unit=tol_unit,
            return_all_within_tol=return_all_within_tol,
            assume_sorted=assume_sorted,
            annotation=annotation,
            annotation_col=annotation_col,
            multi_write=multi_write,
            sep=sep,
        )

    def plot_ion_images(
        self,
        mz: Union[float, str, Sequence[Union[float, str]]],
        samples: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Generate and display spatial heatmaps of specific ion intensities.

        This method provides a high-level wrapper to visualize the spatial 
        distribution of m/z features or aggregated labels across one or multiple 
        tissue sections.

        Parameters
        ----------
        mz : float, str, or list of (float or str)
            The feature(s) to visualize. Can be:
            
            * A single **float**: The m/z value (closest match in ``adata.var['mz']``).
            * A single **str**: A feature ID (e.g., 'mz_123.4567') or an aggregated 
            label name if using `obsm_key`.
            * A **list**: Multiple values to be plotted as a grid.
        samples : str or list of str, optional
            Specific sample name(s) to include in the plot. If None (default), 
            all samples found in ``adata.obs['sample']`` are displayed.
        **kwargs : Any
            Parameters passed to the underlying plotting engine. 
            Supported arguments include:

            * **layer** : *str, optional*
            The data layer to use for intensities (e.g., 'RAW', 'normalized').
            * **obsm_key** : *str, optional*
            If provided, looks for features in ``adata.obsm[obsm_key]`` instead 
            of the main matrix (used for aggregated ion images).
            * **cmap** : *str, default 'viridis'*
            The colormap used for intensity heatmaps.
            * **share_intensity_scale** : *bool, default True*
            If True, all samples for a given m/z will share the same min/max 
            intensity range, allowing for direct visual comparison.
            * **ncols** : *int, default 4*
            Number of columns in the subplot grid.
            * **percentile** : *float, optional*
            Contrast adjustment; clips intensities to the specified percentile 
            (e.g., 99.0 to remove hot pixels).

        Returns
        -------
        None
            Displays the generated Matplotlib figure.

        Raises
        ------
        ValueError
            If ``self.adata`` is empty or if the specified m/z value is not found.

        Examples
        --------
        **1. Plot a single ion across all samples**

        >>> cube.plot_ion_images(mz=885.55)

        **2. Plot multiple ions for a specific sample with contrast adjustment**

        >>> cube.plot_ion_images(
        ...     mz=[124.01, 301.2, 550.8], 
        ...     samples="Section_A", 
        ...     percentile=99.5,
        ...     cmap="magma"
        ... )

        **3. Plot aggregated metabolite images**

        >>> cube.plot_ion_images(
        ...     mz="Glucose", 
        ...     obsm_key="X_by_label", 
        ...     layer=None
        ... )
        """
        if self.adata is None:
            raise ValueError("AnnData is empty.")

        # Default to all samples if None provided
        if samples is None:
            samples = self.adata.obs['sample'].unique().tolist()

        plot_ion_images(self, mz=mz, samples=samples, **kwargs)

    def compute_spatial_chaos_scores(
        self,
        *,
        layer: Optional[str] = None,
        obsm_key: str = "spatial",
        sample_key: str = "sample",
        n_thresholds: int = 30,
        varm_key: str = "spatial_chaos",
    ) -> np.ndarray:
        """
        Compute the spatial chaos score for every ion image across MSI samples.

        Spatial chaos (also known as the "Chaoticness Score") measures the 
        structuredness of an ion's spatial distribution. A low score indicates 
        a highly structured distribution (likely biological), while a high score 
        suggests random noise or chemical background. It is calculated based 
        on the level-set complexity of the image.

        Parameters
        ----------
        layer : str, optional
            The specific layer in ``adata.layers`` to use for the computation. 
            If None, the main matrix ``adata.X`` is used.
        obsm_key : str, default 'spatial'
            The key in ``adata.obsm`` containing the pixel coordinates (x, y) 
            required to reconstruct the 2D images.
        sample_key : str, default 'sample'
            The key in ``adata.obs`` that defines the sample provenance. Chaos 
            scores are computed independently for each sample.
        n_thresholds : int, default 30
            The number of intensity thresholds used to calculate the level-set 
            objects. Higher values provide more precision but increase 
            computation time.
        varm_key : str, default 'spatial_chaos'
            The key used to store the resulting scores in ``adata.varm``.

        Returns
        -------
        numpy.ndarray
            A matrix of shape ``(n_features, n_samples)`` containing the chaos 
            scores for each ion across all samples.

        Raises
        ------
        ValueError
            If ``self.adata`` is empty or if the spatial coordinates are missing.

        Attributes Modified
        -------------------
        adata.varm['spatial_chaos'] : numpy.ndarray
            Stores the chaos score matrix.
        adata.uns['spatial_chaos'] : dict
            Metadata including sample names, threshold counts, and source layers.

        Notes
        -----
        This score is frequently used as a filter in MSI pipelines to remove 
        non-informative features (off-tissue signals or detector noise) 
        before downstream statistical analysis.

        Examples
        --------
        **1. Compute chaos scores with default settings**
        
        >>> scores = cube.compute_spatial_chaos_scores()
        >>> print(f"Chaos matrix shape: {scores.shape}")

        **2. Filtering ions based on a chaos threshold**
        
        If you want to keep only the most structured ions (e.g., score < 0.1):

        >>> cube.compute_spatial_chaos_scores(n_thresholds=50)
        >>> # Keep ions where the average chaos score across samples is low
        >>> mask = cube.adata.varm['spatial_chaos'].mean(axis=1) < 0.1
        >>> filtered_adata = cube.adata[:, mask].copy()
        """

        if self.adata is None:
            raise ValueError("AnnData is empty. Run data extraction first.")

        chaos, samples = compute_spatial_chaos_matrix(
            self.adata,
            layer=layer,
            obsm_key=obsm_key,
            sample_key=sample_key,
            n_thresholds=n_thresholds,
        )

        self.adata.varm[varm_key] = chaos
        self.adata.uns.setdefault("spatial_chaos", {})
        self.adata.uns["spatial_chaos"].update(
            {
                "samples": list(samples),
                "sample_key": sample_key,
                "obsm_key": obsm_key,
                "layer": layer,
                "n_thresholds": int(n_thresholds),
                "varm_key": varm_key,
            }
        )

        return chaos

    def compute_spatial_chaos_fold_change(
        self,
        *,
        groupby: str,
        control_label: Any,
        interaction_label: Any,
        varm_key: str = "spatial_chaos",
        result_key: str = "spatial_chaos_fold_change",
        eps: float = 1e-6,
    ) -> Dict[str, np.ndarray]:
        """
        Compute fold change of spatial chaos scores between two experimental groups.

        This method compares the spatial "structuredness" of ions across different 
        biological conditions (e.g., Healthy vs. Diseased). By calculating the 
        Fold Change (FC) of chaos scores, you can identify ions that undergo 
        significant spatial reorganization or loss of structure between groups.

        The Fold Change is calculated as:
        $$FC = \frac{max(S_{interaction}) + \epsilon}{max(S_{control}) + \epsilon}$$

        Parameters
        ----------
        groupby : str
            The column in ``adata.obs`` that contains group assignments (e.g., 'condition').
        control_label : Any
            The value in the `groupby` column representing the reference/control group.
        interaction_label : Any
            The value in the `groupby` column representing the treated/experimental group.
        varm_key : str, default 'spatial_chaos'
            The key in ``adata.varm`` where per-sample chaos scores are stored. 
            Requires :meth:`compute_spatial_chaos_scores` to have been run.
        result_key : str, default 'spatial_chaos_fold_change'
            Prefix used for storing results in ``adata.var`` and metadata in ``adata.uns``.
        eps : float, default 1e-6
            A small constant (epsilon) to avoid division by zero or log of zero errors.

        Returns
        -------
        Dict[str, numpy.ndarray]
            A dictionary containing:
            
            * ``"S_control_max"``: Max chaos score in the control group.
            * ``"S_interaction_max"``: Max chaos score in the interaction group.
            * ``"FC_S"``: The computed Fold Change.
            * ``"samples"``: List of sample names used.
            * ``"sample_groups"``: Group assignment for each sample.

        Raises
        ------
        ValueError
            If ``self.adata`` is empty or if the specified labels are not found 
            in the `groupby` column.
        KeyError
            If ``varm_key`` is missing from ``adata.varm``.

        Attributes Modified
        -------------------
        adata.var['{result_key}_FC_S'] : pandas.Series
            The computed Fold Change for each m/z feature.
        adata.uns['{result_key}'] : dict
            Metadata regarding the groups compared and epsilon used.

        Examples
        --------
        **1. Compare chaos between 'WildType' and 'Knockout'**

        >>> fc_results = cube.compute_spatial_chaos_fold_change(
        ...     groupby="condition",
        ...     control_label="WT",
        ...     interaction_label="KO"
        ... )
        >>> # Identify ions that become much more chaotic in the KO group
        >>> high_fc_ions = cube.adata.var.index[cube.adata.var["spatial_chaos_fold_change_FC_S"] > 2]
        """

        if self.adata is None:
            raise ValueError("AnnData is empty. Run data extraction first.")

        fold_change = spatial_chaos_fold_change_from_adata(
            self.adata,
            groupby=groupby,
            control_label=control_label,
            interaction_label=interaction_label,
            varm_key=varm_key,
            eps=eps,
        )

        self.adata.var[f"{result_key}_control_max"] = fold_change["S_control_max"]
        self.adata.var[f"{result_key}_interaction_max"] = fold_change[
            "S_interaction_max"
        ]
        self.adata.var[f"{result_key}_FC_S"] = fold_change["FC_S"]

        self.adata.uns[result_key] = {
            "groupby": groupby,
            "control_label": control_label,
            "interaction_label": interaction_label,
            "varm_key": varm_key,
            "eps": float(eps),
            "samples": fold_change["samples"].tolist(),
            "sample_groups": fold_change["sample_groups"].tolist(),
        }

        return fold_change

    def plot_peak_windows(
        self,
        peak_mzs: Sequence[float],
        labels: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot zoomed m/z windows around specified peaks to compare mean spectra across samples.

        This method generates a grid of subplots centered on specific m/z values. 
        It overlays the mean spectra of selected samples, allowing for a direct 
        visual inspection of peak alignment, resolution, and relative intensity 
        differences across the dataset.

        Parameters
        ----------
        peak_mzs : sequence of float
            The list of m/z values to center the windows on. Each value in this 
            list will generate its own subplot.
        labels : sequence of str, optional
            The specific sample names to include in the overlay. These names must 
            exist as keys in ``self.adata.uns['mean_spectra']``. If None, all 
            samples found in the metadata are plotted.
        **kwargs : Any
            Additional keyword arguments to customize the visualization:

            * **span_da** : *float, default 1.0*
            The total width of the x-axis (m/z) for each subplot in Daltons. 
            Use a smaller value (e.g., 0.2) for high-resolution data.
            * **tol_da** : *float, optional*
            If provided, draws grey dotted vertical lines at 
            ``peak_mz ± tol_da`` to represent absolute extraction windows.
            * **tol_ppm** : *float, optional*
            If provided, draws grey dotted vertical lines at 
            ``peak_mz ± tol_ppm`` to represent relative extraction windows.
            * **ncols** : *int, default 3*
            Number of subplot columns in the resulting figure.
            * **figsize** : *tuple of float, optional*
            The dimensions of the figure in inches (width, height).

        Returns
        -------
        None
            Displays the generated Matplotlib figure.

        Raises
        ------
        ValueError
            If ``self.adata.uns['mean_spectra']`` is missing. Run 
            :meth:`compute_all_mean_spectra` first.
        KeyError
            If any provided `labels` do not exist in the mean spectra metadata.

        Notes
        -----
        In each plot:
        
        - A **red dashed line** marks the exact center (``peak_mz``).
        - **Grey dotted lines** mark the tolerance boundaries (if `tol_da` or 
        `tol_ppm` are set).
        - Different samples are distinguished by color and identified in a legend.

        Examples
        --------
        **1. Compare two specific samples around a single peak**

        >>> cube.plot_peak_windows(
        ...     peak_mzs=[885.55], 
        ...     labels=["Control_01", "Treated_01"],
        ...     span_da=0.5
        ... )

        **2. Inspect multiple peaks with a high-resolution window (ppm)**

        >>> # Useful for checking if peak picking was accurate
        >>> cube.plot_peak_windows(
        ...     peak_mzs=[184.07, 760.58, 885.55],
        ...     span_da=0.2,
        ...     tol_ppm=10.0,
        ...     ncols=1
        ... )

        **3. Fast overview of all samples**

        >>> cube.plot_peak_windows(peak_mzs=[550.8])
        """
        plot_mean_spectrum_windows(self, peak_mzs, labels, **kwargs)

    def recalibrate(
        self,
        database_mass_file: str,
        options: RecalibrationOptions,
        output_directory: Optional[str] = None,
        n_workers: int = 1,
    ) -> None:
        """
        Perform mass recalibration on all raw imzML files using a reference database.

        This method corrects systematic mass shifts (mass errors) across all spectra 
        in your datasets. It matches observed peaks against a database of known 
        theoretical masses and uses a robust alignment algorithm (RANSAC/KDE) to 
        calculate the required correction. The method generates new ``.imzML`` and 
        ``.ibd`` files and updates the MSICube to point to these corrected files.

        Parameters
        ----------
        database_mass_file : str
            Path to a text or CSV file containing the list of theoretical 
            reference masses (one per line).
        options : RecalibrationOptions
            A configuration object defining the recalibration strategy. 
            Attributes typically include:

            * **tol_da** : *float*
            Matching tolerance in Daltons for finding reference peaks.
            * **kde_bw_da** : *float*
            Kernel Density Estimation bandwidth for shift estimation.
            * **roi_halfwidth_da** : *float*
            The window size around reference masses to extract signal.
            * **n_peaks** : *int*
            Minimum number of matched peaks required to perform recalibration.
        output_directory : str, optional
            Directory where the new, recalibrated ``.imzML`` and ``.ibd`` files 
            will be saved. If None, creates a ``recalibrated_data`` folder 
            inside the current data directory.
        n_workers : int, default 1
            Number of CPU cores to use for parallel processing. Increasing this 
            significantly speeds up recalibration for multiple samples.

        Returns
        -------
        None
            Updates ``self.org_imzml_path_dict`` and ``self.data_directory`` 
            to point to the newly created files.

        Raises
        ------
        FileNotFoundError
            If the ``database_mass_file`` cannot be found.
        RuntimeError
            If recalibration fails for critical reasons during file writing.

        Notes
        -----
        **Data Integrity:** This process creates physical copies of your data. 
        Ensure you have enough disk space available (equivalent to the size 
        of your original raw datasets). 
        
        Recalibration is highly recommended before performing global peak picking 
        to ensure that the same ion from different samples aligns perfectly on 
        the m/z axis.

        Examples
        --------
        **1. Recalibrate using a standard metabolite database**

        >>> from msicube import RecalibrationOptions
        >>> my_options = RecalibrationOptions(
        ...     tol_da=0.01, 
        ...     n_peaks=20
        ... )
        >>> cube.recalibrate(
        ...     database_mass_file="ref_masses.txt",
        ...     options=my_options,
        ...     n_workers=4
        ... )
        INFO: Recalibration successful for 4 samples.
        INFO: MSICube sample paths updated to use the recalibrated files.

        **2. Specifying a custom output path**

        >>> cube.recalibrate(
        ...     database_mass_file="lipids.csv",
        ...     options=my_options,
        ...     output_directory="/external_drive/project_recalibrated"
        ... )
        """
        if not self.org_imzml_path_dict:
            logger.error("No imzML files loaded for recalibration.")
            return

        # 1. Préparation de la base de données et des paramètres
        try:
            db_masses_sorted = load_database_masses(database_mass_file)
        except Exception as e:
            logger.error(
                f"Failed to load database masses from {database_mass_file}: {e}"
            )
            return

        # Convertir RecalibrationOptions en RecalParams pour le core
        recal_params = RecalParams(
            tol_da=options.tol_da,  # Utilisation de l'ancienne tol pour tol_da par défaut
            kde_bw_da=options.kde_bw_da,
            roi_halfwidth_da=options.roi_halfwidth_da,
            n_peaks=options.n_peaks,
            # Le reste des options RANSAC est laissé par défaut ou doit être ajouté à RecalibrationOptions
        )

        # Définition du répertoire de sortie
        if output_directory is None:
            output_directory = os.path.join(self.data_directory, "recalibrated_data")
        os.makedirs(output_directory, exist_ok=True)

        logger.info(
            f"Starting recalibration for {len(self.org_imzml_path_dict)} samples..."
        )
        logger.info(f"Output files will be saved in: {output_directory}")

        # 2. Fonction de travail pour l'exécution parallèle
        def _recalibrate_one_sample(
            sample_name: str, input_path: str
        ) -> Tuple[str, bool]:
            imzml_filename = os.path.basename(input_path)
            output_path = os.path.join(output_directory, imzml_filename)

            try:
                write_corrected_msi(
                    imzml_path=input_path,
                    out_imzml_path=output_path,
                    db_masses_sorted=db_masses_sorted,
                    params=recal_params,
                )
                logger.info(f"Recalibration successful for {sample_name}.")
                return sample_name, True
            except Exception as e:
                logger.error(
                    f"Recalibration failed for sample {sample_name} ({input_path}): {e}"
                )
                return sample_name, False

        # 3. Exécution (parallèle si n_workers > 1)
        tasks = [(name, path) for name, path in self.org_imzml_path_dict.items()]

        results: List[Tuple[str, bool]] = []
        if n_workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(_recalibrate_one_sample, name, path)
                    for name, path in tasks
                ]
                results = [f.result() for f in futures]
        else:
            for name, path in tasks:
                results.append(_recalibrate_one_sample(name, path))

        # 4. Summary and MSICube Update
        successful_recalibrations = [name for name, success in results if success]
        failed_recalibrations = [name for name, success in results if not success]

        if successful_recalibrations:
            logger.info(
                f"Recalibration successful for {len(successful_recalibrations)} samples."
            )

            # Update MSICube paths to point to the new calibrated files
            new_org_imzml_path_dict = {}
            for sample_name, input_path in self.org_imzml_path_dict.items():
                if sample_name in successful_recalibrations:
                    imzml_filename = os.path.basename(input_path)
                    new_org_imzml_path_dict[sample_name] = os.path.join(
                        output_directory, imzml_filename
                    )
                else:
                    new_org_imzml_path_dict[sample_name] = (
                        input_path  # Keep old path for failed/unprocessed samples
                    )

            self.org_imzml_path_dict = new_org_imzml_path_dict
            self.data_directory = output_directory  # Update primary data directory
            logger.info("MSICube sample paths updated to use the recalibrated files.")

        if failed_recalibrations:
            logger.error(
                f"Recalibration failed for {len(failed_recalibrations)} samples: {', '.join(failed_recalibrations)}"
            )

    def plot_recalibration(
        self,
        sample_name: str,
        database_mass_file: str,
        options: RecalibrationOptions,
        pixel_idx: Optional[Sequence[int]] = None,
        pixel_coord: Optional[Sequence[str]] = None,
        n_random: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """
        Visualize recalibration quality for selected pixels of a specific sample.

        This method generates diagnostic plots to evaluate the accuracy of the 
        mass shift correction. It displays the Kernel Density Estimation (KDE) 
        of the mass errors and the RANSAC regression used to align observed 
        peaks with theoretical database masses. This is essential for verifying 
        that recalibration parameters are appropriate before processing an 
        entire project.

        Parameters
        ----------
        sample_name : str
            The identifier of the sample to analyze (must exist in 
            ``self.org_imzml_path_dict``).
        database_mass_file : str
            Path to the text/CSV file containing reference theoretical masses.
        options : RecalibrationOptions
            The configuration object used for the recalibration logic (defines 
            tolerances and peak requirements).
        pixel_idx : sequence of int, optional
            Specific 1D indices of pixels to plot.
        pixel_coord : sequence of str, optional
            Specific coordinates as strings (e.g., ``['10,20', '15,30']``) to plot.
        n_random : int, optional
            If provided, randomly selects this number of pixels from the sample 
            to visualize general quality.
        seed : int, default 42
            Random seed for reproducible pixel selection when using `n_random`.

        Returns
        -------
        None
            Displays a Matplotlib figure for each selected pixel.

        Raises
        ------
        ValueError
            If the sample name is not recognized or if no pixels are selected.

        Notes
        -----
        The diagnostic plot typically contains two main panels:
        
        1. **KDE Plot**: Shows the distribution of mass errors ($m/z_{obs} - m/z_{th}$). 
        A successful recalibration shows a clear, narrow peak.
        2. **Regression Plot**: Shows the mass shift as a function of $m/z$. 
        The RANSAC algorithm fits a line through valid matches while 
        ignoring outliers.

        Examples
        --------
        **1. Inspecting specific coordinates for quality control**

        >>> from msicube import RecalibrationOptions
        >>> options = RecalibrationOptions(tol_da=0.01, n_peaks=15)
        >>> cube.plot_recalibration(
        ...     sample_name="Sample_01",
        ...     database_mass_file="calibrants.txt",
        ...     options=options,
        ...     pixel_coord=["50,50", "100,100"]
        ... )

        **2. Checking random pixels across a dataset**

        >>> cube.plot_recalibration(
        ...     sample_name="Sample_01",
        ...     database_mass_file="calibrants.txt",
        ...     options=options,
        ...     n_random=3
        ... )
        """
        if sample_name not in self.org_imzml_path_dict:
            available = list(self.org_imzml_path_dict.keys())
            logger.error(
                f"Sample '{sample_name}' not found. Available samples: {available}"
            )
            return

        imzml_path = self.org_imzml_path_dict[sample_name]

        # Convertir RecalibrationOptions en RecalParams
        params = RecalParams(
            tol_da=options.tol_da,
            kde_bw_da=options.kde_bw_da,
            roi_halfwidth_da=options.roi_halfwidth_da,
            n_peaks=options.n_peaks,
        )

        try:
            db = load_database_masses(database_mass_file)
            p = ImzMLParser(imzml_path, parse_lib="ElementTree")

            # 1. Sélection des pixels
            pixels_to_plot = select_pixels(
                p,
                pixel_idx=pixel_idx,
                pixel_coord=pixel_coord,
                n_random=n_random,
                seed=seed,
            )

            if not pixels_to_plot:
                logger.warning("No pixels selected for plotting.")
                return

            # 2. Création et affichage des figures
            for idx in pixels_to_plot:
                # diagnostics_for_pixel retourne une figure
                fig = diagnostics_for_pixel(p, idx, db, params)
                fig.suptitle(
                    f"Sample: {sample_name} | Pixel Index: {idx} | Coordinates: {p.coordinates[idx]}",
                    fontsize=16,
                )

            plt.show()

        except Exception as e:
            logger.error(
                f"Error during recalibration diagnostics plotting for {sample_name}: {e}"
            )
            return

    def cluster_masses(
        self,
        candidates_df: Optional[pd.DataFrame] = None,
        options: Optional[MassClusteringOptions] = None,
        keep_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Group m/z peaks into chemical families using graph-based clustering.

        This method identifies relationships between features in ``adata.var`` based 
        on either structural mass differences (e.g., specific chemical 
        modifications, adducts) or spatial colocalization. It builds a Mass 
        Difference Graph (MDG) and applies the Leiden community detection 
        algorithm to define clusters.

        Parameters
        ----------
        candidates_df : pandas.DataFrame, optional
            A catalog of theoretical mass differences (e.g., delta m/z, labels, 
            and scores). This is **required** if ``options.method='candidates'``.
        options : MassClusteringOptions, optional
            Configuration object for the clustering algorithm. Key attributes:
            
            * **method** : {'candidates', 'colocalization'}
            Whether to cluster based on chemical distances or spatial similarity.
            * **resolution** : float, default 1.0
            Leiden algorithm resolution. Higher values lead to more, smaller clusters.
            * **knn_k** : int, optional
            Number of nearest neighbors for graph pruning.
        keep_mask : numpy.ndarray, optional
            A boolean adjacency matrix (N x N) used to strictly allow or 
            forbid specific edges between m/z peaks.

        Returns
        -------
        None
            The results are stored in-place within the ``self.adata`` object.

        Attributes Modified
        -------------------
        adata.var[options.output_col] : pandas.Series
            Cluster assignment for each m/z peak. Values of -1 indicate noise/unclustered.
        adata.varp['mass_clustering_weights'] : scipy.sparse.csr_matrix
            The weighted adjacency matrix of the Mass Difference Graph.
        adata.varp['mass_clustering_connectivities'] : scipy.sparse.csr_matrix
            Binary connectivity matrix (edges) between m/z peaks.
        adata.var['mdg_degree'] : numpy.ndarray
            The number of connections (edges) for each m/z feature in the graph.
        adata.uns['mass_clustering'] : dict
            Summary statistics (number of clusters, compression ratio) and 
            the full list of edges with labels.

        Raises
        ------
        ValueError
            If ``self.adata`` or ``adata.var['mz']`` are missing, or if 
            ``candidates_df`` is missing when required.
        KeyError
            If the colocalization matrix is missing from ``adata.varp`` when 
            using the 'colocalization' method.

        Notes
        -----
        The clustering process follows these steps:
        
        1. **Edge Creation**: Linking ions $i$ and $j$ if $|mz_i - mz_j|$ matches 
        a known delta (like $CH_2$ or $H_2O$) or if they are spatially correlated.
        2. **Graph Weighting**: Assigning scores to edges based on mass accuracy or 
        colocalization coefficients.
        3. **Community Detection**: Partitioning the graph into dense sub-networks 
        representing chemical families (e.g., all phospholipids of a specific class).

        Examples
        --------
        **1. Cluster based on a chemical catalog (Candidates)**
        
        >>> # catalog_df contains columns: 'delta_m', 'label', 'score'
        >>> cube.cluster_masses(candidates_df=catalog_df)
        >>> print(cube.adata.var['mass_cluster'].value_counts())

        **2. Cluster based on spatial colocalization**
        
        >>> from msicube import MassClusteringOptions
        >>> opts = MassClusteringOptions(method="colocalization", resolution=0.8)
        >>> cube.cluster_masses(options=opts)
        """
        if self.adata is None:
            raise ValueError(
                "AnnData object is empty. Run peak picking first."
            )

        opts = options or MassClusteringOptions()
        opts.validate()

        if "mz" not in self.adata.var:
            raise ValueError(
                "Column 'mz' is missing from adata.var. Peak picking required."
            )

        masses = self.adata.var["mz"].values

        logger.info(
            f"Starting clustering on {len(masses)} masses (Resolution: {opts.resolution}, Method: {opts.method})..."
        )

        if opts.method == "colocalization":
            if opts.coloc_varp_key is None:
                raise ValueError("coloc_varp_key must be set for colocalization clustering.")
            if opts.coloc_varp_key not in self.adata.varp:
                raise KeyError(
                    f"'{opts.coloc_varp_key}' is absent from adata.varp. "
                    "Run compute_cosine_colocalization or adjust coloc_varp_key."
                )

            coloc_matrix = self.adata.varp[opts.coloc_varp_key]
            res = cluster_masses_from_colocalization(
                coloc_matrix=coloc_matrix,
                keep_mask=keep_mask,
                resolution=opts.resolution,
                edge_max_delta_cosine=opts.edge_max_delta_cosine,
                knn_k=opts.knn_k,
                knn_mode=opts.knn_mode,
                return_graph=opts.return_graph,
            )
        else:
            if candidates_df is None:
                raise ValueError("candidates_df is required when method='candidates'.")
            res = cluster_masses_with_candidates(
                masses=masses,
                candidates_df=candidates_df,
                delta_col=opts.delta_col,
                score_col=opts.score_col,
                label_col=opts.label_col,
                tol=opts.get_tol_param(),
                edge_max_delta_m=opts.edge_max_delta_m,
                keep_mask=keep_mask,
                resolution=opts.resolution,
                weight_transform=opts.weight_transform,
                weight_kwargs=opts.weight_kwargs,
                knn_k=opts.knn_k,
                knn_mode=opts.knn_mode,
                return_graph=opts.return_graph,
            )

        self.adata.var[opts.output_col] = res["labels"]

        df_edges = res["edges"]
        n_vars = self.adata.n_vars

        weight_matrix = sp.coo_matrix(
            (
                df_edges["weight"].to_numpy(),
                (df_edges["i"].to_numpy(), df_edges["j"].to_numpy()),
            ),
            shape=(n_vars, n_vars),
        )
        weight_matrix = (weight_matrix + weight_matrix.T).tocsr()

        connectivities = (weight_matrix > 0).astype(np.float32)

        self.adata.varp["mass_clustering_weights"] = weight_matrix
        self.adata.varp["mass_clustering_connectivities"] = connectivities

        deg = np.asarray((connectivities > 0).sum(axis=1)).ravel()
        wdeg = np.asarray(weight_matrix.sum(axis=1)).ravel()
        self.adata.var["mdg_degree"] = deg
        self.adata.var["mdg_wdegree"] = wdeg

        edges_with_names = df_edges.copy()
        edges_with_names["u"] = self.adata.var_names.to_numpy()[
            edges_with_names["i"].to_numpy()
        ]
        edges_with_names["v"] = self.adata.var_names.to_numpy()[
            edges_with_names["j"].to_numpy()
        ]

        self.adata.uns["mass_clustering"] = {
            "n_clusters": res["n_clusters"],
            "n_minus1": res["n_minus1"],
            "compression": res["compression"],
            "edges": edges_with_names,
            "options": opts,
        }

        logger.info(f"Clustering complete: {res['n_clusters']} clusters found.")

    def compute_kendrick(
        self,
        *,
        mz_key: str = "mz",
        base: Union[str, float, Tuple[float, float]] = "CH2",
        kmd_mode: Literal["fraction", "defect"] = "fraction",
        varm_key: Optional[str] = None,
        store_1d_in_var: bool = False,
        var_prefix: str = "kendrick",
    ) -> str:
        """
        Calculate and store Kendrick Mass coordinates in the AnnData object.

        Kendrick Mass (KM) analysis is a data transformation technique used to 
        identify homologous series (compounds differing only by a repeating unit, 
        such as $CH_2$). In a Kendrick Mass Defect (KMD) plot, members of the same 
        series align horizontally, making it easier to identify lipid classes or 
        polymer distributions in complex MSI datasets.

        The transformation follows:
        $$KM = m/z_{obs} \times \frac{M_{nominal}}{M_{exact}}$$
        $$KMD = KM - \text{round}(KM)$$

        Parameters
        ----------
        mz_key : str, default 'mz'
            The column name in ``adata.var`` containing the m/z values.
        base : str, float, or tuple, default 'CH2'
            The repeating unit used for the Kendrick scale.
            
            * **str**: A chemical formula (e.g., "CH2", "H2O", "C2H4O").
            * **float**: The exact mass of the unit.
            * **tuple**: A pair of (exact_mass, nominal_mass).
        kmd_mode : {'fraction', 'defect'}, default 'fraction'
            Determines how the mass defect is calculated:
            
            * 'fraction': The fractional part of the Kendrick mass.
            * 'defect': The standard mass defect ($KM - \text{floor}(KM)$).
        varm_key : str, optional
            Key used to store the 2D coordinates (KM, KMD) in ``adata.varm``. 
            If None, it is automatically generated (e.g., 'kendrick_CH2').
        store_1d_in_var : bool, default False
            If True, also stores Kendrick Mass and Kendrick Mass Defect as 
            individual columns in ``adata.var``.
        var_prefix : str, default 'kendrick'
            Prefix for the column names when ``store_1d_in_var`` is True.

        Returns
        -------
        str
            The ``varm`` key where the coordinates were stored.

        Raises
        ------
        ValueError
            If ``self.adata`` is empty or if the chemical formula is invalid.

        Attributes Modified
        -------------------
        adata.varm['{varm_key}'] : numpy.ndarray
            An array of shape ``(n_features, 2)`` containing [KM, KMD].
        adata.var['{prefix}_km'] : pandas.Series (Optional)
            Stored if ``store_1d_in_var=True``.
        adata.var['{prefix}_kmd'] : pandas.Series (Optional)
            Stored if ``store_1d_in_var=True``.

        Notes
        -----
        Standard Kendrick analysis typically uses $CH_2$ ($14.01565$ Da) as the 
        base. However, in lipidomics, using other bases like oxygen (O) or 
        polyethylene glycol (PEG) units can reveal different structural 
        relationships.

        Examples
        --------
        **1. Standard CH2 Kendrick analysis**
        
        >>> key = cube.compute_kendrick(base="CH2")
        >>> # Coordinates are now in cube.adata.varm['kendrick_CH2']

        **2. Analysis with a custom base and 1D storage**
        
        >>> cube.compute_kendrick(
        ...     base="H2O", 
        ...     store_1d_in_var=True, 
        ...     var_prefix="water"
        ... )
        >>> # Access results directly
        >>> km_values = cube.adata.var["water_km"]

        **3. Plotting the results**
        
        >>> import matplotlib.pyplot as plt
        >>> coords = cube.adata.varm["kendrick_CH2"]
        >>> plt.scatter(coords[:, 0], coords[:, 1], s=1)
        >>> plt.xlabel("Kendrick Mass")
        >>> plt.ylabel("Kendrick Mass Defect")
        """

        if self.adata is None:
            raise ValueError("AnnData object is empty. Cannot compute Kendrick coordinates.")

        return compute_kendrick_varm(
            self.adata,
            mz_key=mz_key,
            base=base,
            kmd_mode=kmd_mode,
            varm_key=varm_key,
            store_1d_in_var=store_1d_in_var,
            var_prefix=var_prefix,
        )

    def direct_mass_neighbors(
        self,
        mz_id: Union[int, str],
        *,
        mass_uns_key: str = "mass_clustering",
        edges_key: str = "edges",
        cosine_key: str = "ion_cosine",
        var_cols: Optional[Sequence[str]] = None,
        edge_cols: Optional[Sequence[str]] = None,
        include_src_cols: bool = False,
    ) -> pd.DataFrame:
        """
        Return direct neighbors (distance=1) of mz_id in the mass-difference network.

        Output: one row per edge incident to src, containing neighbor + edge info
        + cosine(src, nbr).
        """
        if self.adata is None:
            raise ValueError("AnnData object is empty. Load or compute data first.")

        return direct_mass_neighbors(
            self.adata,
            mz_id,
            mass_uns_key=mass_uns_key,
            edges_key=edges_key,
            cosine_key=cosine_key,
            var_cols=var_cols,
            edge_cols=edge_cols,
            include_src_cols=include_src_cols,
        )

    def annotate_kendrick(
        self,
        *,
        varm_key: str,
        label_key: str = "manual_label",
        default_label: str = "unlabeled",
        mz_key: Optional[str] = "mz",
        coord_cols: Tuple[int, int] = (0, 1),
        dragmode: str = "lasso",
        point_size: int = 6,
        height: int = 650,
        max_points_warn: int = 120_000,
    ):
        """
        Launch an interactive widget to manually label m/z features in Kendrick space.

        This method opens an interactive Plotly-based interface (within a Jupyter 
        notebook environment) that allows users to select groups of points in a 
        Kendrick Mass Defect plot using lasso or box selection tools. It is 
        specifically designed to help researchers manually annotate homologous 
        series or chemical families.

        Parameters
        ----------
        varm_key : str
            The key in ``self.adata.varm`` containing the Kendrick coordinates 
            calculated by :meth:`compute_kendrick`.
        label_key : str, default 'manual_label'
            The column name in ``self.adata.var`` where the assigned labels 
            will be stored.
        default_label : str, default 'unlabeled'
            The initial label assigned to all points before manual selection.
        mz_key : str, optional, default 'mz'
            The column in ``adata.var`` to show in the hover tooltip for 
            identification.
        coord_cols : tuple of int, default (0, 1)
            Indices of the columns in ``varm[varm_key]`` to use for the X and Y 
            axes (usually 0 for Kendrick Mass and 1 for Mass Defect).
        dragmode : {'lasso', 'select', 'pan', 'zoom'}, default 'lasso'
            The active selection tool when the widget starts. Lasso is 
            recommended for following diagonal or horizontal series.
        point_size : int, default 6
            Visual size of the markers in the plot.
        height : int, default 650
            The vertical height of the interactive widget in pixels.
        max_points_warn : int, default 120,000
            Threshold above which a warning is issued regarding potential 
            performance lag with Plotly.

        Returns
        -------
        tuple
            A tuple containing ``(ui, state)``:
            
            * **ui**: The IPyWidgets VBox containing the interface.
            * **state**: An object used to synchronize and apply labels to 
            the AnnData object.

        Raises
        ------
        ImportError
            If ``plotly`` or ``ipywidgets`` are not installed.
        ValueError
            If ``self.adata`` is empty or if ``varm_key`` is not found.

        Notes
        -----
        This method requires a live kernel (Jupyter Notebook or Lab) to function. 
        Labels applied in the UI are updated in real-time or upon clicking 
        an "Apply" button within the widget, depending on the implementation.

        Examples
        --------
        **1. Interactive labeling of CH2 series**
        
        >>> # First, compute the coordinates
        >>> cube.compute_kendrick(base="CH2", varm_key="k_ch2")
        >>> # Launch the widget
        >>> ui, state = cube.annotate_kendrick(varm_key="k_ch2", label_key="lipid_family")
        >>> ui  # Display the widget in the notebook cell
        """

        from deltamsi.plotting.kendrick_manual_label import manual_label_vars_from_kendrick

        return manual_label_vars_from_kendrick(
            self,
            varm_key=varm_key,
            label_key=label_key,
            default_label=default_label,
            mz_key=mz_key,
            coord_cols=coord_cols,
            dragmode=dragmode,
            point_size=point_size,
            height=height,
            max_points_warn=max_points_warn,
        )

    def plot_kendrick(
        self, options: Optional[KendrickPlotOptions] = None, **kwargs: Any
    ) -> Tuple[plt.Figure, Union[List[plt.Axes], plt.Axes], pd.DataFrame]:
        """
        Generate a Kendrick Mass Defect (KMD) plot colored by clustering results.

        This visualization maps m/z features into Kendrick space (Kendrick Mass 
        vs. Kendrick Mass Defect) and highlights the chemical families identified 
        during the ``cluster_masses`` step. It is a powerful tool for structural 
        annotation, as homologous series (e.g., lipids differing by CH2 units) 
        form horizontal alignments.

        Parameters
        ----------
        options : KendrickPlotOptions, optional
            A configuration object defining the plot aesthetics and data filtering. 
            If None, default settings are used.
        **kwargs : Any
            Key-value pairs to override specific options in ``KendrickPlotOptions``.
            Common overrides include:

            * **base** : *str or float, default 'CH2'*
            The Kendrick base used for the transformation.
            * **x_axis** : *{'mz', 'km'}, default 'mz'*
            Whether to plot Observed m/z or Kendrick Mass on the X-axis.
            * **point_size** : *float, default 10.0*
            Size of the markers in the scatter plot.
            * **label_col** : *str, optional*
            Column in ``adata.var`` used to color the family/label panel.
            When ``two_panels=False``, the plot uses this column for the single
            panel coloring.
            * **top_k_clusters** : *int, default 10*
            Only color the K largest clusters to maintain visual clarity.
            * **two_panels** : *bool, default False*
            If True, displays a side-by-side plot of the whole dataset 
            versus the filtered/selected clusters.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object.
        axes : matplotlib.axes.Axes or list of Axes
            The axes containing the plot(s).
        df : pandas.DataFrame
            The processed data used for the plot, including calculated Kendrick 
            coordinates and cluster assignments.

        Raises
        ------
        ValueError
            If ``self.adata`` is empty or if no clustering results (``mass_cluster``) 
            are found in ``adata.var``.

        Notes
        -----
        The plot automatically handles:
        
        - **Noise Filtering**: Cluster index ``-1`` is usually treated as 
        background noise and plotted with reduced opacity or in grey.
        - **Homologous Alignment**: Horizontal grid lines (``hgrid_step``) can be 
        added to guide the eye along m/z series.
        - **Annotations**: The most intense or representative ions of each cluster 
        can be automatically labeled.

        Examples
        --------
        **1. Standard Kendrick plot for the top 5 clusters**

        >>> fig, ax, df = cube.plot_kendrick(top_k_clusters=5, point_size=15)

        **2. High-resolution view using H2O base and two panels**

        >>> from msicube import KendrickPlotOptions
        >>> opts = KendrickPlotOptions(base="H2O", two_panels=True, annotate=True)
        >>> cube.plot_kendrick(options=opts)
        """
        # 1. Preparation of options
        if options is None:
            options = KendrickPlotOptions()

        # Potential override via kwargs
        for key, value in kwargs.items():
            if hasattr(options, key):
                setattr(options, key, value)

        options.validate()

        # 2. Data retrieval from AnnData
        if self.adata is None:
            raise ValueError("AnnData object is empty.")

        if "mass_cluster" not in self.adata.var:
            if options.label_col is not None and not options.two_panels:
                clustering_labels = np.zeros(self.adata.n_vars, dtype=int)
            else:
                raise ValueError(
                    "No clustering results found in adata.var. "
                    "Please run 'cluster_masses()' first, or provide label_col with "
                    "two_panels=False."
                )
        else:
            clustering_labels = self.adata.var["mass_cluster"].values

        # Retrieval of masses
        if options.mass_col not in self.adata.var:
            raise ValueError(
                f"Column '{options.mass_col}' is missing from adata.var. Peak picking required."
            )
        masses = self.adata.var[options.mass_col].values

        # Reconstruction of clustering_result dictionary
        clustering_result = {"labels": clustering_labels}

        # Optional family/label retrieval
        family = None
        if options.label_col is not None:
            if options.label_col not in self.adata.var.columns:
                raise ValueError(
                    f"Column '{options.label_col}' is missing from adata.var."
                )
            family = self.adata.var[options.label_col].values
        elif "family" in self.adata.var.columns:
            family = self.adata.var["family"].values
        elif "label" in self.adata.var.columns:
            family = self.adata.var["label"].values

        # 3. Call rendering function
        primary_color_by = (
            "family" if options.label_col is not None and not options.two_panels else "cluster"
        )
        return plot_kendrick_from_clustering(
            masses=masses,
            clustering_result=clustering_result,
            adata=self.adata,
            kendrick_varm_key=options.kendrick_varm_key,
            family=family,
            primary_color_by=primary_color_by,
            base=options.base,
            mass_col=options.mass_col,
            x_axis=options.x_axis,
            kmd_mode=options.kmd_mode,
            point_size=options.point_size,
            alpha=options.alpha,
            hgrid_step=options.hgrid_step,
            jitter=options.jitter,
            annotate=options.annotate,
            max_ann_per_group=options.max_ann_per_group,
            top_k_clusters=options.top_k_clusters,
            selected_clusters=options.selected_clusters,
            include_minus1_in_top=options.include_minus1_in_top,
            min_cluster_size=options.min_cluster_size,
            two_panels=options.two_panels,
            figsize=options.figsize,
        )
