# class MSICube

from concurrent.futures import ProcessPoolExecutor
import os
import re
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Optional, Dict, Any, List, Literal, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from pyimzml.ImzMLParser import ImzMLParser

from pymsix.plotting.ion_images import plot_ion_images
from pymsix.plotting.spectrum import plot_mean_spectrum_windows
from pymsix.processing.mean_spectrum import compute_mean_spectrum
from pymsix.processing.combine_mean_spectra import combine_mean_spectra, Spectrum
from pymsix.processing.peak_picking import peak_picking, extract_peak_matrix
from pymsix.processing.aggregation import aggregate_vars_by_label, Agg
from pymsix.processing.normalization import tic_normalize_msicube
from pymsix.processing.colocalization import (
    CosineColocParams,
    compute_mz_cosine_colocalization,
)

from pymsix.processing.recalibration_core import (
    load_database_masses,
    RecalParams,
)

from pymsix.processing.recalibration_cli_clean import write_corrected_msi
from pymsix.processing.recal_visu_clean import diagnostics_for_pixel, select_pixels

from pymsix.processing.mass_clustering import cluster_masses_with_candidates
from pymsix.processing.kendrick import compute_kendrick_varm
from pymsix.plotting.plot_kendrick_cluster_mz import plot_kendrick_from_clustering

from pymsix.params.options import (
    MeanSpectrumOptions,
    GlobalMeanSpectrumOptions,
    PeakPickingOptions,
    PeakMatrixOptions,
    RecalibrationOptions,
    MassClusteringOptions,
    KendrickPlotOptions,
)


class Logger:
    def info(self, msg: str) -> None:
        print(f"INFO: {msg}")

    def warning(self, msg: str) -> None:
        print(f"WARNING: {msg}")

    def error(self, msg: str) -> None:
        print(f"ERROR: {msg}")


logger = Logger()


LowAction = Literal["keep", "nan", "zero", "clip"]
HighAction = Literal["keep", "nan", "clip"]
def _log1p_inplace_or_copy(X: Any, *, base: Optional[float] = None) -> Any:
    """
    Apply ``log1p`` to a dense or sparse matrix, mirroring Scanpy's behavior.

    Sparse matrices are copied before mutation; dense inputs are modified in-place when
    possible. If ``base`` is provided, intensities are scaled accordingly.
    """

    if sp.issparse(X):
        X = X.copy()
        X.data = np.log1p(X.data)
        if base is not None:
            X.data /= np.log(base)
        return X

    X_arr = np.asarray(X)
    if not np.issubdtype(X_arr.dtype, np.floating):
        X_arr = X_arr.astype(np.float32, copy=False)

    np.log1p(X_arr, out=X_arr)
    if base is not None:
        X_arr /= np.log(base)
    return X_arr


class MSICube:
    """
    Main object for Mass Spectrometry Imaging (MSI) analysis.
    Contains an anndata object for data and a mapping of raw files.
    """

    data_directory: str
    adata: Optional[ad.AnnData]

    def __init__(self, data_directory: str) -> None:
        """
        Initializes the MSICube object by scanning the directory for imzML files.

        :param data_directory: Path to the directory containing imzML files.
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
        Build the default path for persisting AnnData objects based on the data directory.

        Parameters
        ----------
        file_format : {"h5ad", "zarr"}
            The storage format to use.

        Returns
        -------
        str
            The default file path inside ``self.data_directory``.
        """

        extension = "h5ad" if file_format == "h5ad" else "zarr"
        return os.path.join(self.data_directory, f"adata.{extension}")

    def save_adata(
        self,
        adata_path: Optional[str] = None,
        file_format: Literal["h5ad", "zarr"] = "h5ad",
        **kwargs: Any,
    ) -> str:
        """
        Persist the AnnData object to disk in ``h5ad`` or ``zarr`` format.

        Parameters
        ----------
        adata_path : str, optional
            Custom path for saving. Defaults to ``adata.<ext>`` inside ``data_directory``.
        file_format : {"h5ad", "zarr"}, default "h5ad"
            Output format.
        **kwargs : Any
            Additional arguments forwarded to AnnData's write method.

        Returns
        -------
        str
            The path where the AnnData object was saved.

        Raises
        ------
        ValueError
            If ``self.adata`` is ``None``.
        FileNotFoundError
            If the target directory does not exist.
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
            self.adata.write_h5ad(save_path, **kwargs)
        else:
            self.adata.write_zarr(save_path, **kwargs)

        logger.info(f"AnnData saved to {save_path} (format={file_format}).")
        return save_path

    def load_adata(
        self,
        adata_path: Optional[str] = None,
        file_format: Literal["h5ad", "zarr"] = "h5ad",
        **kwargs: Any,
    ) -> ad.AnnData:
        """
        Load an AnnData object from disk and attach it to the MSICube instance.

        Parameters
        ----------
        adata_path : str, optional
            Path to the AnnData file/directory. Defaults to ``adata.<ext>`` inside ``data_directory``.
        file_format : {"h5ad", "zarr"}, default "h5ad"
            Input format.
        **kwargs : Any
            Additional arguments forwarded to AnnData's read method.

        Returns
        -------
        AnnData
            The loaded AnnData object (also stored in ``self.adata``).

        Raises
        ------
        FileNotFoundError
            If the specified AnnData path does not exist.
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
        copy: bool = False,
    ) -> Optional[ad.AnnData]:
        """
        Clip or mask intensity values stored in the MSI cube's AnnData object.

        Parameters
        ----------
        low, high
            Thresholds. If provided:
              - values < ``low`` are handled by ``low_action``
              - values > ``high`` are handled by ``high_action``
        low_action
            - "keep": do nothing
            - "nan":  set values < low to ``NaN``
            - "zero": set values < low to ``0``
            - "clip": set values < low to ``low``
        high_action
            - "keep": do nothing
            - "nan":  set values > high to ``NaN``
            - "clip": set values > high to ``high``
        layer
            If ``None`` operate on ``adata.X`` else on ``adata.layers[layer]``.
        copy
            If ``True``, operate on a copy of :attr:`adata` and return it. Otherwise
            modify the existing object in-place and return ``None``.

        Notes (sparse)
        --------------
        - For sparse matrices, only stored (non-zero) entries are modified. Implicit
          zeros stay zeros.
        - Setting sparse entries to ``NaN`` is allowed but can break downstream
          operations. If more arithmetic follows, prefer ``low_action="zero"`` and
          ``high_action="clip"``.

        Returns
        -------
        AnnData or None
            The modified AnnData object when ``copy=True``; otherwise ``None``.
        """

        if self.adata is None:
            raise ValueError("MSICube.adata is None. Run data extraction first.")

        obj = self.adata.copy() if copy else self.adata

        if layer is None:
            X = obj.X
        else:
            if layer not in obj.layers:
                raise KeyError(f"Layer '{layer}' not found in adata.layers.")
            X = obj.layers[layer]

        if low is None and high is None:
            return obj if copy else None

        if sp.issparse(X):
            X2 = X.astype(np.float32, copy=True)
            data = X2.data

            if low is not None and low_action != "keep":
                mask = data < low
                if low_action == "nan":
                    data[mask] = np.nan
                elif low_action == "zero":
                    data[mask] = 0.0
                elif low_action == "clip":
                    data[mask] = low

            if high is not None and high_action != "keep":
                mask = data > high
                if high_action == "nan":
                    data[mask] = np.nan
                elif high_action == "clip":
                    data[mask] = high

            if low_action == "zero":
                X2.eliminate_zeros()

            X_out = X2

        else:
            X_arr = np.asarray(X, dtype=np.float32).copy()

            if low is not None and low_action != "keep":
                if low_action == "nan":
                    X_arr[X_arr < low] = np.nan
                elif low_action == "zero":
                    X_arr[X_arr < low] = 0.0
                elif low_action == "clip":
                    X_arr[X_arr < low] = low

            if high is not None and high_action != "keep":
                if high_action == "nan":
                    X_arr[X_arr > high] = np.nan
                elif high_action == "clip":
                    X_arr[X_arr > high] = high

            X_out = X_arr

        if layer is None:
            obj.X = X_out
        else:
            obj.layers[layer] = X_out

        obj.uns.setdefault("intensity_clipping", [])
        obj.uns["intensity_clipping"].append(
            {
                "layer": layer,
                "low": None if low is None else float(low),
                "high": None if high is None else float(high),
                "low_action": low_action,
                "high_action": high_action,
            }
        )

        return obj if copy else None

    @classmethod
    def from_saved_adata(
        cls,
        data_directory: str,
        adata_path: Optional[str] = None,
        file_format: Literal["h5ad", "zarr"] = "h5ad",
        **kwargs: Any,
    ) -> "MSICube":
        """
        Construct an MSICube instance and populate it with a persisted AnnData object.

        Parameters
        ----------
        data_directory : str
            Base directory for the MSICube (used for locating raw data and default AnnData path).
        adata_path : str, optional
            Path to the saved AnnData. Defaults to ``adata.<ext>`` inside ``data_directory``.
        file_format : {"h5ad", "zarr"}, default "h5ad"
            Input format for the AnnData file.
        **kwargs : Any
            Additional arguments forwarded to :meth:`load_adata`.

        Returns
        -------
        MSICube
            An initialized MSICube with ``adata`` populated from disk.
        """

        cube = cls(data_directory=data_directory)
        cube.load_adata(adata_path=adata_path, file_format=file_format, **kwargs)
        return cube

    def log1p_intensity(
        self,
        *,
        base: Optional[float] = None,
        layer: Optional[str] = None,
        copy: bool = False,
    ) -> Optional["MSICube"]:
        """
        Apply Scanpy-like ``log1p`` transformation to the MSI intensity matrix.

        Parameters
        ----------
        base : float | None
            If provided, divide by ``log(base)`` to change the logarithm base.
        layer : str | None
            Target a specific ``adata.layers`` entry instead of ``adata.X``.
        copy : bool
            If ``True``, return a new :class:`MSICube` instance with transformed data.
            Otherwise, modify the object in-place and return ``None``.

        Returns
        -------
        MSICube | None
            A new MSICube when ``copy=True``; otherwise ``None``.

        Raises
        ------
        ValueError
            If ``adata`` is missing on the MSICube instance.
        KeyError
            If a specified ``layer`` is not found.
        """

        if self.adata is None:
            raise ValueError("MSICube.adata is None. Run data extraction first.")

        target_cube = MSICube(self.data_directory) if copy else self
        target_cube.org_imzml_path_dict = self.org_imzml_path_dict.copy()
        target_cube.adata = self.adata.copy() if copy else self.adata

        adata_obj = target_cube.adata

        if layer is None:
            if adata_obj.X is None:
                raise ValueError("MSICube.adata.X is None.")
            adata_obj.X = _log1p_inplace_or_copy(adata_obj.X, base=base)
        else:
            if layer not in adata_obj.layers:
                raise KeyError(f"Layer '{layer}' not found in adata.layers")
            adata_obj.layers[layer] = _log1p_inplace_or_copy(
                adata_obj.layers[layer], base=base
            )

        adata_obj.uns.setdefault("log1p", {})
        adata_obj.uns["log1p"]["base"] = base

        return target_cube if copy else None

    def compute_cosine_colocalization(
        self, *, params: CosineColocParams = CosineColocParams()
    ) -> Tuple[Union[np.ndarray, sp.csr_matrix], Optional[np.ndarray]]:
        """Compute cosine similarity between ion images stored on this cube.

        This is a convenience wrapper around
        :func:`pymsix.processing.colocalization.compute_mz_cosine_colocalization`.
        The resulting similarity matrix is stored in ``adata.varp`` when
        ``params.store_varp_key`` is provided, and optional keep masks are stored in
        ``adata.var``.
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
        Aggregate variables that share the same label in ``adata.var[label_col]``.

        The resulting ion images are stored in ``adata.obsm[obsm_key]`` with column
        order recorded in ``adata.uns[f"{obsm_key}_labels"]``.

        Parameters
        ----------
        label_col : str
            Column in ``adata.var`` that contains the grouping labels.
        layer : str | None, default None
            Aggregate a specific layer instead of ``adata.X``.
        agg : {"mean", "median", "max"}, default "mean"
            Aggregation strategy across variables with the same label.
        obsm_key : str, default "X_by_label"
            Key for the aggregated matrix in ``adata.obsm``.
        dropna : bool, default True
            Drop variables with missing labels if ``True``; otherwise replace NaN labels
            with "NA".
        keep_order : bool, default True
            Preserve the first occurrence order of labels instead of sorting.
        as_df : bool, default False
            Store aggregated values as a DataFrame instead of a NumPy array.
        dtype : dtype or type, default numpy.float32
            Data type of the aggregated matrix.

        Returns
        -------
        pandas.Index
            Index of the aggregated label names, named after ``label_col``.
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

    def _scan_imzml_files(self, directory: str) -> None:
        """
        Scans the directory for .imzML files and builds the dictionary.
        Handles both 'imzML' and 'imzml' extensions.
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

    def compute_all_mean_spectra(
        self, mode: Literal["profile", "centroid"], **kwargs: Any
    ) -> None:
        """
        Applies compute_mean_spectrum to all imzML files and stores the results.

        All relevant parameters (min_mz, max_mz, binning_p, tolerance_da,
        mass_accuracy_ppm, n_sigma) must be passed via kwargs and used
        to construct a single MeanSpectrumOptions object.
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

    def compute_global_mean_spectrum(self, **kwargs: Any) -> None:
        """
        Computes a single global mean spectrum by combining all individual mean spectra
        stored in adata.uns["mean_spectra"].
        """
        if self.adata is None or "mean_spectra" not in self.adata.uns:
            logger.error(
                "Individual mean spectra not calculated yet. Run compute_all_mean_spectra first."
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

    def clear_mean_spectra(self) -> None:
        """
        Remove mean spectra-related data from ``adata.uns`` to free up space.

        This clears both per-sample and global mean spectra along with their
        associated options and metadata.
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

    def perform_peak_picking(self, **kwargs: Any) -> None:
        """
        Performs peak picking on the global mean spectrum and stores the selected
        mz values in adata.var.

        The selection criteria (topn, distance_da, distance_ppm, binning_p)
        are passed via kwargs.
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
            # Index formatted for a readable identifier (e.g., 'mz_200.0000')
            index=[f"mz_{m:.4f}" for m in selected_mzs],
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

    def extract_peak_matrix(self, **kwargs: Any) -> None:
        """
        Extracts peak intensities for all pixels in all samples based on the
        selected mz values in adata.var.

        Populates:
            - adata.X (intensities)
            - adata.obsm['spatial'] (coordinates)
            - adata.obs['sample'] (provenance)

        Kwargs:
            tol_da (float): Tolerance in Dalton.
            tol_ppm (float): Tolerance in PPM.
        """
        logger.info("Starting extraction of peak intensity matrix (X) for all samples.")

        # 1. Prerequisites Check
        if self.adata is None or self.adata.var is None or "mz" not in self.adata.var:
            logger.error(
                "Target mz list not found in adata.var['mz']. "
                "Run perform_peak_picking first."
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

    def tic_normalize(
        self,
        *,
        target_sum: float = 1e6,
        layer: Optional[str] = None,
        store_tic_in_obs: Optional[str] = "tic",
        copy: bool = False,
    ) -> Optional["MSICube"]:
        """
        Apply Total Ion Current (TIC) normalization to the cube's intensity matrix.

        Parameters
        ----------
        target_sum : float, default 1e6
            After normalization, each spectrum sums to ``target_sum``.
        layer : str | None, default None
            If ``None``, normalize ``adata.X``; otherwise, normalize ``adata.layers[layer]``.
        store_tic_in_obs : str | None, default "tic"
            Name of the ``adata.obs`` column where pre-normalization TIC values are stored.
            If ``None``, TIC values are not stored.
        copy : bool, default False
            If ``True``, operate on and return a deep copy of the cube. Otherwise, modify
            in place and return ``None``.

        Returns
        -------
        MSICube | None
            A normalized copy when ``copy`` is ``True``; otherwise ``None``.
        """

        return tic_normalize_msicube(
            self,
            target_sum=target_sum,
            layer=layer,
            store_tic_in_obs=store_tic_in_obs,
            copy=copy,
        )

    def plot_ion_images(
        self,
        mz: Union[float, str, Sequence[Union[float, str]]],
        samples: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Wrapper to plot ion images.

        Args:
            mz: One or multiple m/z values or aggregated label names.
            samples: One or multiple sample names. If None, uses all available samples.
            **kwargs: Arguments passed to plot_ion_images (cmap, share_intensity_scale, etc.)
        """
        if self.adata is None:
            raise ValueError("AnnData is empty.")

        # Default to all samples if None provided
        if samples is None:
            samples = self.adata.obs['sample'].unique().tolist()

        plot_ion_images(self, mz=mz, samples=samples, **kwargs)

    def plot_mean_spectrum_windows(
        self,
        peak_mzs: Sequence[float],
        labels: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plots zoomed windows of mean spectra around mz of interest for
        multiple samples.

        This is a wrapper for msix.plotting.spectrum.plot_mean_spectrum_windows.

        Parameters
        ----------
        labels : list of str
            List of sample names (keys in adata.uns['mean_spectra']) to compare.
        peak_mzs : list of float
            List of target mz values to center the windows on.
        span_da : float
            Half-width of the zoom window in Da (default 0.1 Da).
        **kwargs
            Arguments passed to the plotting function (e.g., tol_da, tol_ppm, ncols, figsize).
        """
        plot_mean_spectrum_windows(self, peak_mzs, labels, **kwargs)

    def recalibration(
        self,
        database_mass_file: str,
        options: RecalibrationOptions,
        output_directory: Optional[str] = None,
        n_workers: int = 1,
    ) -> None:
        """
        Performs mass recalibration on all raw imzML files using a mass database.

        The method updates the MSICube object to point to the new recalibrated files.
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

    def plot_recalibration_diagnostics(
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
        Plots the recalibration diagnostics (KDE, RANSAC fit) for selected pixels
        of a specific MSI cube.

        Args:
            sample_name: The name of the sample (MSI cube) to plot.
            database_mass_file: Path to the exact mass list (calibrants.txt).
            options: RecalibrationOptions (used to build RecalParams).
            pixel_idx: Specific 1D pixel indices to plot.
            pixel_coord: Specific (x,y) or (x,y,z) coordinates to plot (e.g., '10,20').
            n_random: Number of random pixels to plot.
            seed: Seed for random selection.
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
        candidates_df: pd.DataFrame,
        options: Optional[MassClusteringOptions] = None,
        keep_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Regroupe les pics (mz) stockés dans adata.var en clusters de familles chimiques.

        Args:
            candidates_df: DataFrame contenant les deltas de masse (ex: delta_da, score, label).
            options: Instance de MassClusteringOptions pour configurer le graphe et Leiden.
            keep_mask: Matrice optionnelle pour filtrer les arêtes autorisées.
        """
        if self.adata is None:
            raise ValueError(
                "L'objet AnnData est vide. Exécutez le peak picking au préalable."
            )

        # 1. Préparation des options
        opts = options or MassClusteringOptions()
        opts.validate()

        # 2. Récupération des masses (pics)
        # On utilise les mz identifiés lors de la phase de peak picking
        if "mz" not in self.adata.var:
            raise ValueError(
                "La colonne 'mz' est absente de adata.var. Peak picking requis."
            )

        masses = self.adata.var["mz"].values

        logger.info(
            f"Démarrage du clustering sur {len(masses)} masses (Resolution: {opts.resolution})..."
        )

        # 3. Appel de la fonction de traitement
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

        # 4. Stockage des résultats
        # Les labels de clusters (ex: 0, 1, 2... et -1 pour les bruits) vont dans var
        self.adata.var["mass_cluster"] = res["labels"]

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

        # Les statistiques et les arêtes vont dans uns
        self.adata.uns["mass_clustering"] = {
            "n_clusters": res["n_clusters"],
            "n_minus1": res["n_minus1"],
            "compression": res["compression"],
            "edges": edges_with_names,
            "options": opts,  # On garde les options pour la traçabilité
        }

        logger.info(f"Clustering terminé: {res['n_clusters']} clusters trouvés.")

    def compute_kendrick_coordinates(
        self,
        *,
        mz_key: str = "mz",
        base: Union[str, float, Tuple[float, float]] = "CH2",
        kmd_mode: Literal["fraction", "defect"] = "fraction",
        varm_key: Optional[str] = None,
        store_1d_in_var: bool = False,
        var_prefix: str = "kendrick",
    ) -> str:
        """Calcule et stocke les coordonnées de Kendrick dans ``adata.varm``.

        Args:
            mz_key: Nom de la colonne de masses dans ``adata.var``.
            base: Base Kendrick (formule chimique, masse exacte ou tuple (exact, nominal)).
            kmd_mode: Mode de calcul du défaut de masse (``"fraction"`` ou ``"defect"``).
            varm_key: Nom de la clé de sortie dans ``adata.varm``. Généré automatiquement si None.
            store_1d_in_var: Si True, stocke aussi KM/KMD comme colonnes 1D dans ``adata.var``.
            var_prefix: Préfixe pour les colonnes 1D optionnelles.

        Returns:
            La clé ``varm`` utilisée pour stocker les coordonnées calculées.
        """

        if self.adata is None:
            raise ValueError("L'objet AnnData est vide. Impossible de calculer les coordonnées Kendrick.")

        return compute_kendrick_varm(
            self.adata,
            mz_key=mz_key,
            base=base,
            kmd_mode=kmd_mode,
            varm_key=varm_key,
            store_1d_in_var=store_1d_in_var,
            var_prefix=var_prefix,
        )

    def manual_label_kendrick(
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
        """Launch an interactive manual labeling widget in Kendrick space.

        Returns the ``(ui, state)`` tuple from
        :func:`pymsix.plotting.kendrick_manual_label.manual_label_vars_from_kendrick`.
        Requires the optional ``viz`` dependencies (``plotly`` and ``ipywidgets``).
        """

        from pymsix.plotting.kendrick_manual_label import manual_label_vars_from_kendrick

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
        Génère un diagramme de Kendrick (KMD) à partir des résultats de clustering.

        Args:
            options: Une instance de KendrickPlotOptions. Si None, les valeurs par défaut sont utilisées.
            **kwargs: Permet de surcharger des options spécifiques (ex: base="H2O").

        Returns:
            Un tuple (Figure, Axes, DataFrame utilisé).
        """
        # 1. Préparation des options
        if options is None:
            options = KendrickPlotOptions()

        # Surcharge éventuelle via kwargs pour modifier les attributs de la dataclass
        for key, value in kwargs.items():
            if hasattr(options, key):
                setattr(options, key, value)

        options.validate()

        # 2. Récupération des données depuis AnnData
        if self.adata is None:
            raise ValueError("L'objet AnnData est vide.")

        if "mass_cluster" not in self.adata.var:
            raise ValueError(
                "Aucun résultat de clustering trouvé dans adata.var. "
                "Veuillez exécuter 'cluster_masses()' au préalable."
            )

        # Récupération des masses (mz)
        if options.mass_col not in self.adata.var:
            raise ValueError(
                f"La colonne '{options.mass_col}' est absente de adata.var. Peak picking requis."
            )
        masses = self.adata.var[options.mass_col].values

        # Reconstruction du dictionnaire clustering_result attendu par la fonction de plot
        clustering_result = {"labels": self.adata.var["mass_cluster"].values}

        # Récupération optionnelle de la famille/composition
        family = None
        if "family" in self.adata.var.columns:
            family = self.adata.var["family"].values
        elif "label" in self.adata.var.columns:
            family = self.adata.var["label"].values

        # 3. Appel de la fonction de rendu fournie

        return plot_kendrick_from_clustering(
            masses=masses,
            clustering_result=clustering_result,
            adata=self.adata,
            kendrick_varm_key=options.kendrick_varm_key,
            family=family,
            base=options.base,
            mass_col=options.mass_col,  # Nom utilisé dans le DataFrame interne pour les axes
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
