# class MSICube

import os
import re
import anndata as ad
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Literal

from msix.processing.mean_spectrum import compute_mean_spectrum
from msix.processing.combine_mean_spectra import combine_mean_spectra, Spectrum
from msix.processing.peak_picking import peak_picking, extract_peak_matrix
from msix.params.options import (
    MeanSpectrumOptions,
    GlobalMeanSpectrumOptions,
    PeakPickingOptions,
    PeakMatrixOptions,
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
    Contains an anndata object for data and a mapping of raw files.
    """

    def __init__(self, data_directory: str) -> None:
        """
        Initializes the MSICube object by scanning the directory for imzML files.

        :param data_directory: Path to the directory containing imzML files.
        """
        if not os.path.isdir(data_directory):
            raise FileNotFoundError(f"The directory does not exist: {data_directory}")

        self.org_imzml_path_dict: Dict[str, str] = {}
        self._scan_imzml_files(data_directory)

        # 2. Initialization of an empty anndata object
        # The anndata object will be filled later; we initialize the slot now.
        self.adata: Optional[ad.AnnData] = None

        logger.info(
            f"MSICube initialized with {len(self.org_imzml_path_dict)} samples found."
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
            "binning_p": kwargs.pop("binning_p", 0.0001),
            "tolerance_da": kwargs.pop("tolerance_da", None),
            "mass_accuracy_ppm": kwargs.pop("mass_accuracy_ppm", None),
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
                options=options,
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

    def perform_peak_picking(self, **kwargs: Any) -> None:
        """
        Performs peak picking on the global mean spectrum and stores the selected
        m/z values in adata.var.

        The selection criteria (topn, distance_da, distance_ppm, binning_p)
        are passed via kwargs.
        """
        logger.info("Starting peak picking on the global mean spectrum.")

        # 1. Check required input data
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

        # 2. Options parsing and validation
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

        # 3. Retrieve data and call the processing function
        global_spectrum = self.adata.uns["mean_spectrum_global"]
        mzs_combined = global_spectrum["mz"]
        ints_combined = global_spectrum["intensity"]

        # Call the peak_picking function with the options object
        selected_mzs = peak_picking(
            mzs_combined,
            ints_combined,
            options=options,
        )

        logger.info(f"Peak picking finished. {len(selected_mzs)} m/z selected.")

        # 4. Store results in adata.var

        # Create a new DataFrame for var based on the selected m/z values.
        # AnnData format prefers unique identifiers for the .var index.
        # We use the formatted m/z as a unique identifier for AnnData.var.
        new_var_data = pd.DataFrame(
            data={"m/z": selected_mzs},
            # Index formatted for a readable identifier (e.g., 'mz_200.0000')
            index=[f"mz_{m:.4f}" for m in selected_mzs],
        )
        new_var_data.index.name = "feature_id"

        # Replace adata.var
        self.adata.var = new_var_data

        # 5. Store options in adata.uns for provenance
        self.adata.uns["peak_picking_options"] = options.to_dict()

        logger.info(
            "Selected m/z values stored in adata.var and adata.uns['peak_picking_options']."
        )

    def extract_peak_matrix(self, **kwargs: Any) -> None:
        """
        Extracts peak intensities for all pixels in all samples based on the
        selected m/z values in adata.var.

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
        if self.adata is None or self.adata.var is None or "m/z" not in self.adata.var:
            logger.error(
                "Target m/z list not found in adata.var['m/z']. "
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
        target_mzs = self.adata.var["m/z"].values

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

        # Store options used for this step
        self.adata.uns["matrix_extraction_options"] = options.to_dict()

        logger.info(
            f"Extraction complete. "
            f"Final shape: {self.adata.shape} (Pixels x Peaks). "
            f"Data stored in adata.X, adata.obsm['spatial'], adata.obs['sample']."
        )
