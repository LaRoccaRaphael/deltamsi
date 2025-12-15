# class MSICube

from concurrent.futures import ProcessPoolExecutor
import os
import re
import anndata as ad
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Literal, Sequence

from pymsix.plotting.ion_images import plot_ion_images
from pymsix.plotting.spectrum import plot_mean_spectrum_windows
from pymsix.processing.mean_spectrum import compute_mean_spectrum
from pymsix.processing.combine_mean_spectra import combine_mean_spectra, Spectrum
from pymsix.processing.peak_picking import peak_picking, extract_peak_matrix
from pymsix.processing.recalibration_DEPRECATED import recalibrate_imzml_file
from pymsix.params.options import (
    MeanSpectrumOptions,
    GlobalMeanSpectrumOptions,
    PeakPickingOptions,
    PeakMatrixOptions,
    RecalibrationOptions,
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

    def perform_peak_picking(self, **kwargs: Any) -> None:
        """
        Performs peak picking on the global mean spectrum and stores the selected
        m/z values in adata.var.

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
        logger.info(f"Peak picking finished. {len(selected_mzs)} m/z selected.")

        # Store results in adata.var

        # Create a new DataFrame for var based on the selected m/z values.
        # AnnData format prefers unique identifiers for the .var index.
        # We use the formatted m/z as a unique identifier for AnnData.var.
        new_var_data = pd.DataFrame(
            data={"m/z": selected_mzs},
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

    def plot_ion_images(self, sample_name: str, **kwargs: Any) -> None:
        """
        Plots ion images for a specific sample.

        This is a wrapper for msix.plotting.ion_images.plot_ion_images.

        Parameters
        ----------
        sample_name : str
            The name of the sample to visualize.
        **kwargs
            Arguments passed to the plotting function (e.g., mz_list, ncols, cmap, vmin, vmax).
        """
        return plot_ion_images(self, sample_name=sample_name, **kwargs)

    def plot_mean_spectrum_windows(
        self,
        labels: Sequence[str],
        peak_mzs: Sequence[float],
        span_da: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """
        Plots zoomed windows of mean spectra around m/z of interest for
        multiple samples.

        This is a wrapper for msix.plotting.spectrum.plot_mean_spectrum_windows.

        Parameters
        ----------
        labels : list of str
            List of sample names (keys in adata.uns['mean_spectra']) to compare.
        peak_mzs : list of float
            List of target m/z values to center the windows on.
        span_da : float
            Half-width of the zoom window in Da (default 0.1 Da).
        **kwargs
            Arguments passed to the plotting function (e.g., tol_da, tol_ppm, ncols, figsize).
        """
        return plot_mean_spectrum_windows(
            self, labels=labels, peak_mzs=peak_mzs, span_da=span_da, **kwargs
        )

    def recalibration(
        self,
        database_mass_file: str,
        options: Optional[RecalibrationOptions] = None,
        output_directory: Optional[str] = None,
    ) -> None:
        """
        Performs mass recalibration on all samples within the MSICube object
        and writes the new calibrated imzML/ibd files to the output directory.
        This process is executed in parallel for each sample.

        :param database_mass_file: Path to the file containing exact calibration masses.
        :param options: RecalibrationOptions object specifying hyperparameters.
                        If None, uses default options.
        :param output_directory: Directory to store the recalibrated files.
                                 Defaults to creating a 'recal_imzML' subdirectory
                                 in the original data location.
        """
        logger.info("Starting mass recalibration across all samples.")

        if options is None:
            options = RecalibrationOptions()

        try:
            options.validate()
        except ValueError as e:
            logger.error(f"Invalid Recalibration Options: {e}")
            return

        # 1. Parsing and preparing the mass database
        try:
            exact_mass_full = np.genfromtxt(database_mass_file)
            # Must order the list of masses for the binary search
            database_exactmass = exact_mass_full[exact_mass_full.argsort()]
            logger.info(
                f"Calibration database loaded ({len(database_exactmass)} masses)."
            )
        except Exception as e:
            logger.error(f"Failed to load or sort the mass file: {e}")
            return

        # 2. Define output directory and create it
        if output_directory is None:
            output_directory = os.path.join(self.data_directory, "recal_imzML")

        os.makedirs(output_directory, exist_ok=True)
        logger.info(f"Recalibrated imzML files will be saved in: {output_directory}")

        # 3. Parallel Execution Setup
        max_workers = min(len(self.org_imzml_path_dict), os.cpu_count() or 4)
        logger.info(f"Starting parallel recalibration using {max_workers} processes.")

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for sample_name, input_path in self.org_imzml_path_dict.items():
                # Keep the original filename (e.g., pseudomonas.imzml)
                imzml_filename = os.path.basename(input_path)
                output_path = os.path.join(output_directory, imzml_filename)

                # Submit task to the process pool
                future = executor.submit(
                    recalibrate_imzml_file,
                    imzml_input_path=input_path,
                    imzml_output_path=output_path,
                    database_exactmass=database_exactmass,
                    options=options,
                )
                futures.append((sample_name, future))

            # Collect results as they complete
            for sample_name, future in futures:
                try:
                    success = future.result()
                    results.append((sample_name, success))
                except Exception as exc:
                    logger.error(f"Recalibration of {sample_name} failed: {exc}")
                    results.append((sample_name, False))

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
