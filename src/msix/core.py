# class MSICube

import os
import re
import anndata as ad
from typing import Dict, Literal, Optional, Any

from .processing.mean_spectrum import compute_mean_spectrum
from .params.options import MeanSpectrumOptions


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

        print(
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
                    print(f"WARNING: Missing .ibd file for {filename}")
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
            print(
                f"WARNING: Ignoring unknown arguments passed to mean spectra computation: {list(kwargs.keys())}"
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
            print(
                f"Calculating mean spectrum for: {sample_name} (Mode: {options.mode})"
            )

            try:
                mean_mz, mean_y = compute_mean_spectrum(imzml_path, options=options)

                mean_spectra_dict[sample_name] = {"mz": mean_mz, "intensity": mean_y}
            except Exception as e:
                print(f"Error during calculation for {sample_name}: {e}")
                continue

        # 3. Storing in AnnData
        self.adata.uns["mean_spectra"] = mean_spectra_dict
        self.adata.uns["mean_spectra_options"] = options.__dict__
        self.adata.uns["mean_spectra_samples"] = list(self.org_imzml_path_dict.keys())
        print("Mean spectra calculated and stored.")
