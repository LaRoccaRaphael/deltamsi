# class MSICube

import os
import re
import anndata as ad
from typing import Dict, Literal, Optional, Any

from .processing.mean_spectrum import compute_mean_spectrum
from .params.options import MeanSpecParams, BinningParams


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

        kwargs can include parameters from BinningParams or MeanSpecParams.
        """
        if self.adata is None:
            self.adata = ad.AnnData()

        mean_spectra_dict = {}

        # 1. Parameter Preparation
        params = MeanSpecParams()
        binning = BinningParams()

        # Simple method to update dataclasses with kwargs
        param_keys = [f.name for f in MeanSpecParams.__dataclass_fields__.values()]
        binning_keys = [f.name for f in BinningParams.__dataclass_fields__.values()]

        # Use only relevant keys for each class
        mean_kwargs = {k: v for k, v in kwargs.items() if k in param_keys}
        bin_kwargs = {k: v for k, v in kwargs.items() if k in binning_keys}

        # Create new dataclass instances with updated values
        params = params.__class__(**{**params.__dict__, **mean_kwargs})
        binning = binning.__class__(**{**binning.__dict__, **bin_kwargs})

        # 2. Iteration and Calculation
        for sample_name, imzml_path in self.org_imzml_path_dict.items():
            print(f"Calculating mean spectrum for: {sample_name} (Mode: {mode})")

            try:
                mean_mz, mean_y = compute_mean_spectrum(
                    imzml_path, mode=mode, binning=binning, params=params
                )

                mean_spectra_dict[sample_name] = {"mz": mean_mz, "intensity": mean_y}
            except Exception as e:
                print(f"Error during calculation for {sample_name}: {e}")

        # 3. Storing in AnnData
        self.adata.uns["mean_spectra"] = mean_spectra_dict
        self.adata.uns["mean_spectra_samples"] = list(self.org_imzml_path_dict.keys())
        print("Mean spectra calculated and stored.")
