# tests/test_core.py

from pathlib import Path
import pytest
import numpy as np
import anndata as ad
from traitlets import Any

# Import the class and parameters to be tested
from msix.core import MSICube
from msix.params.options import MeanSpecParams, BinningParams

# ----------------------------------------------------------------------
# FIXTURES (Helper functions for setting up test conditions)
# ----------------------------------------------------------------------


@pytest.fixture
def mock_imzml_data(tmp_path: Path) -> str:
    """
    Creates a temporary directory structure with fake imzML/ibd files
    for simulating real data.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Sample A: Normal case (uppercase extension)
    (data_dir / "sample_A.imzML").touch()
    (data_dir / "sample_A.ibd").touch()

    # Sample B: Normal case (lowercase extension)
    (data_dir / "sample_B.imzml").touch()
    (data_dir / "sample_B.ibd").touch()

    # Sample C: Missing .ibd file (should be ignored)
    (data_dir / "sample_C.imzML").touch()

    # Non-relevant file (should be ignored)
    (data_dir / "config.txt").touch()

    return str(data_dir)


@pytest.fixture
def mock_mean_spectrum(mocker: Any) -> Any:
    """Simulates the external function compute_mean_spectrum."""
    # Define a fixed return value for the mocked function: (mz_array, intensity_array)
    return_value = (np.array([100.0, 200.0]), np.array([5.0, 10.0]))

    # Patch the real function in the module where it is imported (msix.core)
    mock_func = mocker.patch(
        "msix.core.compute_mean_spectrum", return_value=return_value
    )
    return mock_func


# ----------------------------------------------------------------------
# TESTS FOR MSICUBE INITIALIZATION AND FILE SCANNING
# ----------------------------------------------------------------------


def test_init_raises_file_not_found() -> None:
    """Tests that initialization fails if the data directory does not exist."""
    with pytest.raises(FileNotFoundError):
        MSICube("/nonexistent/path/to/data")


def test_init_success_and_state(mock_imzml_data: str) -> None:
    """Tests successful initialization and checks the initial state of the object."""
    cube = MSICube(data_directory=mock_imzml_data)

    # Check basic state
    assert cube.adata is None

    # Only sample_A and sample_B should be found (sample_C is ignored due to missing .ibd)
    assert len(cube.org_imzml_path_dict) == 2
    assert "sample_A" in cube.org_imzml_path_dict
    assert "sample_B" in cube.org_imzml_path_dict
    assert "sample_C" not in cube.org_imzml_path_dict

    # Verify path handling (ensures the function can resolve file names)
    assert cube.org_imzml_path_dict["sample_A"].endswith("sample_A.imzML")


# ----------------------------------------------------------------------
# TESTS FOR compute_all_mean_spectra METHOD
# ----------------------------------------------------------------------


def test_compute_mean_spectra_stores_in_adata(
    mock_imzml_data: str, mock_mean_spectrum: Any
) -> None:
    """Tests that mean spectra are calculated and correctly stored in adata.uns."""

    cube = MSICube(data_directory=mock_imzml_data)

    # Execute the method
    cube.compute_all_mean_spectra(mode="profile")

    # 1. Verification of the call count (must be called once per found sample)
    assert mock_mean_spectrum.call_count == 2

    # 2. Verification of AnnData structure
    assert isinstance(cube.adata, ad.AnnData)
    assert "mean_spectra" in cube.adata.uns

    # 3. Verification of stored data
    mean_spectra = cube.adata.uns["mean_spectra"]
    assert len(mean_spectra) == 2

    # Check that the data stored matches the mocked return value
    expected_mz = np.array([100.0, 200.0])
    expected_intensity = np.array([5.0, 10.0])

    assert np.array_equal(mean_spectra["sample_A"]["mz"], expected_mz)
    assert np.array_equal(mean_spectra["sample_B"]["intensity"], expected_intensity)
    expected_samples = ["sample_A", "sample_B"]
    actual_samples = cube.adata.uns["mean_spectra_samples"]
    assert sorted(actual_samples) == sorted(expected_samples)


def test_compute_mean_spectra_passes_kwargs(
    mock_imzml_data: str, mock_mean_spectrum: Any
) -> None:
    """Tests that arguments passed via kwargs are correctly routed to dataclasses."""

    cube = MSICube(data_directory=mock_imzml_data)

    # Custom parameters for the test
    test_bin_width = 0.5

    cube.compute_all_mean_spectra(
        mode="centroid",
        bin_width=test_bin_width,
        extra_param="should_be_ignored",  # Should be ignored by the dataclasses
    )

    # Get the arguments used in the first call to the mocked function
    call_kwargs = mock_mean_spectrum.call_args_list[0].kwargs

    # Verify the mode argument
    assert call_kwargs["mode"] == "centroid"

    # Verify BinningParams update
    called_binning_obj = call_kwargs["binning"]
    assert isinstance(called_binning_obj, BinningParams)
    assert called_binning_obj.bin_width == test_bin_width

    # Verify MeanSpecParams update
    called_params_obj = call_kwargs["params"]
    assert isinstance(called_params_obj, MeanSpecParams)
