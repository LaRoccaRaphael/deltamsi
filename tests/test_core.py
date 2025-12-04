from pathlib import Path
import pytest
import numpy as np
import anndata as ad
from typing import Any, Literal, cast, List

# Import the class and parameters to be tested
from msix.core import MSICube
from msix.params.options import (
    MeanSpectrumOptions,
)  # Import added

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
    # This simulation is used to populate adata.uns['mean_spectra']
    return_value = (np.arange(100.0, 110.0, 1.0), np.ones(10) * 10.0)

    mock_func = mocker.patch(
        "msix.core.msicube.compute_mean_spectrum", return_value=return_value
    )
    return mock_func


@pytest.fixture
def mock_combine_mean_spectra(mocker: Any) -> Any:
    """Simulates the external function combine_mean_spectra."""
    # Define a fixed return value for the mocked function: (mz_array, intensity_array)
    return_value = (np.array([100.0, 101.0, 102.0]), np.array([15.0, 15.0, 15.0]))

    mock_func = mocker.patch(
        "msix.core.msicube.combine_mean_spectra", return_value=return_value
    )
    return mock_func


@pytest.fixture
def cube_with_mean_spectra(mock_imzml_data: str) -> MSICube:
    """Creates an MSICube object pre-populated with individual mean spectra."""
    cube = MSICube(data_directory=mock_imzml_data)

    # Manually populate the necessary adata structure for global combination test
    # This avoids mocking compute_mean_spectrum inside this fixture
    cube.adata = ad.AnnData()
    cube.adata.uns["mean_spectra"] = {
        "sample_A": {
            "mz": np.array([100.0, 101.0]),
            "intensity": np.array([10.0, 10.0]),
        },
        "sample_B": {
            "mz": np.array([100.0, 101.0]),
            "intensity": np.array([20.0, 20.0]),
        },
    }
    cube.adata.uns["mean_spectra_samples"] = ["sample_A", "sample_B"]

    return cube


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
# TESTS FOR compute_all_mean_spectra METHOD (Existing tests)
# ----------------------------------------------------------------------


def test_compute_mean_spectra_stores_in_adata(
    mock_imzml_data: str, mock_mean_spectrum: Any
) -> None:
    """Tests that mean spectra are calculated and correctly stored in adata.uns,
    and checks for the storage of options for provenance."""

    cube = MSICube(data_directory=mock_imzml_data)

    cube.compute_all_mean_spectra(
        mode="profile", min_mz=100.0, max_mz=1000.0, binning_p=0.01
    )

    # 1. Verification of the call count (must be called once per found sample)
    assert mock_mean_spectrum.call_count == 2

    # 2. Verification of AnnData structure
    assert isinstance(cube.adata, ad.AnnData)
    assert "mean_spectra" in cube.adata.uns
    assert "mean_spectra_options" in cube.adata.uns

    # 3. Verification of stored data
    mean_spectra = cube.adata.uns["mean_spectra"]
    assert len(mean_spectra) == 2

    # Check options provenance (should match the input kwargs)
    stored_options = cube.adata.uns["mean_spectra_options"]
    assert stored_options["mode"] == "profile"
    assert stored_options["min_mz"] == 100.0
    assert stored_options["binning_p"] == 0.01

    # 4. Verification of sample names
    expected_samples = ["sample_A", "sample_B"]
    actual_samples = cube.adata.uns["mean_spectra_samples"]
    assert sorted(actual_samples) == sorted(expected_samples)


def test_compute_mean_spectra_passes_options_object(
    mock_imzml_data: str, mock_mean_spectrum: Any
) -> None:
    """Tests that all kwargs are correctly merged into a single MeanSpectrumOptions object
    and passed as the 'options' argument to compute_mean_spectrum."""

    cube = MSICube(data_directory=mock_imzml_data)

    test_kwargs = {
        "mode": "centroid",
        "min_mz": 50.0,
        "max_mz": 1500.0,
        "binning_p": 0.005,
        "mass_accuracy_ppm": 10.0,
        "n_sigma": 4.0,
    }

    mode_arg = cast(Literal["profile", "centroid"], test_kwargs.pop("mode"))

    cube.compute_all_mean_spectra(mode=mode_arg, **test_kwargs)

    call_kwargs = mock_mean_spectrum.call_args_list[0].kwargs

    assert "options" in call_kwargs
    assert isinstance(call_kwargs["options"], MeanSpectrumOptions)

    called_options_obj = call_kwargs["options"]
    assert called_options_obj.mode == "centroid"
    assert called_options_obj.min_mz == 50.0
    assert called_options_obj.max_mz == 1500.0
    assert called_options_obj.binning_p == 0.005
    assert called_options_obj.mass_accuracy_ppm == 10.0
    assert called_options_obj.n_sigma == 4.0
    assert called_options_obj.tolerance_da is None


def test_compute_mean_spectra_raises_value_error_on_invalid_options(
    mock_imzml_data: str, mock_mean_spectrum: Any
) -> None:
    """Tests that compute_all_mean_spectra raises a ValueError when MeanSpectrumOptions.validate() fails."""

    cube = MSICube(data_directory=mock_imzml_data)

    invalid_kwargs = {
        "min_mz": 1000.0,
        "max_mz": 100.0,
        "binning_p": 0.001,
    }

    with pytest.raises(ValueError, match="min_mz must be strictly less than max_mz"):
        cube.compute_all_mean_spectra(mode="profile", **invalid_kwargs)

    assert mock_mean_spectrum.call_count == 0


# ----------------------------------------------------------------------
# TESTS FOR compute_global_mean_spectrum METHOD
# ----------------------------------------------------------------------


def test_compute_global_mean_spectrum_requires_individual_spectra(
    mock_imzml_data: str,
) -> None:
    """Tests that the global mean spectrum calculation is guarded if individual spectra are missing."""
    cube = MSICube(data_directory=mock_imzml_data)

    # cube.adata is None
    # No assertion on output print/log, but internal logic should stop
    cube.compute_global_mean_spectrum()

    # cube.adata exists but 'mean_spectra' is missing
    cube.adata = ad.AnnData()
    cube.compute_global_mean_spectrum()


def test_compute_global_mean_spectrum_calls_combine(
    cube_with_mean_spectra: MSICube, mock_combine_mean_spectra: Any
) -> None:
    """Tests that combine_mean_spectra is called correctly with data and default options."""
    cube = cube_with_mean_spectra

    cube.compute_global_mean_spectrum()

    # 1. Check call count
    assert mock_combine_mean_spectra.call_count == 1

    # 2. Check arguments passed to combine_mean_spectra (especially the spectra list)
    call_args = mock_combine_mean_spectra.call_args[0]
    spectra_list = call_args[0]

    assert isinstance(spectra_list, List)
    assert len(spectra_list) == 2  # Two samples were mocked in the fixture

    # Check content structure: list of tuples (mz, intensity)
    assert np.all(spectra_list[0][0] == np.array([100.0, 101.0]))  # mz sample A
    assert np.all(spectra_list[1][1] == np.array([20.0, 20.0]))  # intensity sample B

    # 3. Check default keyword arguments (matching GlobalMeanSpectrumOptions defaults)
    call_kwargs = mock_combine_mean_spectra.call_args[1]
    assert call_kwargs["binning_p"] == 0.0001
    assert call_kwargs["use_intersection"] is True  # E712 fix
    assert call_kwargs["tic_normalize"] is True  # E712 fix
    assert call_kwargs["compress_axis"] is False  # E712 fix


def test_compute_global_mean_spectrum_passes_custom_options(
    cube_with_mean_spectra: MSICube, mock_combine_mean_spectra: Any
) -> None:
    """Tests that custom options provided as kwargs are passed correctly to the combiner."""
    cube = cube_with_mean_spectra

    custom_kwargs = {
        "binning_p": 0.5,
        "use_intersection": False,
        "tic_normalize": False,
        "compress_axis": True,
    }

    cube.compute_global_mean_spectrum(**custom_kwargs)

    # Check keyword arguments
    call_kwargs = mock_combine_mean_spectra.call_args[1]
    assert call_kwargs["binning_p"] == 0.5
    assert call_kwargs["use_intersection"] is False  # E712 fix
    assert call_kwargs["tic_normalize"] is False  # E712 fix
    assert call_kwargs["compress_axis"] is True  # E712 fix


def test_compute_global_mean_spectrum_stores_results_and_options(
    cube_with_mean_spectra: MSICube, mock_combine_mean_spectra: Any
) -> None:
    """Tests that the results and the options are correctly stored in adata.uns."""
    cube = cube_with_mean_spectra

    # Call with a custom option to ensure the stored options reflect it
    cube.compute_global_mean_spectrum(tic_normalize=False)

    # 1. Check result storage
    assert cube.adata is not None
    assert "mean_spectrum_global" in cube.adata.uns
    global_spectrum = cube.adata.uns["mean_spectrum_global"]

    # The stored result should match the mocked return value
    assert np.all(global_spectrum["mz"] == np.array([100.0, 101.0, 102.0]))
    assert np.all(global_spectrum["intensity"] == np.array([15.0, 15.0, 15.0]))

    # 2. Check options provenance
    assert "mean_spectrum_global_options" in cube.adata.uns
    stored_options = cube.adata.uns["mean_spectrum_global_options"]

    # Verify that the stored options reflect the GlobalMeanSpectrumOptions dataclass,
    # including the custom non-default value.
    assert stored_options["tic_normalize"] is False  # E712 fix
    assert stored_options["use_intersection"] is True  # E712 fix
    assert stored_options["binning_p"] == 0.0001  # Default value


def test_compute_global_mean_spectrum_raises_value_error_on_invalid_options(
    cube_with_mean_spectra: MSICube, mock_combine_mean_spectra: Any
) -> None:
    """Tests that a ValueError is raised if GlobalMeanSpectrumOptions validation fails."""
    cube = cube_with_mean_spectra

    invalid_kwargs = {
        "binning_p": 0.0,  # Should fail validation (must be > 0)
    }

    with pytest.raises(ValueError, match="binning_p must be strictly positive"):
        cube.compute_global_mean_spectrum(**invalid_kwargs)

    # The external function should not be called if validation fails
    assert mock_combine_mean_spectra.call_count == 0
