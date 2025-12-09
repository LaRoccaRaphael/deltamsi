import pytest
import numpy as np
import anndata as ad
from unittest.mock import MagicMock, patch
from pymsix.core.msicube import MSICube
from pymsix.plotting.spectrum import plot_mean_spectrum_windows
from typing import Any


@pytest.fixture
def cube_with_spectra(tmp_path: str) -> MSICube:
    """Creates a mock MSICube populated with mean spectra."""
    cube = MSICube(data_directory=str(tmp_path))
    cube.adata = ad.AnnData()

    # Create mock spectra
    mz = np.linspace(100, 1000, 100)
    int1 = np.random.rand(100)
    int2 = np.random.rand(100)

    cube.adata.uns["mean_spectra"] = {
        "sample_A": {"mz": mz, "intensity": int1},
        "sample_B": {"mz": mz, "intensity": int2},
    }
    return cube


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.subplots")
def test_plot_mean_spectrum_windows_success(
    mock_subplots: MagicMock, mock_show: MagicMock, cube_with_spectra: MSICube
) -> None:
    """Test successful data extraction and plotting call."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, np.array([mock_ax]))

    plot_mean_spectrum_windows(
        cube_with_spectra,
        labels=["sample_A", "sample_B"],
        peak_mzs=[200.0, 500.0],
        span_da=0.1,
        tol_ppm=10.0,
    )

    assert mock_subplots.called
    assert mock_show.called


def test_plot_mean_spectrum_windows_missing_sample(
    cube_with_spectra: MSICube, capsys: Any
) -> None:
    """Test that missing samples are skipped and warned, but plotting continues."""
    # We patch the core function to verify it receives only valid data
    with patch(
        "pymsix.plotting.spectrum._plot_mean_spectrum_windows_core"
    ) as mock_core:
        plot_mean_spectrum_windows(
            cube_with_spectra,
            labels=["sample_A", "missing_sample"],
            peak_mzs=[200.0],
            span_da=0.1,
            tol_da=0.01,
        )

        # Check warning
        captured = capsys.readouterr()
        assert "WARNING: Sample 'missing_sample' not found" in captured.out
        assert mock_core.called

        # Check that core was called with only 1 sample
        # Accessing arguments via kwargs instead of positional args
        _, kwargs = mock_core.call_args
        # We assume the mean spectra list is passed under the key 'mean_spectra'
        mean_spectra_arg = kwargs["mean_spectra"]

        assert len(mean_spectra_arg) == 1

        # FIX: The previous assertion failed due to NumPy ambiguity (comparing array to string).
        # We now assume the labels are passed separately as 'labels'.
        assert "labels" in kwargs
        assert kwargs["labels"] == [
            "sample_A"
        ]  # Check that only the valid label is passed

        # Check that the spectrum tuple structure is correct (mz_array, intensity_array)
        spectrum_tuple = mean_spectra_arg[0]
        assert len(spectrum_tuple) == 2
        assert isinstance(spectrum_tuple[0], np.ndarray)  # Check mz_array type
        assert isinstance(spectrum_tuple[1], np.ndarray)  # Check intensity_array type

        # Check that peak_mzs and span_da are also passed correctly
        np.testing.assert_array_equal(kwargs["peak_mzs"], [200.0])
        assert kwargs["span_da"] == 0.1

        # Check that the number of arguments passed through is correct (4 items: cube, labels, peak_mzs, span_da + **kwargs)
        # Note: Depending on how the wrapper passes kwargs, you might also want to check extra kwargs like 'tol_da'
        assert "tol_da" in kwargs
        assert kwargs["tol_da"] == 0.01


def test_plot_mean_spectrum_windows_no_data(cube_with_spectra: MSICube) -> None:
    """Test ValueError if no valid samples are found."""
    with pytest.raises(ValueError, match="No valid samples found"):
        plot_mean_spectrum_windows(
            cube_with_spectra,
            labels=["missing_1", "missing_2"],
            peak_mzs=[200.0],
            span_da=0.1,
            tol_da=0.01,
        )
