from typing import Any
import pytest
import numpy as np
import anndata as ad
from unittest.mock import MagicMock, patch
from msix.core.msicube import MSICube
from msix.plotting.spectrum import plot_mean_spectrum_windows


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
    with patch("msix.plotting.spectrum._plot_mean_spectrum_windows_core") as mock_core:
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

        # Check that core was called with only 1 sample
        args, _ = mock_core.call_args
        mean_spectra_arg = args[0]  # First arg is mean_spectra list
        labels_arg = (
            args[7] if len(args) > 7 else mock_core.call_args.kwargs["labels"]
        )  # labels arg

        assert len(mean_spectra_arg) == 1
        assert labels_arg == ["sample_A"]


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
