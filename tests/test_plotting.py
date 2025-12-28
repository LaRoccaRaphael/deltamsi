import pytest
import numpy as np
import pandas as pd
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
def test_plot_mean_spectrum_windows_success(
    mock_show: MagicMock, cube_with_spectra: MSICube
) -> None:
    """Test successful data extraction and plotting call."""
    plot_mean_spectrum_windows(
        cube_with_spectra,
        labels=["sample_A", "sample_B"],
        peak_mzs=[200.0, 500.0],
        span_da=0.1,
        tol_ppm=10.0,
    )


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


class DummyAxis:
    def __init__(self) -> None:
        self.imshow_calls = []
        self.titles = []

    def imshow(self, img: np.ndarray, **_: Any) -> MagicMock:
        self.imshow_calls.append(img)
        return MagicMock()

    def set_aspect(self, *_: Any, **__: Any) -> None:
        return None

    def set_title(self, title: str, **_: Any) -> None:
        self.titles.append(title)

    def tick_params(self, *_: Any, **__: Any) -> None:
        return None

    def set_xlabel(self, *_: Any, **__: Any) -> None:
        return None

    def set_ylabel(self, *_: Any, **__: Any) -> None:
        return None

    def axis(self, *_: Any, **__: Any) -> None:
        return None

    def set_visible(self, *_: Any, **__: Any) -> None:
        return None


class DummyFormatter:
    def set_powerlimits(self, *_: Any, **__: Any) -> None:
        return None


class DummyColorbar:
    def __init__(self) -> None:
        self.ax = MagicMock()
        self.formatter = DummyFormatter()


class DummyFigure:
    def __init__(self) -> None:
        self.colorbars = []

    def suptitle(self, *_: Any, **__: Any) -> None:
        return None

    def colorbar(self, *_: Any, **__: Any) -> DummyColorbar:
        cb = DummyColorbar()
        self.colorbars.append(cb)
        return cb


class DummyDivider:
    def append_axes(self, *_: Any, **__: Any) -> MagicMock:
        return MagicMock()


@patch("matplotlib.pyplot.show")
def test_plot_ion_images_with_aggregated_labels(mock_show: MagicMock) -> None:
    cube = MSICube(data_directory=".")
    cube.adata = ad.AnnData(
        X=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        obs=pd.DataFrame({"sample": ["s1", "s1"]}, index=["p0", "p1"]),
        var=pd.DataFrame(
            {"mz": [100.0, 101.0, 102.0], "label": ["g1", "g1", "g2"]},
            index=["mz1", "mz2", "mz3"],
        ),
        obsm={"spatial": np.array([[0, 0], [1, 0]])},
    )

    cube.aggregate_vars_by_label("label", obsm_key="X_label_mean")

    dummy_axis = DummyAxis()
    dummy_fig = DummyFigure()

    with patch("matplotlib.pyplot.subplots", return_value=(dummy_fig, dummy_axis)), patch(
        "mpl_toolkits.axes_grid1.make_axes_locatable", return_value=DummyDivider()
    ):
        cube.plot_ion_images(mz="g1", samples="s1", obsm_key="X_label_mean")

    np.testing.assert_array_equal(dummy_axis.imshow_calls[0], np.array([[1.5, 4.5]]))
    assert dummy_axis.titles[0] == "s1\ng1"
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_ion_images_with_custom_obsm_key(mock_show: MagicMock) -> None:
    cube = MSICube(data_directory=".")
    cube.adata = ad.AnnData(
        X=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        obs=pd.DataFrame({"sample": ["s1", "s1"]}, index=["p0", "p1"]),
        var=pd.DataFrame(
            {"mz": [100.0, 101.0, 102.0], "label": ["unlabeled", "region_1", "region_1"]},
            index=["mz1", "mz2", "mz3"],
        ),
        obsm={"spatial": np.array([[0, 0], [1, 0]])},
    )

    cube.aggregate_vars_by_label("label", obsm_key="manual_annot")

    dummy_axis = DummyAxis()
    dummy_axis_2 = DummyAxis()
    dummy_fig = DummyFigure()

    axes = np.array([dummy_axis, dummy_axis_2], dtype=object)

    with patch("matplotlib.pyplot.subplots", return_value=(dummy_fig, axes)), patch(
        "mpl_toolkits.axes_grid1.make_axes_locatable", return_value=DummyDivider()
    ):
        cube.plot_ion_images(mz=["unlabeled", "region_1"], samples="s1", obsm_key="manual_annot")

    np.testing.assert_array_equal(dummy_axis.imshow_calls[0], np.array([[1.0, 4.0]]))
    np.testing.assert_array_equal(dummy_axis_2.imshow_calls[0], np.array([[2.5, 5.5]]))
    assert dummy_axis.titles[0] == "unlabeled"
    assert dummy_axis_2.titles[0] == "region_1"
    assert mock_show.called
