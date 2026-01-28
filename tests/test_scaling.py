from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from deltamsi.core.msicube import MSICube


@pytest.fixture
def msicube_with_data(tmp_path: Path) -> MSICube:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    for sample_name in ["sample_a", "sample_b"]:
        (data_dir / f"{sample_name}.imzML").touch()
        (data_dir / f"{sample_name}.ibd").touch()

    cube = MSICube(data_directory=str(data_dir))

    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(["sample_a", "sample_a", "sample_b"]),
            "condition": pd.Categorical(["cond1", "cond1", "cond2"]),
        },
        index=["a0", "a1", "b0"],
    )

    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    cube.adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["mz1", "mz2"]))
    cube.adata.layers["RAW"] = cube.adata.X.copy()

    return cube


def test_scale_ion_images_global(msicube_with_data: MSICube) -> None:
    stats = msicube_with_data.scale_ion_images_zscore(mode="all", return_stats=True)

    assert stats is not None
    assert None in stats

    scaled = msicube_with_data.adata.X
    np.testing.assert_allclose(np.mean(scaled, axis=0), np.zeros(2), atol=1e-6)
    np.testing.assert_allclose(np.std(scaled, axis=0), np.ones(2), atol=1e-6)


def test_scale_ion_images_per_sample(msicube_with_data: MSICube) -> None:
    msicube_with_data.adata.X = np.array([[1.0, 2.0], [2.0, 4.0], [10.0, 20.0]])

    msicube_with_data.scale_ion_images_zscore(mode="per_sample")

    scaled = msicube_with_data.adata.X
    np.testing.assert_allclose(scaled[0], [-1.0, -1.0])
    np.testing.assert_allclose(scaled[1], [1.0, 1.0])
    np.testing.assert_allclose(scaled[2], [0.0, 0.0])


def test_scale_ion_images_condition_copy(msicube_with_data: MSICube) -> None:
    scaled, stats = msicube_with_data.scale_ion_images_zscore(
        mode="per_condition",
        layer="RAW",
        output_layer="SCALED",
        return_stats=True,
        copy=True,
    )

    assert isinstance(stats, dict)
    assert set(stats.keys()) == {"cond1", "cond2"}

    # Original cube remains untouched
    assert "SCALED" not in msicube_with_data.adata.layers

    # Scaled copy contains the requested output layer
    assert "SCALED" in scaled.layers
    np.testing.assert_allclose(
        scaled.layers["SCALED"].mean(axis=0), np.zeros(2), atol=1e-6
    )


def test_scale_ion_images_layer_preserves_source(msicube_with_data: MSICube) -> None:
    raw = msicube_with_data.adata.layers["RAW"].copy()
    msicube_with_data.adata.X = np.full_like(raw, 10.0)

    msicube_with_data.scale_ion_images_zscore(layer="RAW")

    np.testing.assert_allclose(msicube_with_data.adata.layers["RAW"], raw)
    np.testing.assert_allclose(
        msicube_with_data.adata.X.mean(axis=0), np.zeros(2), atol=1e-6
    )


def test_scale_ion_images_condition_requires_column(msicube_with_data: MSICube) -> None:
    msicube_with_data.adata.obs = msicube_with_data.adata.obs.drop(columns=["condition"])

    with pytest.raises(KeyError):
        msicube_with_data.scale_ion_images_zscore(mode="per_condition")
