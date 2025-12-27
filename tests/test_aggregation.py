from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pymsix.core.msicube import MSICube


@pytest.fixture
def mock_imzml_data(tmp_path: Path) -> str:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    (data_dir / "sample_A.imzML").touch()
    (data_dir / "sample_A.ibd").touch()
    (data_dir / "sample_B.imzml").touch()
    (data_dir / "sample_B.ibd").touch()

    return str(data_dir)


@pytest.fixture
def cube_with_labels(mock_imzml_data: str) -> MSICube:
    cube = MSICube(data_directory=mock_imzml_data)
    cube.adata = ad.AnnData(
        X=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        obs=pd.DataFrame(index=["p0", "p1"]),
        var=pd.DataFrame({"label": ["g1", "g1", "g2"]}, index=["mz1", "mz2", "mz3"]),
    )
    return cube


def test_aggregate_vars_by_label_dense(cube_with_labels: MSICube) -> None:
    labels = cube_with_labels.aggregate_vars_by_label(
        "label", obsm_key="X_label_mean", agg="mean"
    )

    assert list(labels) == ["g1", "g2"]
    np.testing.assert_allclose(
        cube_with_labels.adata.obsm["X_label_mean"],
        np.array([[1.5, 3.0], [4.5, 6.0]], dtype=np.float32),
    )
    assert cube_with_labels.adata.uns["X_label_mean_labels"] == ["g1", "g2"]
    assert cube_with_labels.adata.uns["X_label_mean_source"] == {
        "label_col": "label",
        "layer": None,
        "agg": "mean",
    }


def test_aggregate_vars_by_label_sparse_median_error(cube_with_labels: MSICube) -> None:
    cube_with_labels.adata.layers["RAW"] = sp.csr_matrix(cube_with_labels.adata.X)

    with pytest.raises(ValueError, match="median aggregation is not supported"):
        cube_with_labels.aggregate_vars_by_label("label", layer="RAW", agg="median")
