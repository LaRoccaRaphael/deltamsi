from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from deltamsi.core.msicube import MSICube


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
def cube_with_matrix(mock_imzml_data: str) -> MSICube:
    cube = MSICube(data_directory=mock_imzml_data)
    cube.adata = ad.AnnData(
        X=np.array([[1.0, 1.0], [2.0, 0.0]]),
        obs=pd.DataFrame(index=["p0", "p1"]),
        var=pd.DataFrame(index=["mz1", "mz2"]),
    )
    return cube


def test_tic_normalize_dense_matrix(cube_with_matrix: MSICube) -> None:
    cube_with_matrix.tic_normalize(target_sum=10.0, store_tic_in_obs="tic_before")

    assert np.allclose(cube_with_matrix.adata.X.sum(axis=1), [10.0, 10.0])
    assert list(cube_with_matrix.adata.obs["tic_before"]) == [2.0, 2.0]
    assert cube_with_matrix.adata.uns["tic_normalize"]["target_sum"] == 10.0


def test_tic_normalize_sparse_layer_copy(mock_imzml_data: str) -> None:
    cube = MSICube(data_directory=mock_imzml_data)
    cube.adata = ad.AnnData(
        X=np.zeros((2, 2)),
        obs=pd.DataFrame(index=["p0", "p1"]),
        var=pd.DataFrame(index=["mz1", "mz2"]),
    )
    raw_layer = sp.csr_matrix([[0.0, 0.0], [3.0, 3.0]])
    cube.adata.layers["RAW"] = raw_layer

    normalized_cube = cube.tic_normalize(
        layer="RAW", target_sum=6.0, store_tic_in_obs=None, copy=True
    )

    assert normalized_cube is not None and normalized_cube is not cube
    assert np.allclose(
        normalized_cube.adata.layers["RAW"].sum(axis=1).A.ravel(), [6.0, 6.0]
    )
    assert cube.adata.layers["RAW"].sum() == raw_layer.sum()
