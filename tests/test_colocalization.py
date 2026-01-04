import numpy as np
import anndata as ad
import pandas as pd
from pathlib import Path
import pytest

from pymsix.core.msicube import MSICube
from pymsix.processing.colocalization import CosineColocParams, compute_mz_cosine_colocalization


@pytest.fixture
def mock_imzml_data(tmp_path: Path) -> str:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    (data_dir / "sample_A.imzML").touch()
    (data_dir / "sample_A.ibd").touch()
    (data_dir / "sample_B.imzml").touch()
    (data_dir / "sample_B.ibd").touch()

    return str(data_dir)


def test_cosine_dense_storage(mock_imzml_data: str) -> None:
    cube = MSICube(data_directory=mock_imzml_data)
    cube.adata = ad.AnnData(
        X=np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        obs=pd.DataFrame(index=["p1", "p2"]),
        var=pd.DataFrame(index=["mz1", "mz2"]),
    )

    S = cube.compute_cosine_colocalization(
        params=CosineColocParams(mode="dense", include_self=False)
    )

    expected = np.array([[0.0, 0.70710677], [0.70710677, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(S, expected)
    assert cube.adata is not None
    assert "ion_cosine" in cube.adata.varp
    assert "ion_cosine_params" in cube.adata.uns


def test_cosine_topk_sparse(mock_imzml_data: str) -> None:
    cube = MSICube(data_directory=mock_imzml_data)
    cube.adata = ad.AnnData(
        X=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        obs=pd.DataFrame(index=["p1", "p2"]),
        var=pd.DataFrame(index=["mz1", "mz2", "mz3"]),
    )

    params = CosineColocParams(
        mode="topk_sparse",
        topk=1,
        min_sim=0.1,
        symmetrize=True,
        store_varp_key="cosine_topk",
    )
    S = compute_mz_cosine_colocalization(cube, params=params)

    assert cube.adata is not None
    assert S.shape == (3, 3)
    assert S.nnz > 0
    assert cube.adata.varp["cosine_topk"].shape == (3, 3)
