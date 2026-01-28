import numpy as np
import pandas as pd
import anndata as ad

from deltamsi.core.msicube import MSICube
from deltamsi.processing.spatial_chaos import (
    compute_spatial_chaos_matrix,
    spatial_chaos_fold_change_from_adata,
    spatial_chaos_score,
)


def _build_test_adata() -> ad.AnnData:
    coords = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],  # sample 1 (4 pixels)
            [0, 0],
            [1, 1],  # sample 2 (2 pixels)
        ]
    )

    obs = pd.DataFrame(
        {"sample": ["s1", "s1", "s1", "s1", "s2", "s2"]}
    )
    var = pd.DataFrame(index=["v1", "v2"])

    X = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [1.0, 0.0],
        ]
    )

    return ad.AnnData(X=X, obs=obs, var=var, obsm={"spatial": coords})


def test_spatial_chaos_score_basic_cases() -> None:
    full_image = np.ones((2, 2))
    split_image = np.array([[1.0, 0.0], [0.0, 1.0]])
    empty_image = np.zeros((2, 2))

    assert np.isclose(spatial_chaos_score(full_image, n_thresholds=5), 0.75)
    assert np.isclose(spatial_chaos_score(split_image, n_thresholds=5), 0.0)
    assert np.isclose(spatial_chaos_score(empty_image, n_thresholds=5), 0.0)


def test_compute_spatial_chaos_matrix_returns_per_sample_scores() -> None:
    adata = _build_test_adata()

    chaos, samples = compute_spatial_chaos_matrix(adata, n_thresholds=5)

    assert samples == ["s1", "s2"]
    assert chaos.shape == (adata.n_vars, 2)
    assert np.isclose(chaos[0, 0], 0.75)
    assert np.isclose(chaos[0, 1], 0.0)


def test_msicube_stores_spatial_chaos(tmp_path) -> None:
    adata = _build_test_adata()

    cube = MSICube(data_directory=str(tmp_path))
    cube.adata = adata

    chaos = cube.compute_spatial_chaos_scores(n_thresholds=5, varm_key="chaos")

    assert "chaos" in cube.adata.varm
    assert np.allclose(chaos, cube.adata.varm["chaos"], equal_nan=True)
    assert cube.adata.uns["spatial_chaos"]["samples"] == ["s1", "s2"]
    assert cube.adata.uns["spatial_chaos"]["varm_key"] == "chaos"


def test_spatial_chaos_fold_change_from_adata() -> None:
    adata = _build_test_adata()
    adata.obs["condition"] = ["ctrl", "ctrl", "ctrl", "ctrl", "treated", "treated"]

    chaos, samples = compute_spatial_chaos_matrix(adata, n_thresholds=5)
    adata.varm["chaos"] = chaos
    adata.uns["spatial_chaos"] = {"samples": samples, "sample_key": "sample"}

    fold_change = spatial_chaos_fold_change_from_adata(
        adata,
        groupby="condition",
        control_label="ctrl",
        interaction_label="treated",
        varm_key="chaos",
        eps=1e-6,
    )

    assert np.allclose(fold_change["S_control_max"], [0.75, 0.0], equal_nan=True)
    assert np.allclose(fold_change["S_interaction_max"], [0.0, 0.0], equal_nan=True)
    assert np.allclose(fold_change["FC_S"], [0.0, 0.0])
    assert fold_change["samples"].tolist() == ["s1", "s2"]
    assert fold_change["sample_groups"].tolist() == ["ctrl", "treated"]


def test_msicube_spatial_chaos_fold_change(tmp_path) -> None:
    adata = _build_test_adata()
    adata.obs["condition"] = ["ctrl", "ctrl", "ctrl", "ctrl", "treated", "treated"]

    cube = MSICube(data_directory=str(tmp_path))
    cube.adata = adata

    cube.compute_spatial_chaos_scores(n_thresholds=5, varm_key="chaos")
    fc = cube.compute_spatial_chaos_fold_change(
        groupby="condition",
        control_label="ctrl",
        interaction_label="treated",
        varm_key="chaos",
        result_key="chaos_fc",
    )

    assert np.allclose(fc["FC_S"], [0.0, 0.0])
    assert "chaos_fc_FC_S" in cube.adata.var
    assert cube.adata.uns["chaos_fc"]["sample_groups"] == ["ctrl", "treated"]
