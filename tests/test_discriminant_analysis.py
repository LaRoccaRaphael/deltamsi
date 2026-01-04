import numpy as np
import pandas as pd
import anndata as ad

from pymsix.processing import RankIonsMSIParams, rank_ions_groups_msi


def test_rank_ions_groups_msi_replicated() -> None:
    X = np.array(
        [
            [5.0, 1.0],
            [6.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=["ion1", "ion2"]))
    adata.obs["condition"] = ["treated", "treated", "control", "control"]
    adata.obs["sample"] = ["s1", "s2", "s3", "s4"]
    adata.var["mz"] = [100.0, 200.0]

    params = RankIonsMSIParams()
    res = rank_ions_groups_msi(adata, params=params)

    assert list(res["ion"]) == ["ion1", "ion2"]
    assert adata.uns[params.key_added]["params"]["analysis_mode"] == "pseudobulk_samples"
    assert np.isfinite(adata.uns[params.key_added]["pvals"]).any()
    assert adata.uns[params.key_added]["top_table"].equals(res)


def test_rank_ions_groups_msi_single_sample_effects() -> None:
    X = np.array(
        [
            [4.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=float,
    )
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=["g1", "g2"]))
    adata.obs["condition"] = ["treated", "control"]
    adata.obs["sample"] = ["s1", "s2"]

    params = RankIonsMSIParams(n_top=2)
    res = rank_ions_groups_msi(adata, params=params)

    assert adata.uns[params.key_added]["params"]["analysis_mode"] == "single_sample_effects"
    assert np.all(np.isnan(adata.uns[params.key_added]["pvals"]))
    assert res.shape[0] == 2
    assert np.allclose(res["log2fc"], res["log2fc_pixels"])
