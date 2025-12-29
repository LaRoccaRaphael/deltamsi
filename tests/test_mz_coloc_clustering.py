import numpy as np

from pymsix.processing.mass_clustering import cluster_masses_from_colocalization


def test_coloc_clustering_threshold_splits_components():
    # Graph: (0,1) connected strongly, (2,3) connected strongly, others weak
    S = np.array(
        [
            [0.0, 0.9, 0.1, 0.0],
            [0.9, 0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0, 0.8],
            [0.0, 0.0, 0.8, 0.0],
        ],
        dtype=float,
    )

    res = cluster_masses_from_colocalization(
        S, resolution=1.0, edge_max_delta_cosine=0.5, return_graph=False
    )

    labels = res["labels"]
    assert set(np.unique(labels)) == {0, 1}
    # Each dense component should form its own cluster
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    # Weak edges should be discarded
    assert len(res["edges"]) == 2


def test_coloc_clustering_knn_pruning():
    # Fully connected triangle with different weights
    S = np.array(
        [
            [0.0, 0.9, 0.8],
            [0.9, 0.0, 0.2],
            [0.8, 0.2, 0.0],
        ],
        dtype=float,
    )

    res = cluster_masses_from_colocalization(S, knn_k=1, knn_mode="union")

    # k-NN pruning should remove the weakest edge (1,2)
    edges = {(min(i, j), max(i, j)) for i, j in zip(res["edges"]["i"], res["edges"]["j"])}
    assert len(res["edges"]) == 2
    assert (1, 2) not in edges
    # Leiden should still keep the graph connected
    labels = res["labels"]
    assert len(np.unique(labels)) == 1
