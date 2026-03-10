import numpy as np
import pandas as pd
import decoupling.models.ephys_morph_clustering as emc

#----------------------------------------
def toy_data(n_cells=20, morph_dim=3, ephys_dim=2):
    """Generate a small deterministic dataset for clustering tests."""
    rng = np.random.default_rng(0)

    specimen_ids = np.arange(n_cells)
    morph_X = rng.normal(size=(n_cells, morph_dim))
    ephys_spca = pd.DataFrame(rng.normal(size=(n_cells, ephys_dim)))

    return specimen_ids, morph_X, ephys_spca

#----------------------------------------
def test_hc_nn_cluster_calls_returns_expected_columns():
    """Check hc_nn_cluster_calls adds the correct column for KNN clustering."""
    specimen_ids, morph_X, ephys_spca = toy_data()
    df = pd.DataFrame({"specimen_id": specimen_ids}).set_index("specimen_id")
    res = emc.hc_nn_cluster_calls(df, morph_X, ephys_spca, n_nn=[3], n_cl=[4])
    assert "hc_conn_3_4" in res.columns
    assert len(res) == len(specimen_ids)

#----------------------------------------
def test_hc_combo_cluster_calls_adds_columns():
    """Check hc_combo_cluster_calls adds the expected weighted combination column."""
    specimen_ids, morph_X, ephys_spca = toy_data()
    df = pd.DataFrame({"specimen_id": specimen_ids}).set_index("specimen_id")
    res = emc.hc_combo_cluster_calls(df, morph_X, ephys_spca, weights=[2], n_cl=[3])
    assert "hc_combo_2_3" in res.columns
    assert res.shape[0] == len(specimen_ids)

#----------------------------------------
def test_gmm_combo_cluster_calls_label_range():
    """Verify GMM clustering produces labels in the valid range [0, n_clusters-1]."""
    specimen_ids, morph_X, ephys_spca = toy_data()
    df = pd.DataFrame({"specimen_id": specimen_ids}).set_index("specimen_id")
    res = emc.gmm_combo_cluster_calls(df, morph_X, ephys_spca, weights=[1], n_cl=[3])
    labels = res["gmm_combo_1_3"].values
    assert labels.min() >= 0
    assert labels.max() < 3

#----------------------------------------
def test_spectral_combo_cluster_calls_runs():
    """Ensure spectral_combo_cluster_calls executes and adds the expected column."""
    specimen_ids, morph_X, ephys_spca = toy_data()
    df = pd.DataFrame({"specimen_id": specimen_ids}).set_index("specimen_id")
    res = emc.spectral_combo_cluster_calls(df, morph_X, ephys_spca, weights=[1], n_cl=[3])
    assert "spec_combo_1_3" in res.columns

#----------------------------------------
def test_all_cluster_calls_combines_methods():
    """Check that all_cluster_calls runs all clustering methods and adds all expected columns."""
    specimen_ids, morph_X, ephys_spca = toy_data()
    res = emc.all_cluster_calls(
        specimen_ids,
        morph_X,
        ephys_spca,
        weights=[1],
        n_cl=[3],
        n_nn=[3],
    )
    expected_cols = {
        "hc_conn_3_3",
        "hc_combo_1_3",
        "gmm_combo_1_3",
        "spec_combo_1_3",
    }
    assert expected_cols.issubset(set(res.columns))

#----------------------------------------
def test_consensus_clusters_returns_valid_output():
    """Verify consensus_clusters returns labels and matrices of correct shapes."""
    rng = np.random.default_rng(0)
    results = rng.integers(0, 4, size=(20, 5))
    labels, shared, cc_rates = emc.consensus_clusters(results)
    assert len(labels) == 20
    assert shared.shape == (20, 20)
    assert cc_rates.shape[0] == cc_rates.shape[1]

#----------------------------------------
def test_coclust_rates_matrix_shape():
    """Ensure coclust_rates produces a square matrix of size equal to the number of unique clusters."""
    rng = np.random.default_rng(0)
    shared = rng.random((10, 10))
    labels = rng.integers(0, 3, size=10)
    rates = emc.coclust_rates(shared, labels)
    k = len(np.unique(labels))
    assert rates.shape == (k, k)