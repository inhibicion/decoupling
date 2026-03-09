import numpy as np
import pandas as pd
from decoupling.models import consensus

#----------------------------------------
def test_merge_unstable_clusters_basic():
    """
    Test that merge_unstable_clusters merges clusters smaller than min_size
    into larger clusters and reindexes labels properly.
    """
    labels = np.array([0, 0, 1, 1, 2])
    cc_rates = np.array([
        [1, 0.2, 0.1],
        [0.2, 1, 0.3],
        [0.1, 0.3, 1]
    ])
    min_size = 2
    merged = consensus.merge_unstable_clusters(labels.copy(), cc_rates, min_size=min_size)
    # Cluster 2 should be merged
    assert set(merged) <= set([0, 1])
    # Labels should be consecutive integers starting from 0
    assert np.all(np.unique(merged) == np.arange(len(np.unique(merged))))

#----------------------------------------
def test_merge_unstable_clusters_no_merge():
    """
    Ensure merge_unstable_clusters leaves labels unchanged
    when all clusters satisfy min_size.
    """
    labels = np.array([0, 0, 1, 1])
    cc_rates = np.ones((2, 2))
    min_size = 1
    merged = consensus.merge_unstable_clusters(labels.copy(), cc_rates, min_size=min_size)
    assert np.array_equal(merged, labels)

#----------------------------------------
def test_optimal_weight_basic():
    """
    Test that optimal_weight returns a valid weight from the provided options
    and a combined feature matrix of correct shape.
    """
    Morph = pd.DataFrame(np.ones((5, 2)))
    Ephys = pd.DataFrame(np.ones((5, 2)))
    clust_labels = np.array([0, 0, 1, 1, 0])
    weights = [0.5, 1.0]
    w, X = consensus.optimal_weight(Morph, Ephys, clust_labels, weights, random_state=0)
    assert w in weights
    assert X.shape == (5, 4)  # combined matrix has Morph + Ephys features

#----------------------------------------
def test_consensus_clustering_fit_predict():
    """
    Test that ConsensusClustering.fit_predict runs correctly on synthetic data
    and produces output of expected shape. All internal steps are patched
    to avoid heavy clustering computation.
    """
    # Synthetic feature data
    Morph = pd.DataFrame(np.random.rand(6, 3))
    Ephys = pd.DataFrame(np.random.rand(6, 2))
    # Patch internal functions to simplify execution
    def dummy_align_inputs(Morph, Ephys):
        return Morph, Ephys, list(range(len(Morph)))
    def dummy_preprocess_features(Features, **kwargs):
        return Features
    def dummy_all_cluster_calls(**kwargs):
        # Return a DataFrame with one column per weight
        return pd.DataFrame(np.column_stack([np.arange(len(kwargs["specimen_ids"]))]*len(kwargs["weights"])))
    def dummy_consensus_clusters(results, min_clust_size):
        labels = np.zeros(results.shape[0], dtype=int)
        cc_rates = np.ones((results.shape[0], results.shape[0]))
        return labels, None, cc_rates
    def dummy_merge_unstable_clusters(clust_labels, cc_rates, min_size):
        return clust_labels
    def dummy_optimal_weight(Morph, Ephys, clust_labels, weights, random_state):
        return weights[0], np.hstack([Morph.values, Ephys.values])
    def dummy_remap_labels(X, labels):
        return labels
    # Patch the module functions
    consensus.align_inputs = dummy_align_inputs
    consensus.preprocess_features = dummy_preprocess_features
    consensus.all_cluster_calls = dummy_all_cluster_calls
    consensus.consensus_clusters = dummy_consensus_clusters
    consensus.merge_unstable_clusters = dummy_merge_unstable_clusters
    consensus.optimal_weight = dummy_optimal_weight
    consensus.remap_labels = dummy_remap_labels
    # Fit the model and predict labels
    model = consensus.ConsensusClustering(weights=[0.5, 1.0], random_state=0)
    labels = model.fit_predict(Morph, Ephys)
    # Output should match input sample size
    assert labels.shape[0] == Morph.shape[0]
    # Optimal weight should be set to the first weight in patched function
    assert model.optimal_weight_ == 0.5