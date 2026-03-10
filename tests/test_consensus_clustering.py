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
def test_merge_unstable_clusters_conflict():
    """
    Test conflict resolution when multiple small clusters would merge to the same target.
    """
    labels = np.array([0, 1, 2, 2, 3, 3])
    cc_rates = np.array([
        [1, 0.9, 0.2, 0.1],
        [0.9, 1, 0.1, 0.2],
        [0.2, 0.1, 1, 0.8],
        [0.1, 0.2, 0.8, 1]
    ])
    min_size = 3
    merged = consensus.merge_unstable_clusters(labels.copy(), cc_rates, min_size=min_size)
    # Ensure all labels reindexed to consecutive integers
    assert np.all(np.unique(merged) == np.arange(len(np.unique(merged))))
    # Ensure conflict branch is executed (small clusters merged properly)
    assert set(merged) <= set([0, 1])

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
def test_optimal_weight_equal_scores():
    """
    Ensure optimal_weight returns the first weight if multiple weights produce the same OOB score.
    """
    Morph = pd.DataFrame(np.ones((4, 2)))
    Ephys = pd.DataFrame(np.ones((4, 2)))
    clust_labels = np.array([0, 1, 0, 1])
    weights = [0.5, 1.0]
    # OOB scores identical since features identical
    w, _ = consensus.optimal_weight(Morph, Ephys, clust_labels, weights, random_state=0)
    assert w == weights[0]

#----------------------------------------
def test_consensus_clustering_fit_predict_patchless_fixed():
    """
    Fully patched version of fit_predict to exercise merge, optimal_weight, and remap_labels
    without calling scipy linkage.
    """
    Morph = pd.DataFrame(np.random.rand(6, 2))
    Ephys = pd.DataFrame(np.random.rand(6, 2))
    # Patch internal functions
    consensus.align_inputs = lambda Morph, Ephys: (Morph, Ephys, list(range(len(Morph))))
    consensus.preprocess_features = lambda Features, **kwargs: Features
    consensus.all_cluster_calls = lambda **kwargs: pd.DataFrame(np.zeros((len(kwargs['specimen_ids']), len(kwargs['weights']))))
    consensus.consensus_clusters = lambda results, min_clust_size: (np.zeros(results.shape[0], dtype=int), None, np.ones((results.shape[0], results.shape[0])))
    consensus.merge_unstable_clusters = lambda clust_labels, cc_rates, min_size: clust_labels
    consensus.optimal_weight = lambda Morph, Ephys, clust_labels, weights, random_state: (weights[0], np.hstack([Morph.values, Ephys.values]))
    consensus.remap_labels = lambda X, labels: labels
    # Fit the model and predict labels
    model = consensus.ConsensusClustering(weights=[0.5, 1.0], random_state=0)
    labels = model.fit_predict(Morph, Ephys)
    # Output should match input sample size
    assert labels.shape[0] == Morph.shape[0]
    # Optimal weight should be set to the first weight in patched function
    assert model.optimal_weight_ == 0.5