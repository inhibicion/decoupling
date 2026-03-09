import numpy as np
import pandas as pd
from decoupling.models.hierarchical import HierarchicalClustering

#----------------------------------------
def generate_dummy_features(n_neurons=100, n_morph=5, n_ephys=4, seed=0):
    """
    Generate dummy morphological and electrophysiological feature DataFrames
    for testing clustering algorithms.

    Parameters
    ----------
    n_neurons : int
        Number of neurons (rows) to generate.
    n_morph : int
        Number of morphological features.
    n_ephys : int
        Number of electrophysiological features.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    morph : pd.DataFrame
        Dummy morphological features with one column renamed to 'soma_depth'.
    ephys : pd.DataFrame
        Dummy electrophysiological features.
    """
    rng = np.random.default_rng(seed)
    morph = pd.DataFrame(
        rng.normal(size=(n_neurons, n_morph)) * 2,
        columns=[f"m{i}" for i in range(n_morph)]
    )
    # Rename one existing column to 'soma_depth'
    morph = morph.rename(columns={"m0": "soma_depth"})
    ephys = pd.DataFrame(
        rng.normal(size=(n_neurons, n_ephys)) * 2,
        columns=[f"e{i}" for i in range(n_ephys)]
    )
    return morph, ephys

#----------------------------------------
def test_hierarchical_clustering_runs():
    """
    Ensure that HierarchicalClustering can run on dummy data without errors.

    The test checks that fit_predict executes and returns a label array
    with length equal to the number of input neurons.
    """
    morph, ephys = generate_dummy_features()
    model = HierarchicalClustering()
    labels = model.fit_predict(
        Morph=morph,
        Ephys=ephys
    )
    assert len(labels) == len(morph)

#----------------------------------------
def test_hierarchical_labels_are_integers():
    """
    Verify that the labels returned by HierarchicalClustering are integers.

    This ensures that cluster assignments are proper discrete values
    suitable for downstream analyses.
    """
    morph, ephys = generate_dummy_features()
    model = HierarchicalClustering()
    labels = model.fit_predict(
        Morph=morph,
        Ephys=ephys
    )
    assert np.issubdtype(labels.dtype, np.integer)

#----------------------------------------
def test_hierarchical_labels_have_multiple_clusters():
    """
    Check that HierarchicalClustering produces more than one cluster.

    Confirms that the algorithm does not assign all neurons to a single cluster,
    which would indicate a failure in clustering separation.
    """
    morph, ephys = generate_dummy_features()
    model = HierarchicalClustering()
    labels = model.fit_predict(
        Morph=morph,
        Ephys=ephys
    )
    assert len(np.unique(labels)) > 1