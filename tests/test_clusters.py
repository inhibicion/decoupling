import pytest
import numpy as np
import pandas as pd
from decoupling.metrics import clusters

#----------------------------------------
def test_count_per_cluster_basic():
    """
    Verify that count_per_cluster correctly counts elements per cluster
    in a basic multi-cluster input.
    """
    labels = [0, 0, 1, 1, 1, 2]
    result = clusters.count_per_cluster(labels)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [0, 1, 2]
    assert result.iloc[0, 0] == 2
    assert result.iloc[0, 1] == 3
    assert result.iloc[0, 2] == 1

#----------------------------------------
def test_count_per_cluster_empty():
    """
    Ensure count_per_cluster handles empty input gracefully,
    returning an empty DataFrame.
    """
    labels = []
    result = clusters.count_per_cluster(labels)
    assert isinstance(result, pd.DataFrame)
    assert result.empty or result.shape[1] == 0

#----------------------------------------
def test_count_per_cluster_single_label():
    """
    Test that count_per_cluster works when all elements belong
    to the same cluster.
    """
    labels = [7, 7, 7]
    result = clusters.count_per_cluster(labels)
    assert list(result.columns) == [7]
    assert result.iloc[0, 0] == 3

#----------------------------------------
def test_cluster_predictability_shape():
    """
    Verify cluster_predictability returns a DataFrame of expected structure
    for original labels without shuffling.
    """
    X = pd.DataFrame(np.random.rand(10, 3))
    y = np.array([0, 1] * 5)
    df = clusters.cluster_predictability(X, y, k=2, shuffle=False)
    assert isinstance(df, pd.DataFrame)
    assert "classifier" in df.columns
    assert "accuracy" in df.columns
    assert "class_labels" in df.columns
    assert all(df["class_labels"] == "original")

#----------------------------------------
def test_cluster_predictability_with_shuffle():
    """
    Verify that cluster_predictability includes both 'original'
    and 'shuffled' labels when shuffle=True.
    """
    X = pd.DataFrame(np.random.rand(10, 2))
    y = np.array([0, 1] * 5)
    df = clusters.cluster_predictability(X, y, k=2, shuffle=True)
    assert set(df["class_labels"]) == {"original", "shuffled"}

#----------------------------------------
def test_get_default_classifiers_returns_list():
    """
    Ensure get_default_classifiers returns a list of sklearn-like
    classifier objects that implement a fit method.
    """
    clfs = clusters.get_default_classifiers()
    assert isinstance(clfs, list)
    assert all(hasattr(clf, "fit") for clf in clfs)

#----------------------------------------
@pytest.mark.parametrize("labels", [
    [0],
    [1, 1, 1],
    [],
])
def test_count_per_cluster_various_labels(labels):
    """
    Parameterized test to check count_per_cluster handles various edge cases:
    single-element clusters, repeated labels, and empty lists.
    """
    df = clusters.count_per_cluster(labels)
    assert isinstance(df, pd.DataFrame)