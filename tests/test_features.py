import numpy as np
import pandas as pd
from scipy.stats import zscore
from decoupling.metrics.features import variance_per_sample_size

#----------------------------------------
def test_zscore_preserves_shape():
    """
    Ensure that applying z-score normalization does not change
    the shape of the original DataFrame.
    """
    X = pd.DataFrame(
        np.random.randn(20, 5),
        columns=[f"f{i}" for i in range(5)]
    )
    Z = zscore(X)
    assert Z.shape == X.shape

#----------------------------------------
def test_dataframe_index_alignment():
    """
    Verify that z-score normalization preserves the original
    DataFrame's index alignment.
    """
    X = pd.DataFrame(
        np.random.randn(20, 3),
        columns=["a", "b", "c"]
    )
    Z = pd.DataFrame(
        zscore(X),
        columns=X.columns,
        index=X.index
    )
    assert (Z.index == X.index).all()

#----------------------------------------
def test_variance_per_sample_size_basic():
    """
    Test that variance_per_sample_size computes summary statistics
    correctly for basic feature matrices.
    Checks that the output DataFrame has expected columns and
    non-negative values.
    """
    X = pd.DataFrame({
        'f1': np.arange(10),
        'f2': np.arange(10, 20),
        'f3': np.arange(20, 30)
    })
    sample_sizes = [5, 10]
    df = variance_per_sample_size(X, sample_sizes, n_components=2, n_shuffles=3)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["sample_size", "mean", "ci_95"]
    assert set(df["sample_size"]) == set(sample_sizes)
    assert all(df["mean"] >= 0)
    assert all(df["ci_95"] >= 0)

#----------------------------------------
def test_variance_per_sample_size_with_nans():
    """
    Check that variance_per_sample_size handles NaN values
    without throwing errors and returns a DataFrame of expected length.
    """
    X = pd.DataFrame({
        'f1': [1, 2, np.nan, 4],
        'f2': [np.nan, 1, 2, 3],
        'f3': [0, 1, 2, 3]
    })
    sample_sizes = [2]
    df = variance_per_sample_size(X, sample_sizes, n_components=2, n_shuffles=2)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == len(sample_sizes)