# import pytest
# import numpy as np
# import pandas as pd
# from decoupling.utils.preprocessing import (
#     set_seed,
#     align_inputs,
#     preprocess_features,
#     get_low_variance_features,
#     get_high_correlation_features,
# )
# from decoupling.utils.postprocessing import remap_labels

# #----------------------------------------
# def test_set_seed_consistency():
#     """
#     Verify that set_seed reproducibly sets the random seed for NumPy
#     such that identical calls produce the same random numbers.
#     """
#     seed = set_seed(42, verbose=False)
#     arr1 = np.random.rand(3)
#     set_seed(42, verbose=False)
#     arr2 = np.random.rand(3)
#     np.testing.assert_array_equal(arr1, arr2)

# #----------------------------------------
# def test_align_inputs_basic():
#     """
#     Ensure align_inputs correctly aligns two DataFrames with identical indices,
#     returning both DataFrames unchanged and the original index as an array.
#     """
#     df1 = pd.DataFrame({"a":[1,2,3]}, index=["s1","s2","s3"])
#     df2 = pd.DataFrame({"b":[4,5,6]}, index=["s1","s2","s3"])
#     M, E, ids = align_inputs(df1, df2)
#     pd.testing.assert_frame_equal(M, df1)
#     pd.testing.assert_frame_equal(E, df2)
#     np.testing.assert_array_equal(ids, np.array(["s1","s2","s3"]))

# #----------------------------------------
# def test_align_inputs_partial_overlap():
#     """
#     Verify that align_inputs returns only the overlapping indices
#     when the input DataFrames partially overlap.
#     """
#     df1 = pd.DataFrame({"a":[1,2,3]}, index=["s1","s2","s3"])
#     df2 = pd.DataFrame({"b":[4,5,6]}, index=["s2","s3","s4"])
#     M, E, ids = align_inputs(df1, df2)
#     np.testing.assert_array_equal(ids, np.array(["s2","s3"]))
#     assert list(M.index) == ["s2","s3"]
#     assert list(E.index) == ["s2","s3"]

# #----------------------------------------
# def test_align_inputs_no_overlap_raises():
#     """
#     Confirm that align_inputs raises a ValueError when the DataFrames
#     have no overlapping indices.
#     """
#     df1 = pd.DataFrame({"a":[1]}, index=["s1"])
#     df2 = pd.DataFrame({"b":[2]}, index=["s2"])
#     with pytest.raises(ValueError):
#         align_inputs(df1, df2)

# #----------------------------------------
# def test_preprocess_features_filters():
#     """
#     Ensure preprocess_features correctly filters out low-variance
#     features and highly correlated features according to thresholds.
#     """
#     X = pd.DataFrame({
#         "low_var": [1,1,1,1],
#         "high_corr1": [1,2,3,4],
#         "high_corr2": [2,4,6,8],
#         "normal": [1,2,1,2],
#     })
#     processed = preprocess_features(X, low_var_thresh=0.01, corr_thresh=0.9)
#     assert "low_var" not in processed.columns
#     assert "high_corr2" not in processed.columns
#     assert "high_corr1" in processed.columns
#     assert "normal" in processed.columns

# #----------------------------------------
# def test_get_low_variance_features():
#     """
#     Verify get_low_variance_features identifies columns with variance below a threshold.
#     """
#     X = pd.DataFrame({
#         "a":[1,1,1],
#         "b":[1,2,3],
#     })
#     low_var = get_low_variance_features(X, 0.01)
#     assert low_var == ["a"]

# #----------------------------------------
# def test_get_high_correlation_features():
#     """
#     Verify get_high_correlation_features identifies columns
#     that are highly correlated with any other column above a threshold.
#     """
#     X = pd.DataFrame({
#         "x":[1,2,3],
#         "y":[2,4,6],
#         "z":[1,2,1],
#     })
#     high_corr = get_high_correlation_features(X, 0.9)
#     assert "y" in high_corr

# #----------------------------------------
# def test_remap_labels_ordering():
#     """
#     Ensure remap_labels maps cluster labels to a consecutive range starting from 1
#     while preserving the number of samples.
#     """
#     X = pd.DataFrame({
#         "f1":[0,0,1,1],
#         "f2":[0,1,0,1]
#     })
#     labels = np.array([0,0,1,1])
#     remapped = remap_labels(X, labels)
#     assert remapped.min() == 1
#     assert remapped.max() == 2
#     assert remapped.shape[0] == len(labels)
import pytest
import numpy as np
import pandas as pd
from decoupling.utils.preprocessing import (
    set_seed,
    align_and_preprocess_features,
    align_inputs,
    preprocess_features,
    get_low_variance_features,
    get_high_correlation_features,
)
from decoupling.utils.postprocessing import remap_labels

#----------------------------------------
def test_set_seed_consistency():
    """
    Verify that set_seed reproducibly sets the random seed for NumPy
    such that identical calls produce the same random numbers.
    """
    seed = set_seed(42, verbose=False)
    arr1 = np.random.rand(3)
    set_seed(42, verbose=False)
    arr2 = np.random.rand(3)
    np.testing.assert_array_equal(arr1, arr2)

#----------------------------------------
def test_set_seed_none_and_verbose_true(capsys):
    """
    Verify that set_seed generates a random seed when seed=None
    and prints the seed when verbose=True.
    """
    seed = set_seed(None, verbose=True)
    captured = capsys.readouterr()
    assert "Random seed" in captured.out
    assert isinstance(seed, int)
    # Ensure reproducibility
    random_numbers = [np.random.rand() for _ in range(3)]
    set_seed(seed, verbose=False)
    np.testing.assert_array_equal(random_numbers, [np.random.rand() for _ in range(3)])

#----------------------------------------
def test_align_inputs_basic():
    """
    Ensure align_inputs correctly aligns two DataFrames with identical indices,
    returning both DataFrames unchanged and the original index as an array.
    """
    df1 = pd.DataFrame({"a":[1,2,3]}, index=["s1","s2","s3"])
    df2 = pd.DataFrame({"b":[4,5,6]}, index=["s1","s2","s3"])
    M, E, ids = align_inputs(df1, df2)
    pd.testing.assert_frame_equal(M, df1)
    pd.testing.assert_frame_equal(E, df2)
    np.testing.assert_array_equal(ids, np.array(["s1","s2","s3"]))

#----------------------------------------
def test_align_inputs_partial_overlap():
    """
    Verify that align_inputs returns only the overlapping indices
    when the input DataFrames partially overlap.
    """
    df1 = pd.DataFrame({"a":[1,2,3]}, index=["s1","s2","s3"])
    df2 = pd.DataFrame({"b":[4,5,6]}, index=["s2","s3","s4"])
    M, E, ids = align_inputs(df1, df2)
    np.testing.assert_array_equal(ids, np.array(["s2","s3"]))
    assert list(M.index) == ["s2","s3"]
    assert list(E.index) == ["s2","s3"]

#----------------------------------------
def test_align_inputs_no_overlap_raises():
    """
    Confirm that align_inputs raises a ValueError when the DataFrames
    have no overlapping indices.
    """
    df1 = pd.DataFrame({"a":[1]}, index=["s1"])
    df2 = pd.DataFrame({"b":[2]}, index=["s2"])
    with pytest.raises(ValueError):
        align_inputs(df1, df2)

#----------------------------------------
def test_preprocess_features_filters():
    """
    Ensure preprocess_features correctly filters out low-variance
    features and highly correlated features according to thresholds.
    """
    X = pd.DataFrame({
        "low_var": [1,1,1,1],
        "high_corr1": [1,2,3,4],
        "high_corr2": [2,4,6,8],
        "normal": [1,2,1,2],
    })
    processed = preprocess_features(X, low_var_thresh=0.01, corr_thresh=0.9)
    assert "low_var" not in processed.columns
    assert "high_corr2" not in processed.columns
    assert "high_corr1" in processed.columns
    assert "normal" in processed.columns

#----------------------------------------
def test_preprocess_features_skip_filters():
    """
    Confirm preprocess_features returns z-scored data unchanged when thresholds are None.
    """
    X = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })
    processed = preprocess_features(X, low_var_thresh=None, corr_thresh=None)
    # Columns remain
    assert list(processed.columns) == ["a", "b"]
    # Mean approximately 0 after zscore
    assert np.allclose(processed.mean(), 0, atol=1e-7)

#----------------------------------------
def test_align_and_preprocess_features_basic():
    """
    Ensure align_and_preprocess_features properly aligns, z-scores, filters, and weights
    Morph and Ephys data and returns a combined DataFrame.
    """
    Morph = pd.DataFrame({
        "m1": [1, 2, 3, 4],
        "m2": [2, 3, 4, 5]
    }, index=["s1", "s2", "s3", "s4"])

    Ephys = pd.DataFrame({
        "e1": [1, 2, 1, 2],
        "e2": [2, 1, 2, 1]
    }, index=["s1", "s2", "s3", "s4"])

    combined = align_and_preprocess_features(Morph, Ephys, weight=0.5, low_var_thresh=None, corr_thresh=None)
    
    # Check shape
    assert combined.shape[0] == 4
    assert set(["m1", "m2", "e1", "e2"]).issubset(combined.columns)
    # Check that z-scoring works (mean close to 0)
    assert np.allclose(combined.mean(), 0, atol=1e-7)

#----------------------------------------
def test_get_low_variance_features():
    """
    Verify get_low_variance_features identifies columns with variance below a threshold.
    """
    X = pd.DataFrame({
        "a":[1,1,1],
        "b":[1,2,3],
    })
    low_var = get_low_variance_features(X, 0.01)
    assert low_var == ["a"]

#----------------------------------------
def test_get_low_variance_features_edge_case():
    """
    Check that get_low_variance_features returns empty list if all CV values exceed threshold.
    """
    X = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [2, 4, 6]
    })
    low_var = get_low_variance_features(X, threshold=0.1)
    assert low_var == []

#----------------------------------------
def test_get_high_correlation_features():
    """
    Verify get_high_correlation_features identifies columns
    that are highly correlated with any other column above a threshold.
    """
    X = pd.DataFrame({
        "x":[1,2,3],
        "y":[2,4,6],
        "z":[1,2,1],
    })
    high_corr = get_high_correlation_features(X, 0.9)
    assert "y" in high_corr

#----------------------------------------
def test_get_high_correlation_features_partial():
    """
    Ensure get_high_correlation_features identifies only columns exceeding threshold
    and does not drop columns below threshold.
    """
    X = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [1, 2, 3, 4],
        "z": [1, 0, 1, 0]
    })
    high_corr = get_high_correlation_features(X, threshold=0.99)
    assert "y" in high_corr
    assert "x" not in high_corr  # Because we only mark one in a pair

#----------------------------------------
def test_remap_labels_ordering():
    """
    Ensure remap_labels maps cluster labels to a consecutive range starting from 1
    while preserving the number of samples.
    """
    X = pd.DataFrame({
        "f1":[0,0,1,1],
        "f2":[0,1,0,1]
    })
    labels = np.array([0,0,1,1])
    remapped = remap_labels(X, labels)
    assert remapped.min() == 1
    assert remapped.max() == 2
    assert remapped.shape[0] == len(labels)