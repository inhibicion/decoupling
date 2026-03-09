import numpy as np

#----------------------------------------
def test_numpy_in1d_available():
    """
    Check that NumPy has the `in1d` function available.

    dynamicTreeCut relies on np.in1d, which was removed
    in newer NumPy versions. The project aliases np.isin
    to np.in1d when necessary.
    """
    assert hasattr(np, "in1d")

#----------------------------------------
def test_numpy_isin_behavior_matches_in1d():
    """
    Verify that `np.in1d` behaves the same as `np.isin`.

    Ensures compatibility across NumPy versions by comparing
    outputs for the same arrays.
    """
    a = np.array([1, 2, 3, 4])
    b = np.array([2, 4])
    expected = np.isin(a, b)
    result = np.in1d(a, b)
    assert np.array_equal(expected, result)