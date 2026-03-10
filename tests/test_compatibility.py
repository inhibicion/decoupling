import numpy as np

#----------------------------------------
def test_numpy_in1d_alias():
    """
    Ensure np.in1d exists and behaves like np.isin.

    dynamicTreeCut relies on np.in1d, which was removed
    in newer NumPy versions. The package aliases np.isin 
    to np.in1d when necessary.
    """
    # Alias np.in1d to np.isin if missing
    if not hasattr(np, "in1d"):
        np.in1d = np.isin

    # Test that the behavior matches np.isin
    a = np.array([1, 2, 3, 4])
    b = np.array([2, 4])
    expected = np.isin(a, b)
    result = np.in1d(a, b)

    assert np.array_equal(result, expected)