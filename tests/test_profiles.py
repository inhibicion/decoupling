import pytest
import numpy as np
from decoupling.utils.postprocessing import compute_molecular_profiles

#----------------------------------------
def toy_data(n=50, seed=0):
    """
    Helper function to generate synthetic marker and soma depth data
    for testing molecular profile computations.
    """
    rng = np.random.default_rng(seed)
    soma_depth = rng.uniform(0, 2, n)
    markers = rng.choice(
        ["PV", "Sst", "Vip", "-/-/-"],
        size=n
    )
    return markers, soma_depth

#----------------------------------------
def test_compute_profiles_structure():
    """
    Ensure the output of compute_molecular_profiles contains
    all required columns: depth, marker, counts, fraction.
    """
    markers, soma_depth = toy_data()
    profiles = compute_molecular_profiles(
        markers=markers,
        soma_depth=soma_depth
    )
    required_columns = {"depth", "marker", "counts", "fraction"}
    assert required_columns.issubset(profiles.columns)

#----------------------------------------
def test_profiles_depth_not_lost():
    """
    Verify that depth information is preserved in the computed profiles
    and no NaN values are present.
    """
    markers, soma_depth = toy_data()
    profiles = compute_molecular_profiles(
        markers=markers,
        soma_depth=soma_depth
    )
    assert "depth" in profiles.columns
    assert profiles["depth"].notna().all()

#----------------------------------------
def test_profiles_fraction_range():
    """
    Ensure that the fraction column in profiles is between 0 and 100.
    """
    markers, soma_depth = toy_data()
    profiles = compute_molecular_profiles(
        markers=markers,
        soma_depth=soma_depth
    )
    assert profiles["fraction"].min() >= 0
    assert profiles["fraction"].max() <= 100

#----------------------------------------
def test_profiles_fraction_normalization():
    """
    Check that the counts per depth bin sum to 100 after normalization.
    """
    markers, soma_depth = toy_data()
    profiles = compute_molecular_profiles(
        markers=markers,
        soma_depth=soma_depth
    )
    sums = profiles.groupby("depth")["counts"].sum()
    sums = sums[sums > 0]
    assert np.allclose(sums.values, 100, atol=1e-6)

#----------------------------------------
def test_profiles_handles_empty_marker_group():
    """
    Ensure the function works even if some marker groups are not present
    in the input data.
    """
    soma_depth = np.array([0.1, 0.2, 0.3])
    markers = np.array(["PV", "PV", "PV"])
    profiles = compute_molecular_profiles(
        markers=markers,
        soma_depth=soma_depth
    )
    assert len(profiles) > 0

#----------------------------------------
def test_markers_soma_length_mismatch():
    """
    Confirm that a ValueError is raised if markers and soma_depth lengths differ.
    """
    markers = np.array(["PV", "Sst"])
    soma_depth = np.array([0.1])
    with pytest.raises(ValueError, match="Length of markers and soma_depths must be equal"):
        compute_molecular_profiles(markers=markers, soma_depth=soma_depth)

#----------------------------------------
def test_max_depth_none_uses_actual_max():
    """
    Verify that setting max_depth=None uses the maximum value from soma_depth.
    """
    markers, soma_depth = toy_data(n=10)
    profiles = compute_molecular_profiles(markers=markers, soma_depth=soma_depth, max_depth=None)
    assert profiles["depth"].max() <= np.nanmax(soma_depth)

#----------------------------------------
def test_smooth_window_effect():
    """
    Ensure that changing smooth_window parameter alters the fraction column.
    """
    markers, soma_depth = toy_data(n=20)
    profiles_small = compute_molecular_profiles(markers, soma_depth, smooth_window=1)
    profiles_large = compute_molecular_profiles(markers, soma_depth, smooth_window=5)
    assert not np.allclose(profiles_small["fraction"], profiles_large["fraction"])

#----------------------------------------
def test_depth_bin_width_effect():
    """
    Ensure that modifying depth_bin_width changes the number of depth bins.
    """
    markers, soma_depth = toy_data(n=20)
    profiles_default = compute_molecular_profiles(markers, soma_depth)
    profiles_wide = compute_molecular_profiles(markers, soma_depth, depth_bin_width=0.2)
    assert len(profiles_default["depth"].unique()) != len(profiles_wide["depth"].unique())