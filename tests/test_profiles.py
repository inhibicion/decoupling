import numpy as np
from decoupling.utils.postprocessing import compute_molecular_profiles

#----------------------------------------
def generate_dummy_data(n=50, seed=0):
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
    markers, soma_depth = generate_dummy_data()
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
    markers, soma_depth = generate_dummy_data()
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
    markers, soma_depth = generate_dummy_data()
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
    markers, soma_depth = generate_dummy_data()
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