import pytest
import numpy as np
import pandas as pd
from decoupling.models.depth_independent import Decoupling, evaluate_rule, check_schema_features

#----------------------------------------
def test_decoupling_fit_predict_basic():
    """
    Test that Decoupling.fit_predict runs on basic synthetic data
    and assigns labels consistent with the schema.
    """
    features = pd.DataFrame({
        "spike_frequency": [0.2, 0.3, -0.6, -0.7, 0.0, 0.1],
        "spike_frequency_adaptation": [-0.4, -0.3, -0.6, -0.8, 0.0, 0.1],
        "dendrite_vertical_extent": [0.5, 0.6, -1.2, -0.9, 0.0, 0.1],
        "axon_vertical_extent": [0.2, 0.3, 0.0, 0.1, 0.5, 0.6],
    })
    local_grouping = np.array([0, 0, 1, 1, 2, 2])
    model = Decoupling()
    labels = model.fit_predict(features, local_grouping)
    assert labels.shape[0] == features.shape[0]
    valid_groups = list(model.schema.keys())
    assert all(l in valid_groups for l in labels)

#----------------------------------------
def test_decoupling_multiple_clusters():
    """
    Ensure that Decoupling can handle multiple clusters and does not assign
    more clusters than the number of provided local groups.
    """
    features = pd.DataFrame({
        "spike_frequency": [0.2, -0.6, 0.0],
        "spike_frequency_adaptation": [-0.4, -0.6, 0.0],
        "dendrite_vertical_extent": [0.5, -1.2, 0.1],
        "axon_vertical_extent": [0.2, 0.0, 0.5],
    })
    local_grouping = np.array([0, 1, 2])
    model = Decoupling()
    labels = model.fit_predict(features, local_grouping)
    assert len(np.unique(labels)) <= len(local_grouping)

#----------------------------------------
def test_check_schema_features_missing_raises():
    """
    Verify that check_schema_features raises KeyError when schema
    references features not in the data.
    """
    schema = {"Group X": {"feat_missing": (">", 0)}}
    features = ["feat_present"]
    with pytest.raises(KeyError):
        check_schema_features(schema, features)

#----------------------------------------
def test_evaluate_rule_exceptions():
    """
    Test evaluate_rule raises ValueError for invalid input types or unknown operators.
    """
    row = pd.Series({"f1": 1, "f2": 2})
    # Not a tuple
    with pytest.raises(ValueError):
        evaluate_rule(row, "f1", 123)
    # Unknown operator
    with pytest.raises(ValueError):
        evaluate_rule(row, "f1", ("???", 1))

#----------------------------------------
def test_evaluate_rule_linear():
    """
    Test evaluate_rule with '<linear' operator returns correct boolean.
    """
    row = pd.Series({"f1": 1.0, "f2": 2.0})
    # f1 < 0.5 * f2 + 0.2 -> 1.0 < 0.5*2 + 0.2 = 1.2 -> True
    assert evaluate_rule(row, "f1", ("<linear", "f2", (0.5, 0.2))) == True
    # f1 < 0.5 * f2 - 0.5 -> 1.0 < 0.5*2 - 0.5 = 0.5 -> False
    assert evaluate_rule(row, "f1", ("<linear", "f2", (0.5, -0.5))) == False

#----------------------------------------
def test_decoupling_with_meta_logic_key():
    """
    Verify that Decoupling correctly ignores schema keys starting with '_'.
    """
    features = pd.DataFrame({
        "spike_frequency": [0.1, 0.2],
        "spike_frequency_adaptation": [-0.2, -0.3],
        "dendrite_vertical_extent": [0.5, 0.6],
        "axon_vertical_extent": [0.1, 0.2],
    })
    local_grouping = np.array([0, 0])
    schema = {
        "Group A": {
            "_logic": "and",
            "spike_frequency": (">", 0)
        },
        "Group B": {
            "spike_frequency": (">", 1)
        }
    }
    model = Decoupling(schema=schema)
    labels = model.fit_predict(features, local_grouping)
    # Ensure no exceptions and labels exist
    assert len(labels) == len(features)

#----------------------------------------
def test_decoupling_default_label_assignment():
    """
    Check that clusters failing all rules get assigned the penultimate schema label as default.
    """
    features = pd.DataFrame({
        "f1": [0, 0],
        "f2": [0, 0],
    })
    local_grouping = np.array([0, 1])
    schema = {
        "Group X": {"f1": (">", 1)},  # impossible
        "Group Y": {"f2": (">", 1)},  # impossible
        "Group Z": {"f1": (">", 1)},  # penultimate key for default
        "Group Default": {"f2": (">", 1)},  # last key
    }
    model = Decoupling(schema=schema)
    labels = model.fit_predict(features, local_grouping)
    penultimate_label = list(schema.keys())[-2]  # "Group Z"
    assert all(l == penultimate_label for l in labels)