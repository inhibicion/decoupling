import pytest
from decoupling.models.depth_independent import (
    default_schema,
    check_schema_features,
)

#----------------------------------------
def test_default_schema_structure():
    """
    Ensure the default schema has the expected structure:
    a dictionary mapping feature groups to rule dictionaries.
    """
    schema = default_schema()
    assert isinstance(schema, dict)
    assert len(schema) > 0
    for group, rules in schema.items():
        assert isinstance(group, str)
        assert isinstance(rules, dict)
        assert len(rules) > 0

#----------------------------------------
def test_schema_feature_validation_passes():
    """
    Validation should pass when all required features
    defined in the schema are provided.
    """
    schema = default_schema()
    features = [
        "spike_frequency",
        "spike_frequency_adaptation",
        "dendrite_vertical_extent",
        "axon_vertical_extent",
    ]
    # Should not raise an exception
    check_schema_features(schema, features)

#----------------------------------------
def test_schema_feature_validation_fails():
    """
    Validation should fail if required features
    defined in the schema are missing.
    """
    schema = default_schema()
    incomplete_features = ["spike_frequency"]
    with pytest.raises(KeyError):
        check_schema_features(schema, features=incomplete_features)