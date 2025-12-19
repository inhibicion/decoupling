import pytest
from decoupling.models.depth_independent import default_schema, check_schema_features


def test_default_schema_structure():
    schema = default_schema()
    assert isinstance(schema, dict)
    for group, rules in schema.items():
        assert isinstance(group, str)
        assert isinstance(rules, dict)


def test_schema_feature_validation_passes():
    schema = default_schema()
    features = [
        "spike_frequency",
        "spike_frequency_adaptation",
        "dendrite_vertical_extent",
        "axon_vertical_extent",
    ]
    check_schema_features(schema, features)


def test_schema_feature_validation_fails():
    schema = default_schema()
    with pytest.raises(KeyError):
        check_schema_features(schema, features=["spike_frequency"])
