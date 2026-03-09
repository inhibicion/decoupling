import numpy as np
import pandas as pd
from decoupling.models.depth_independent import Decoupling

#----------------------------------------
def test_decoupling_fit_predict_basic():
    """
    Test that Decoupling.fit_predict runs on basic synthetic data
    and assigns labels consistent with the schema.
    """
    # Synthetic features: 6 samples, 4 features
    features = pd.DataFrame({
        "spike_frequency": [0.2, 0.3, -0.6, -0.7, 0.0, 0.1],
        "spike_frequency_adaptation": [-0.4, -0.3, -0.6, -0.8, 0.0, 0.1],
        "dendrite_vertical_extent": [0.5, 0.6, -1.2, -0.9, 0.0, 0.1],
        "axon_vertical_extent": [0.2, 0.3, 0.0, 0.1, 0.5, 0.6],
    })
    # Local grouping (clusters)
    local_grouping = np.array([0, 0, 1, 1, 2, 2])
    # Fit the model and predict labels
    model = Decoupling()
    labels = model.fit_predict(features, local_grouping)
    # Output length should match input
    assert labels.shape[0] == features.shape[0]
    # All assigned labels must exist in the schema
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
    # Number of predicted clusters should not exceed the number of local groups
    assert len(np.unique(labels)) <= len(local_grouping)