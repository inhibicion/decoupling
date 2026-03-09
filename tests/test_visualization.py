import numpy as np
import pandas as pd
from decoupling.utils.visualization import (
    rotation_matrix,
    linear_reg_coeffs,
    plot_embedding,
    default_colormap,
    plot_embedding_by_layer,
    plot_features_by_layer,
    plot_decoupling_schema,
    colorbar,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

#----------------------------------------
def test_rotation_matrix_properties():
    """
    Ensure the rotation_matrix function returns a valid rotation matrix:
    - determinant equals 1
    - inverse equals the transpose
    """
    theta = np.pi / 4
    R = rotation_matrix(theta)
    det = np.linalg.det(R)
    assert np.isclose(det, 1.0)
    assert np.allclose(np.linalg.inv(R), R.T)

#----------------------------------------
def test_linear_reg_coeffs_basic():
    """
    Linear regression should correctly compute the slope coefficient.
    """
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    coeffs = linear_reg_coeffs(X, y)
    assert np.allclose(coeffs, [2.0])

#----------------------------------------
def test_plot_embedding_output():
    """
    plot_embedding should return a valid figure and axis.
    """
    np.random.seed(0)
    X_emb = np.random.rand(5, 2)
    colors = np.array([0, 1, 0, 1, 0])
    fig, ax = plot_embedding(X_emb, colors)
    assert fig is not None
    assert ax is not None
    plt.close(fig)

#----------------------------------------
def test_default_colormap_type():
    """
    default_colormap should return a Matplotlib ListedColormap.
    """
    cmap = default_colormap()
    from matplotlib.colors import ListedColormap
    assert isinstance(cmap, ListedColormap)

#----------------------------------------
def test_plot_embedding_by_layer_shapes():
    """
    plot_embedding_by_layer should produce a figure with the expected grid
    of subplots for layer-wise features.
    """
    np.random.seed(0)
    embedding = np.random.rand(10, 2)
    feature = np.linspace(0, 1, 10)
    layers = np.array([0, 1, 2, 0, 1, 2, 3, 4, 2, 4])
    fig, axes = plot_embedding_by_layer(embedding, feature, layers)
    assert fig is not None
    assert axes.shape[0] == 6  # 2x3 grid flattened
    plt.close(fig)

#----------------------------------------
def test_plot_features_by_layer_basic():
    """
    plot_features_by_layer should create a figure and axes for each feature group.
    """
    X = pd.DataFrame({
        "layer": [0,0,1,1],
        "feat1": [1,2,1,2],
        "feat2": [2,3,2,3]
    })
    ticks = [[0,1,2,3],[0,1,2,3]]
    fig, axs = plot_features_by_layer(X, ticks)
    assert fig is not None
    assert len(axs) == 2
    plt.close(fig)

#----------------------------------------
def test_plot_decoupling_schema_runs():
    """
    plot_decoupling_schema should produce a figure with one axis per schema group.
    """
    cluster_means = pd.DataFrame({
        "spike_frequency": [-0.5, 0, 0.5],
        "spike_frequency_adaptation": [-0.4, 0, 0.4],
        "dendrite_vertical_extent": [-1,0,1],
        "axon_vertical_extent": [-1,0,1],
    })
    schema = {
        "Group I": {"spike_frequency": (">", 0), "spike_frequency_adaptation": ("<", 0)},
        "Group II": {"axon_vertical_extent": ("<", 0), "dendrite_vertical_extent": (">", 0)}
    }
    fig, axs = plot_decoupling_schema(cluster_means, schema)
    assert fig is not None
    assert len(axs) == len(schema)
    plt.close(fig)

#----------------------------------------
def test_colorbar_creation_continuous_and_categorical():
    """
    colorbar should work for both continuous and categorical color mappings.
    """
    # Continuous
    fig, ax = plt.subplots()
    points = ax.scatter(np.arange(5), np.arange(5), c=np.linspace(0,1,5))
    cbar1 = colorbar(fig, ax)
    assert cbar1 is not None
    plt.close(fig)

    # Categorical
    fig, ax = plt.subplots()
    labels = np.array([0,1,0,1,2])
    points = ax.scatter(np.arange(5), np.arange(5), c=labels)
    cbar2 = colorbar(fig, ax, labels=labels)
    assert cbar2 is not None
    plt.close(fig)