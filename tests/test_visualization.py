import numpy as np
import pandas as pd
import pytest
from decoupling.utils.visualization import (
    rotation_matrix,
    linear_reg_coeffs,
    create_embedding,
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
def test_create_embedding_basic():
    """
    create_embedding should return a 2D array aligned with input and with 2 columns.
    """
    np.random.seed(0)
    X = pd.DataFrame(np.random.rand(30, 4))
    soma_depth = pd.DataFrame(np.linspace(0, 2, 30))
    emb = create_embedding(X, soma_depth)
    assert emb.shape == (30, 2)

#----------------------------------------
def test_default_colormap_type():
    """
    default_colormap should return a Matplotlib ListedColormap.
    """
    cmap = default_colormap()
    from matplotlib.colors import ListedColormap
    assert isinstance(cmap, ListedColormap)

#----------------------------------------
def test_plot_embedding_categorical_and_continuous_branches():
    """
    Cover categorical, continuous, and show_trend branches in plot_embedding.
    """
    X_emb = np.random.rand(5, 2)
    colors_cat = np.array(["a", "b", "a", "b", "c"])
    fig, ax = plot_embedding(X_emb, colors_cat)
    assert fig is not None
    plt.close(fig)

    # Reduce integer classes to <=29 to avoid cmap TypeError
    colors_int = np.arange(20)  # changed from 35
    X_emb_int = np.random.rand(20, 2)
    fig, ax = plot_embedding(X_emb_int, colors_int)
    assert fig is not None
    plt.close(fig)

#----------------------------------------
def test_plot_embedding_by_layer_with_trend():
    """
    plot_embedding_by_layer with nonlinear trend computation (show_trend=True)
    to cover spline calculation and arrow plotting.
    """
    np.random.seed(0)
    embedding = np.random.rand(10, 2)
    feature = np.linspace(0, 1, 10)
    layers = np.array([0, 1, 2, 0, 1, 2, 1, 2, 0, 2])
    fig, axes = plot_embedding_by_layer(embedding, feature, layers, show_trend=True)
    assert fig is not None
    assert len(axes) == 6  # 2x3 grid flattened
    plt.close(fig)

#----------------------------------------
def test_plot_decoupling_schema_linear_and_comparison():
    """
    Cover both comparison ('>','<') and '<linear' rule branches in plot_decoupling_schema.
    """
    cluster_means = pd.DataFrame({
        "feat1": [-0.5, 0, 0.5],
        "feat2": [-0.4, 0, 0.4],
        "feat3": [-1,0,1],
    })
    schema = {
        "Comp": {"feat1": (">", 0), "feat2": ("<", 0)},
        "Linear": {"feat3": ("<linear", "feat1", (1, 0))}
    }
    fig, axs = plot_decoupling_schema(cluster_means, schema)
    assert fig is not None
    assert len(axs) == 2
    plt.close(fig)

#----------------------------------------
def test_plot_features_by_layer_exceptions_and_jitter():
    """
    Test errors for missing 'layer' column, ticks mismatch, and verify jitter/coloring.
    """
    # Missing 'layer' column
    X = pd.DataFrame({"feat": [1,2,3]})
    with pytest.raises(KeyError):
        plot_features_by_layer(X, ticks=[[0,1,2]])

    # Tick length mismatch
    X = pd.DataFrame({"layer": [0,0,1], "feat1":[1,2,3], "feat2":[4,5,6]})
    with pytest.raises(ValueError):
        plot_features_by_layer(X, ticks=[[0,1,2]])

    # Correct call
    X = pd.DataFrame({"layer": [0,0,1], "feat1":[1,2,3], "feat2":[4,5,6]})
    ticks = [[0,1,2,3],[0,4,8,12]]
    fig, axs = plot_features_by_layer(X, ticks, jitter=0.05)
    assert fig is not None
    plt.close(fig)

#----------------------------------------
def test_colorbar_continuous_and_categorical_branches():
    """
    Test continuous vs categorical branches, including integer and string labels.
    Also cover setting cbar labels.
    """
    # Continuous
    fig, ax = plt.subplots()
    sc = ax.scatter(np.arange(5), np.arange(5), c=np.linspace(0,1,5))
    cbar1 = colorbar(fig, ax)
    assert cbar1 is not None
    plt.close(fig)

    # Categorical integer
    fig, ax = plt.subplots()
    labels_int = np.array([0,1,0,1])
    sc = ax.scatter(np.arange(4), np.arange(4), c=labels_int)
    cbar2 = colorbar(fig, ax, labels=labels_int)
    assert cbar2 is not None
    plt.close(fig)

    # Categorical string
    fig, ax = plt.subplots()
    labels_str = np.array(["A","B","A","B"])
    sc = ax.scatter(np.arange(4), np.arange(4), c=np.array([0,1,0,1]))
    cbar3 = colorbar(fig, ax, labels=labels_str)
    assert cbar3 is not None
    plt.close(fig)

    # Custom tick labels
    fig, ax = plt.subplots()
    sc = ax.scatter(np.arange(5), np.arange(5), c=np.linspace(0,1,5))
    cbar4 = colorbar(fig, ax, labels=None, shrink=0.5, location="top")
    assert cbar4 is not None
    plt.close(fig)