##########################################################################################
### LOAD LIBRARIES
##########################################################################################
import umap
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import Colorbar
from scipy.interpolate import UnivariateSpline
from scipy.stats import zscore
from functools import reduce
from typing import Any, Sequence

##########################################################################################
### VISUALIZATION HELPERS
##########################################################################################
def create_embedding(
        X: pd.DataFrame,
        soma_depth: pd.DataFrame,
        seed: int = 0
    ) -> np.ndarray:
    """
    Create a 2D embedding from high-dimensional data with layer-wise geometric adjustment.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
    soma_depth : pd.DataFrame
        Soma depth values aligned to `X`.
    seed : int, optional
        Random seed for UMAP and GMM reproducibility.

    Returns
    -------
    np.ndarray
        A 2D embedding of shape (n_samples, 2) after anatomical adjustments.
    
    Notes
    -----
    Steps:
    1. Align `X` and `soma_depth`.
    2. Create a UMAP embedding.
    3. Normalize to unit ball.
    4. Cluster embedding into superficial / granular / infragranular layers
       using a Gaussian Mixture Model.
    5. Apply flips and rotations to align geometry to soma depth.
    6. Shift layers horizontally for visualization.
    """
    # Step 1: Make sure X and soma_depth are aligned
    X, soma_depth = X.align(soma_depth, join="inner", axis=0)
    
    # Step 2: UMAP embedding
    embedding = umap.UMAP(
        n_neighbors=50, 
        min_dist=0.2, 
        n_components=2, 
        metric="cosine", 
        random_state=seed,
    )
    X_embedding = embedding.fit_transform(X)

    # Step 3: Normalize to unit ball
    X_embedding -= X_embedding.mean(axis=0)
    X_embedding /= np.sqrt(np.sum(X_embedding**2, axis=1)).max()

    # Step 4: Gaussian Mixture Model clustering
    GMM = GaussianMixture(n_components=3, random_state=seed)
    labels = GMM.fit_predict(X_embedding)

    # Heuristic assignment of layer identities
    sup_layer = labels == 0
    gran_layer = labels == 1
    inf_layer = labels == 2

    # Step 5: Layer-specific flips
    X_embedding[sup_layer, 1] *= -1
    X_embedding[gran_layer, 0] *= -1

    # Rotate and center each layer
    for layer_mask in [sup_layer, gran_layer, inf_layer]:
        X_tmp = X_embedding[layer_mask]
        if len(X_tmp) >= 2:
            coeffs = linear_reg_coeffs(X_tmp, soma_depth[layer_mask])
            theta = np.arctan2(coeffs[1], coeffs[0])
            X_tmp = X_tmp @ rotation_matrix(-theta)
            X_embedding[layer_mask] = X_tmp - X_tmp.mean(axis=0)

    # Step 6: Shift layers along x-axis
    X_embedding[sup_layer] += np.array([-0.35, 0])
    X_embedding[inf_layer] += np.array([0.35, 0])

    return X_embedding

#----------------------------------------------------------------------------------------#
def linear_reg_coeffs(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit a linear regression model and return coefficient vector.

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix of shape (n_samples, n_features).
    y : np.ndarray
        Response vector of shape (n_samples,).

    Returns
    -------
    np.ndarray
        Fitted regression coefficients.
    """
    lr = LinearRegression().fit(X, y)
    return lr.coef_

#----------------------------------------------------------------------------------------#
def rotation_matrix(theta: float = 0.0) -> np.ndarray:
    """
    Construct a 2x2 rotation matrix.

    Parameters
    ----------
    theta : float, optional
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        A 2x2 rotation matrix.
    """
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, sin], [-sin, cos]])

#----------------------------------------------------------------------------------------#
def plot_embedding(
        X_embedding: np.ndarray,
        color_mapping: np.ndarray | list,
        cmap: str | ListedColormap = None,
        marker_size: int = 150,
        show_trend: bool = True,
        bg_color: np.array = np.array([[0.3, 0.3, 0.3]]),
        figsize: tuple = (6, 6),
        vmin: float = None,
        vmax: float = None
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D embedding with points colored either by categorical labels or a continuous 
    feature.

    Parameters
    ----------
    X_embedding : np.ndarray
        Array of shape (n_samples, 2) representing the embedding coordinates.
    color_mapping : array-like
        Categorical labels (str or integer) or continuous feature values.
    cmap : str or ListedColormap, optional
        Colormap to use for continuous features or categorical labels.
        For continuous, default is `default_colormap()`. For categorical, defaults to 
        concatenated "Set3", "Dark2", "Pastel1".
    marker_size : int, optional
        Size of the points (default: 100).
    show_trend : bool, optional
        If True, draw regression trend arrow for continuous features (default True).
    bg_color : array-like, optional
        Background point color (default dark gray).
    figsize : tuple, optional
        Figure size (default (6, 6)).
    vmin : float, optional
        Minimum value for colormap scaling (for continuous features).
    vmax : float, optional
        Maximum value for colormap scaling (for continuous features).

    Returns
    -------
    plt.Figure
        The figure object containing the embedding plot.
    plt.Axes
        The axes object for further customization.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ensure numpy arrays
    X_embedding = np.asarray(X_embedding)
    color_mapping = np.asarray(color_mapping)

    # Background points
    ax.scatter(*X_embedding.T, c=bg_color, s=marker_size)
   
    # Determine categorical vs continuous
    unique_classes = np.unique(color_mapping)
    n_classes = len(unique_classes)
    is_categorical = False
    if color_mapping.dtype.kind in {"U", "O"}:  # strings or objects
        is_categorical = True
    elif color_mapping.dtype.kind in {"i", "f"}:  # integer or float
        if np.all(np.mod(unique_classes, 1) == 0):
            is_categorical = True

    if is_categorical:
        # Warn if more than 29 classes
        max_default_colors = 29
        if n_classes > max_default_colors:
            print(
                f"Warning: {n_classes} classes detected. "
                f"Default categorical colormap supports {max_default_colors} colors. "
                "Generating distinct colors automatically.",
            )
            cmap = plt.cm.get_cmap("tab20", n_classes).colors
        
        # Use default categorical cmap if none provided
        if cmap is None:
            colors = np.vstack(
                (
                    mpl.colormaps["Set3"].colors,
                    mpl.colormaps["Dark2"].colors,
                    mpl.colormaps["Pastel1"].colors,
                )
            )
            cmap = ListedColormap(colors)

        for i, j in zip(unique_classes, range(n_classes)):
            ax.scatter(
                *X_embedding[np.array(color_mapping)==i].T,
                label=i, 
                color=cmap(j), 
                s=marker_size/2,
            )

    else:
        # Continuous case
        if cmap is None:
            cmap = default_colormap()
        if vmin is None:
            vmin = np.min(color_mapping)
        if vmax is None:
            vmax = np.max(color_mapping)

        ax.scatter(
            *X_embedding.T,
            c=color_mapping,
            s=marker_size/2,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # Regression trend arrow
        if show_trend:
            coeffs = linear_reg_coeffs(X_embedding, color_mapping)
            trend = coeffs / np.linalg.norm(coeffs)

            circ_sz = 0.1
            xmin, ymin = X_embedding.min(axis=0)
            ctr_x, ctr_y = xmin + circ_sz*2, ymin + circ_sz*2

            ax.add_artist(
                plt.Circle((ctr_x, ctr_y), circ_sz, color=bg_color, fill=False),
            )
            ax.arrow(
                ctr_x - 0.98*circ_sz*trend[0],
                ctr_y - 0.98*circ_sz*trend[1],
                1.8*circ_sz*trend[0],
                1.8*circ_sz*trend[1],
                linewidth=1,
                head_width=0.05,
                alpha=1.0,
                overhang=0.25,
                shape="full",
                length_includes_head=True,
                color=bg_color,
                zorder=100,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    return fig, ax

#----------------------------------------------------------------------------------------#
def default_colormap() -> ListedColormap:
    """
    Create a blue-white-red continuous colormap.

    Returns
    -------
    ListedColormap
        A matplotlib colormap that transitions from blue to white to red.
    """
    rr = np.array([239, 138,  98, 255]) / 255
    ww = np.array([254, 254, 254, 255]) / 255
    bb = np.array([103, 169, 207, 255]) / 255

    newcolors = np.vstack(
        (np.linspace(bb, ww, 128), np.linspace(ww, rr, 128))
    )
    return ListedColormap(newcolors, name="blue_white_red")

#----------------------------------------------------------------------------------------#
def plot_embedding_by_layer(
        embedding: np.ndarray,
        feature: np.ndarray,
        layers: np.ndarray,
        show_trend: bool = False,
        res: float = 0.1,
        cbar_labels: list = None,
        figsize: tuple = (8, 4),
    ) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D embedding split by cortical layers, highlighting neurons in each layer and 
    optionally overlaying a nonlinear trend.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_samples, 2)
        2D embedding coordinates.
    feature : np.ndarray, shape (n_samples,)
        Continuous feature used for color-coding (e.g. soma depth).
    layers : np.ndarray, shape (n_samples,)
        Layer labels corresponding to each sample.
    show_trend : bool, optional
        Whether to compute and plot a nonlinear trajectory through the embedding.
    res : float, optional
        Resolution used to bin `feature` values when computing the nonlinear trend.
    cbar_labels : list, optional
        Labels for colorbar.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    plt.Figure
        The figure object containing the embedding plot.
    plt.Axes
        The axes object for further customization.
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    axes = axes.ravel()

    # Styling
    point_size_bg = 20
    point_size_fg = 50
    color_bg = np.array([[0.8, 0.8, 0.8]])
    color_fg = np.array([[0.3, 0.3, 0.3]])

    annotation_kwargs = dict(
        xy=(-0.05, 1.05),
        xycoords="axes fraction",
        fontsize=16,
        ha="center",
        va="top",
    )

    cmap = default_colormap()
    vmin, vmax = feature.min(), feature.max()

    # Axis limits (shared)
    mins = embedding.min(axis=0)
    maxs = embedding.max(axis=0)
    pad = (maxs - mins) * 0.08
    xmin, ymin = mins - pad
    xmax, ymax = maxs + pad
 
    # Optional nonlinear trend computation
    nonlinear_trend = None
    if show_trend:
        digitized = np.digitize(
            feature,
            bins=np.arange(0, vmax + res, res),
        )

        points = (
            pd.DataFrame(embedding)
            .groupby(digitized)
            .mean()
            .values
        )

        distance = np.cumsum(
            np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        )
        distance = np.insert(distance, 0, 0)
        distance /= distance[-1]

        splines = [
            UnivariateSpline(distance, coords, k=3, s=0.2)
            for coords in points.T
        ]

        nonlinear_trend = np.vstack(
            [spl(np.linspace(0, 1, 100)) for spl in splines]
        ).T

    # Prepare layer ordering
    unique_layers = np.unique(layers)
    layer_labels = np.concatenate([[""], unique_layers])

    for ax, layer in zip(axes, layer_labels):

        if layer == "":
            # Show all neurons
            selected = np.ones(len(feature), dtype=bool)
            if show_trend:
                ax.plot(
                    *nonlinear_trend.T,
                    color="black",
                    linewidth=5,
                    zorder=10,
                )
                ax.arrow(
                    *nonlinear_trend[-2],
                    *(nonlinear_trend[-1] - nonlinear_trend[-2]),
                    head_width=0.75,
                    color="black",
                    zorder=20,
                )
        else:
            selected = layers == layer

        # Layer label
        ax.annotate(layer, **annotation_kwargs)

        # Background neurons
        ax.scatter(
            *embedding[~selected].T,
            s=point_size_bg,
            c=color_bg,
            zorder=1,
        )

        # Highlighted neurons
        ax.scatter(
            *embedding[selected].T,
            s=point_size_fg,
            c=color_fg,
            zorder=2,
        )

        # Feature-colored overlay
        f = ax.scatter(
            *embedding[selected].T,
            s=point_size_bg,
            c=feature[selected],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            zorder=3,
        )

        # Axis formatting
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis("off")

    # Add colorbar
    cbar_ax = fig.add_axes([0.3, 0.05, 0.05, 0.5])
    cbar_ax.set_axis_off()
    c = plt.colorbar(
        f, 
        location="bottom", 
        fraction=1, 
        aspect=5, 
        ax=cbar_ax,
    )
    c.set_ticks([vmin, vmax])
    if cbar_labels is not None:
        c.set_ticklabels(cbar_labels)

    return fig, axes

#----------------------------------------------------------------------------------------#
def plot_decoupling_schema(
    cluster_means,
    schema: dict[str, dict[str, Any]],
    figsize: tuple = (8, 2),
    xlim: tuple = (-3, 3),
    ylim: tuple = (-3, 3),
    ticks: tuple = (-2, 0, 2),
    marker: str = "o",
    size: int = 40
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot rule-based decoupling steps defined by a schema.

    Parameters
    ----------
    cluster_means : pd.DataFrame
        Cluster-averaged feature dataframe.
    schema : dict
        Rule schema defining groups and decision boundaries.
    figsize : tuple, optional
        Figure size.
    xlim, ylim : tuple, optional
        Axis limits.
    ticks : tuple, optional
        Tick locations for both axes.
    marker : str, optional
        Scatter marker.
    size : int, optional
        Marker size.

    Returns
    -------
    plt.Figure
        The figure object containing the plots.
    plt.Axes
        The axes object for further customization.
    """
    bg_light = np.array([[0.8, 0.8, 0.8]])
    bg_dark = np.array([[0.3, 0.3, 0.3]])

    fig, axs = plt.subplots(
        nrows=1, 
        ncols=len(schema), 
        figsize=figsize,
    )

    remaining = cluster_means.copy()

    for ax, (group, rules) in zip(axs, schema.items()):

        # Infer plotting axes from rules
        feats = list(rules.keys())
        if rules[feats[0]][0] == "<linear":
            y_feat = feats[0]
            x_feat = rules[feats[0]][1]
        else:
            x_feat, y_feat = feats[1], feats[0]

        # Plot background
        sns.scatterplot(
            data=remaining,
            x=x_feat,
            y=y_feat,
            c=bg_light,
            legend=False,
            marker=marker,
            s=size,
            ax=ax,
        )

        # Evaluate rules and draw boundaries
        rule = rules[y_feat]
        op = rule[0]

        if op in {">", "<"}:
            op_y, thresh_y = rules[y_feat]
            op_x, thresh_x = rules[x_feat]

            op_y = operator.gt if op_y==">" else operator.lt
            op_x = operator.gt if op_x==">" else operator.lt

            logic = rules.get("_logic", "and")
            combine = operator.and_ if logic=="and" else operator.or_

            conditions = [
                op_y(remaining[y_feat], thresh_y),
                op_x(remaining[x_feat], thresh_x),
            ]

            mask = reduce(combine, conditions)
            
            val = 1 if logic=="and" else -1
            ax.plot(
                [thresh_x, xlim[op_x(val, 0)]], 
                [thresh_y, thresh_y], 
                color=bg_dark,
            )
            ax.plot(
                [thresh_x, thresh_x], 
                [thresh_y, ylim[op_y(val, 0)]], 
                color=bg_dark,
            )

        elif op == "<linear":
            _, xref, (slope, intercept) = rule
            mask = remaining[y_feat] < slope * remaining[xref] + intercept

            ax.axline((0, intercept), slope=slope, color=bg_dark)

        else:
            raise ValueError(f"Unknown rule type: {op}")

        # Highlight selected group
        sns.scatterplot(
            data=remaining.loc[mask],
            x=x_feat,
            y=y_feat,
            c=bg_dark,
            legend=False,
            marker=marker,
            s=size,
            ax=ax,
        )
        ax.set_title(group)

        # Update remaining points
        remaining = remaining.loc[~mask]

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    fig.tight_layout()

    return fig, axs

#----------------------------------------------------------------------------------------#
def plot_features_by_layer(
    X: pd.DataFrame,
    ticks: Sequence[Sequence[float]],
    figsize: tuple = (10, 2),
    marker_size: float = 6,
    vmax: float = 2.0,
    jitter: float = 0.125
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot layer-wise feature distributions with boxplots and z-score-colored scatter points.

    Each feature is plotted in its own axis. Values are restricted to the provided
    tick range per feature. Scatter points are jittered horizontally and colored
    by z-score.

    Parameters
    ----------
    X : pd.DataFrame
        Input dataframe containing numeric feature columns and a categorical
        column named `layer`.
        Each row corresponds to one observation.
    ticks : sequence of sequences
        Per-feature y-axis ticks. Length must match the number of plotted features.
        Example:
            [
                [0, 0.4, 0.8, 1.2],
                [0, 0.8, 1.6, 2.4],
                [0,  90, 180, 270],
                [-40, 20, 80, 140],
            ]
    figsize : tuple, optional
        Figure size in inches.
    marker_size : float, optional
        Size of scatter markers.
    vmax : float, optional
        Absolute value for z-score color limits.
    jitter : float, optional
        Horizontal jitter applied to scatter points.

    Returns
    -------
    plt.Figure
        The figure object containing the plots.
    plt.Axes
        The axes object for further customization.
    """
    if "layer" not in X.columns:
        raise KeyError("Input DataFrame must contain a `layer` column")

    features = [c for c in X.columns if c != "layer"]
    if len(features) != len(ticks):
        raise ValueError("Length of ticks must match number of feature columns")

    # Sort for consistent layer ordering
    X = X.copy().sort_values("layer")

    # Z-score numeric features only
    X_z = (
        X[features]
        .apply(zscore)
        .assign(layer=X["layer"])
    )

    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(features),
        figsize=figsize,
        sharex=True,
        sharey=False,
    )

    scatter_kwargs = dict(
        s=marker_size,
        marker="o",
        cmap=default_colormap(),
        vmin=-vmax,
        vmax=vmax,
        legend=False,
        edgecolor=np.array([[0.3, 0.3, 0.3]]),
        linewidth=0.3,
        clip_on=False,
    )

    for ax, feature, yticks in zip(axs, features, ticks):

        ymin, ymax = yticks[0], yticks[-1]
        mask = (X[feature] >= ymin) & (X[feature] <= ymax)

        X_now = X.loc[mask]
        z_now = X_z.loc[mask, feature]

        # Boxplot (distribution)
        sns.boxplot(
            data=X_now,
            x="layer",
            y=feature,
            showfliers=False,
            linewidth=0.5,
            boxprops={"facecolor": "None"},
            width=0.8,
            ax=ax,
        )

        # Scatter (z-score colored)
        sc = sns.scatterplot(
            data=X_now,
            x="layer",
            y=feature,
            c=z_now,
            ax=ax,
            **scatter_kwargs,
        )

        # Horizontal jitter
        points = sc.collections[0]
        n = len(X_now)
        offsets = points.get_offsets()
        offsets[:, 0] += np.random.uniform(-jitter, jitter, n)
        points.set_offsets(offsets)
        points.set_zorder(20)

        ax.set_ylim(ymin, ymax)
        ax.set_yticks(yticks)
        ax.set_xlabel("")
        ax.legend([], [], frameon=False)

    fig.tight_layout()

    return fig, axs

#----------------------------------------------------------------------------------------#
def colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    labels: np.ndarray | pd.Series | list = None,
    shrink: float = 1.0,
    location: str = "bottom",
) -> Colorbar:
    """
    Add a categorical or continuous colorbar to a matplotlib figure.

    The function automatically infers whether the colorbar should be
    categorical (discrete) or continuous based on the provided labels.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to which the colorbar will be added.
    ax : matplotlib.axes.Axes
        Axes associated with the plotted data.
    labels : array-like, optional
        Labels corresponding to the plotted data. If provided and inferred
        to be categorical, a discrete colorbar is created. If None, a
        continuous colorbar is assumed.
    shrink : float, optional
        Fraction by which to shrink the colorbar size (default is 1.0).
    location : str, optional
        Location of the colorbar (e.g., "bottom", "top", "left", "right").

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The created colorbar instance.
    """
     # Determine categorical vs continuous
    is_categorical = False
    if labels is not None:
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        if labels.dtype.kind in {"U", "O"}:  # strings or objects
            is_categorical = True
        elif labels.dtype.kind in {"i", "f"}:  # integer or float
            if np.all(np.mod(unique_labels, 1) == 0):
                is_categorical = True

    if is_categorical:
        # Retrieve colormap
        colors = [
            coll.get_facecolor()[0] for coll in ax.collections[1:1+n_labels]
        ]
        cmap = ListedColormap(colors)

        # Discrete normalization
        norm = BoundaryNorm(
            boundaries=np.arange(n_labels + 1) - 0.5,
            ncolors=n_labels,
        )

        # ScalarMappable required for colorbar
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])

        # Create colorbar
        cbar = fig.colorbar(
            mappable,
            ax=ax,
            ticks=np.arange(n_labels),
            shrink=shrink,
            location=location,
        )

        if np.issubdtype(unique_labels.dtype, np.number):
            cbar.ax.set_xticklabels([int(l) for l in unique_labels])
        else:
            cbar.ax.set_xticklabels(unique_labels)
        cbar.ax.minorticks_off()

    else:
        mappable = ax.collections[-1]
        cbar = fig.colorbar(
            mappable, 
            aspect=10,
            shrink=shrink,
            location=location,
            ax=ax, 
        )

    return cbar

##########################################################################################
