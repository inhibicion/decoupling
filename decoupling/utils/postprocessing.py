##########################################################################################
# LOAD LIBRARIES
##########################################################################################
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.ndimage import uniform_filter1d
from typing import Iterable

##########################################################################################
# POST-PROCESSING HELPERS
##########################################################################################
def remap_labels(
        X: pd.DataFrame,
        labels: Iterable[int | np.integer]
    ) -> np.ndarray:
    """
    Reorder cluster labels using PCA and hierarchical clustering to produce a more 
    interpretable, spatially consistent ordering.

    This improves visual consistency by assigning cluster IDs based on dendrogram 
    ordering of PCA-mean cluster centroids.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix of shape (n_samples, n_features).
    labels : iterable of int
        Original cluster labels for each sample.

    Returns
    -------
    np.ndarray
        Array of remapped cluster labels (starting at 1).

    Notes
    -----
    Steps:
    1. Compute 2D PCA projection of X.
    2. Compute mean PCA coordinates for each cluster.
    3. Use hierarchical clustering (Ward) to infer an ordering.
    4. Map original labels to the dendrogram-based ordering.
    """
    labels_arr = np.asarray(labels)
    X_pca = pd.DataFrame(PCA(n_components=2).fit_transform(X))

    # Mean PCA coordinates per cluster
    avg = X_pca.groupby(labels_arr).mean()

    # Dendrogram ordering
    Z = linkage(pdist(avg), method="ward", optimal_ordering=True)
    order = list(leaves_list(Z)[::-1])

    # Shift labels to start at zero before remapping
    shifted = labels_arr - labels_arr.min()

    return np.array([1 + order.index(x) for x in shifted])

#----------------------------------------------------------------------------------------#
def compute_molecular_profiles(
    markers: np.ndarray,
    soma_depth: np.ndarray,
    depth_bin_width: float = 0.05,
    smooth_window: int = 4,
    max_depth: float = 2.0,
) -> pd.DataFrame:
    """
    Compute smoothed depth profiles for molecular markers.

    Parameters
    ----------
    markers : np.ndarray
        Array of molecular marker assignments for each neuron.
    soma_depth : np.ndarray
        Array of corresponding soma depth for each neuron.
    depth_bin_width : float, optional
        Width of depth bins in the same units as `soma_depth`.
    smooth_window : int, optional
        Window size for moving average smoothing (number of bins).
    max_depth : float, optional
        Maximum depth for binning. If None, uses the max of `soma_depth`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["depth", "marker", "counts", "fraction"] representing
        binned counts and smoothed fractions per marker.
    """
    if len(markers) != len(soma_depth):
        raise ValueError("Length of markers and soma_depths must be equal.")
    
    if max_depth is None:
        max_depth = np.nanmax(soma_depth)
    
    # Combine depths and markers into a DataFrame
    df = pd.DataFrame({"depth": soma_depth, "marker": markers}).dropna()
    
    # Create depth bins
    depth_bins = np.arange(0, max_depth - depth_bin_width, depth_bin_width)
    
    marker_profiles = []
    for marker in np.unique(df["marker"]):
        depths = df.loc[df["marker"] == marker, "depth"]
        counts, edges = np.histogram(depths, bins=depth_bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        
        marker_profiles.append(
            pd.DataFrame({
                "depth": bin_centers,
                "counts": counts,
                "marker": marker,
            })
        )
    
    profiles = pd.concat(marker_profiles, ignore_index=True)
    
    # Normalize counts to fractions per depth bin (%)
    profiles["counts"] = (
        100 * profiles["counts"]
        / profiles.groupby("depth")["counts"].transform("sum")
    )
    
    # Apply moving average smoothing
    profiles = profiles.assign(
        fraction=lambda x: uniform_filter1d(
            x["counts"].fillna(0).to_numpy(), 
            size=smooth_window, 
            mode="nearest",
        )
    )
    
    return profiles

##########################################################################################