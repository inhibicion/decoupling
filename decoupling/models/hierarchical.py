##########################################################################################
### LOAD LIBRARIES
##########################################################################################
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from dynamicTreeCut import cutreeHybrid

from .base import BaseModel
from .ephys_morph_clustering import consensus_clusters
from ..utils.preprocessing import align_inputs, preprocess_features
from ..utils.postprocessing import remap_labels

##########################################################################################
### HELPER FUNCTIONS
##########################################################################################
def hierarchical_clustering(
    X: np.ndarray,
    metric: str = "euclidean",
    method: str = "ward",
    min_cluster_size: int = 1,
    verbose: int = 0,
) -> np.ndarray:
    """
    Perform hierarchical clustering with automatic tree cut.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    metric : str, optional
        Distance metric for pdist.
    method : str, optional
        Linkage method.
    min_cluster_size : int, optional
        Minimum cluster size.
    verbose : int, optional
        Verbosity level for dynamic tree cut.

    Returns
    -------
    np.ndarray
        Cluster labels.
    """
    distance_matrix = pdist(
        X, 
        metric=metric,
    )

    linkage_matrix = linkage(
        distance_matrix, 
        method=method,
    )

    # Ensure compatibility with dynamicTreeCut, which still uses np.in1d.
    # np.in1d was removed in newer NumPy versions, so it is aliased to np.isin.
    if not hasattr(np, "in1d"):
        np.in1d = np.isin

    model = cutreeHybrid(
        link=linkage_matrix,
        distM=distance_matrix,
        minClusterSize=min_cluster_size,
        verbose=verbose,
    )

    return remap_labels(X=X, labels=model["labels"])

##########################################################################################
### MODEL CLASS
##########################################################################################
class HierarchicalClustering(BaseModel):
    """
    Hierarchical clustering model combining morphological and electrophysiological 
    features.
    """
    def __init__(
            self,
            min_clust_size: int = 2,
            weights: list[float] = [0.5, 0.75, 1, 1.5],
            random_state: int = 0,
        ):
        self.min_clust_size = min_clust_size
        self.weights = weights
        self.random_state = random_state
        self.labels_: np.ndarray = None

    #------------------------------------------------------------------------------------#
    def fit(
            self, 
            Morph: pd.DataFrame, 
            Ephys: pd.DataFrame,
            n_depth_bins: int = 2,
            low_var_thresh: float = 0.25, 
            corr_thresh: float = 0.95            
        ) -> "HierarchicalClustering":
        """
        Run hierarchical clustering on `Morph` and `Ephys` features.

        Parameters
        ----------
        Morph : pd.DataFrame
            Morphological feature matrix.
        Ephys : pd.DataFrame
            Electrophysiological feature matrix.
        n_depth_bins : int, optional
            Number of depth bins for stratified clustering.
        low_var_thresh : float, optional
            Threshold for coefficient of variation below which features are removed.
        corr_thresh : float, optional
            Correlation threshold above which highly correlated features are removed.

        Returns
        -------
        self : HierarchicalClustering
            The fitted hierarchical model. Labels are stored in `self.labels_`.
        """
        # 1) Align inputs
        Morph, Ephys, specimen_ids = align_inputs(
            Morph=Morph, 
            Ephys=Ephys,
        )

        # 2) Determine depth bins using soma depth
        possible_keys = ["Soma depth", "soma_depth", "soma depth"]
        for key in possible_keys:
            if key in Morph.columns:
                soma_depth = Morph[key].values
                break
        else:
            raise KeyError(
                f"None of the expected keys {possible_keys} found in DataFrame columns."
            )
        
        max_depth = soma_depth.max()
        bins = np.arange(0, max_depth, np.round(max_depth / n_depth_bins, 1))
        depth_bins = np.digitize(soma_depth, bins)

        # 3) Preprocess features
        Morph = preprocess_features(
            Features=Morph, 
            low_var_thresh=low_var_thresh,
            corr_thresh=corr_thresh,
        )
        Ephys = preprocess_features(
            Features=Ephys, 
            low_var_thresh=low_var_thresh,
            corr_thresh=corr_thresh,
        )

        # 4) Initialize label array
        labels_hc = np.empty(len(specimen_ids), dtype=int)
        offset = 0

        # 5) Cluster per depth bin
        for i in np.unique(depth_bins):

            idx = depth_bins==i

            label_matrix = pd.DataFrame(
                index=specimen_ids[idx],
            )

            for weight in self.weights:
                X = Morph.loc[idx].join(weight * Ephys.loc[idx])
                label_matrix[f"hc_{weight:g}"] = hierarchical_clustering(X)

            clust_labels, _, _ = consensus_clusters(
                results=label_matrix.values, 
                min_clust_size=self.min_clust_size,
            )
            
            labels_hc[idx] = offset + clust_labels 
            offset += len(np.unique(clust_labels))

        self.labels_ = labels_hc

        return self

    #------------------------------------------------------------------------------------#
    def fit_predict(
            self, 
            Morph: pd.DataFrame, 
            Ephys: pd.DataFrame,
            n_depth_bins: int = 2,
            low_var_thresh: float = 0.25, 
            corr_thresh: float = 0.95 
        ) -> np.ndarray:
        """
        Run hierarchical clustering on `Morph` and `Ephys` and get cluster assignments.

        Parameters
        ----------
        Morph : pd.DataFrame
            Morphological feature matrix.
        Ephys : pd.DataFrame
            Electrophysiological feature matrix.
        n_depth_bins : int, optional
            Number of depth bins for stratified clustering.
        low_var_thresh : float, optional
            Threshold for coefficient of variation below which features are removed.
        corr_thresh : float, optional
            Correlation threshold above which highly correlated features are removed.

        Returns
        -------
        np.ndarray
            Hierarchical cluster labels after fitting.
        """
        self.fit(
            Morph=Morph, 
            Ephys=Ephys,
            n_depth_bins=n_depth_bins,
            low_var_thresh=low_var_thresh,
            corr_thresh=corr_thresh,
        )
        return self.labels_

##########################################################################################
    