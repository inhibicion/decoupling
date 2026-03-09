##########################################################################################
### LOAD LIBRARIES
##########################################################################################
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from decoupling.models.base import BaseModel
from decoupling.models.ephys_morph_clustering import all_cluster_calls, consensus_clusters
from decoupling.utils.preprocessing import align_inputs, preprocess_features
from decoupling.utils.postprocessing import remap_labels
from decoupling.metrics.clusters import count_per_cluster

##########################################################################################
### HELPER FUNCTIONS
##########################################################################################
def merge_unstable_clusters(
        clust_labels: np.ndarray, 
        cc_rates: np.ndarray, 
        min_size: int
    ) -> np.ndarray:
    """
    Merge clusters smaller than `min_size` by assigning them to the cluster with the 
    highest consensus rate neighbor. Re-indexes clusters to consecutive integers.

    Parameters
    ----------
    clust_labels : np.ndarray
        Initial cluster labels.
    cc_rates : np.ndarray
        Co-clustering rates from consensus clustering.
    min_size : int
        Minimum cluster size threshold.

    Returns
    -------
    np.ndarray
        Updated cluster labels.
    """
    cluster_sizes = count_per_cluster(clust_labels)
    small_clusters = np.flatnonzero(cluster_sizes < min_size)

    if len(small_clusters) == 0:
        return clust_labels  # Nothing to merge

    # Remove diagonal self-consensus
    cc_tmp = cc_rates - np.diag(np.diag(cc_rates))
    cc_tmp = cc_tmp[:, small_clusters]

    # Pick cluster with highest consensus for each small cluster
    merge_targets = cc_tmp.argmax(axis=0)

    # Handle conflicts by redirecting to a previously used merge
    used_targets = []
    for i, target in enumerate(merge_targets):
        if target in used_targets:
            # Redirect merge to whichever cluster merged into same target
            conflict_idx = np.where(merge_targets == target)[0]
            merge_targets[i] = merge_targets[conflict_idx][0]
        used_targets.append(target)

    # Apply merges
    for small_clust, target_clust in zip(small_clusters, merge_targets):
        clust_labels[clust_labels == small_clust] = target_clust

    # Reindex labels to 0..K
    unique_labs = np.unique(clust_labels)
    for new_idx, old_val in enumerate(unique_labs):
        clust_labels[clust_labels == old_val] = new_idx

    return clust_labels

#----------------------------------------------------------------------------------------#
def optimal_weight(
        Morph: np.ndarray, 
        Ephys: np.ndarray, 
        clust_labels: np.ndarray, 
        weights: list[float], 
        random_state: int
    ) -> tuple[float, np.ndarray]:
    """
    Select the best weight for combining `Morph` and `Ephys` features using RandomForest 
    out-of-bag (OOB) score.

    Parameters
    ----------
    Morph : np.ndarray
        Morphological feature matrix of shape (n_samples, n_features_morph).
    Ephys : np.ndarray
        Electrophysiological feature matrix of shape (n_samples, n_features_ephys).
    clust_labels : np.ndarray
        Cluster labels for each sample of shape (n_samples,).
    weights : list of float
        List of scaling weights to apply to `Ephys` features.
    random_state : int
        Random seed for reproducibility of RandomForestClassifier.

    Returns
    -------
    float
        The weight from `weights` that yields the highest out-of-bag (OOB) accuracy.
    np.ndarray
        Combined feature matrix (`Morph` + weighted `Ephys`).
    """
    best_acc = -np.inf
    best_weight = None
    best_X = None

    for w in weights:
        X_now = np.hstack([Morph.values, w * Ephys.values])

        rf = RandomForestClassifier(
            n_estimators=500,
            oob_score=True,
            criterion="gini",
            random_state=random_state,
        )

        rf.fit(X_now, clust_labels)

        if rf.oob_score_ > best_acc:
            best_acc = rf.oob_score_
            best_weight = w
            best_X = X_now

    return best_weight, best_X

##########################################################################################
### MODEL CLASS
##########################################################################################
class ConsensusClustering(BaseModel):
    """
    Consensus clustering model combining morphological and electrophysiological features.
    """
    def __init__(
            self,
            min_clust_size: int = 5,
            weights: list[float] = [0.5, 0.75, 1, 1.5],
            n_cl: list[int] = [10, 15, 20, 25],
            n_nn: list[int] = [4, 7, 10],
            random_state: int = 0,
        ):
        self.clust_size_lower_bound = min_clust_size - 1
        self.clust_size_upper_bound = min_clust_size + 1
        self.weights = weights
        self.n_cl = n_cl
        self.n_nn = n_nn
        self.random_state = random_state
        self.labels_: np.ndarray = None
        self.optimal_weight_: float = None

    #------------------------------------------------------------------------------------#
    def fit(
            self, 
            Morph: pd.DataFrame, 
            Ephys: pd.DataFrame,
            low_var_thresh: float = 0.25, 
            corr_thresh: float = 0.95            
        ) -> "ConsensusClustering":
        """
        Run consensus clustering on `Morph` and `Ephys` features.

        Parameters
        ----------
        Morph : pd.DataFrame
            Morphological feature matrix.
        Ephys : pd.DataFrame
            Electrophysiological feature matrix.
        low_var_thresh : float, optional
            Threshold for coefficient of variation below which features are removed.
        corr_thresh : float, optional
            Correlation threshold above which highly correlated features are removed.

        Returns
        -------
        self : ConsensusClustering
            The fitted clustering model. Labels are stored in `self.labels_`.
        """
        # 1) Align and process inputs
        Morph, Ephys, specimen_ids = align_inputs(
            Morph=Morph, 
            Ephys=Ephys,
        )
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

        # 2) Run all cluster calls
        label_matrix = all_cluster_calls(
            specimen_ids=specimen_ids,
            morph_X=Morph,
            ephys_spca=Ephys,
            weights=self.weights,
            n_cl=self.n_cl,
            n_nn=self.n_nn,
        )

        # 3) Initial class assignment
        clust_labels, _, cc_rates = consensus_clusters(
            results=label_matrix.values[:, 1:],
            min_clust_size=self.clust_size_lower_bound,
        )

        # 4) Merge unstable clusters
        clust_labels = merge_unstable_clusters(
            clust_labels=clust_labels, 
            cc_rates=cc_rates, 
            min_size=self.clust_size_upper_bound,
        )

        # 5) Select optimal weighting between morph and ephys
        self.optimal_weight_, X = optimal_weight(
            Morph=Morph, 
            Ephys=Ephys, 
            clust_labels=clust_labels, 
            weights=self.weights, 
            random_state=self.random_state,
        )

        # 6) Remap cluster IDs by dendrogram order
        self.labels_ = remap_labels(X=X, labels=clust_labels)

        return self

    #------------------------------------------------------------------------------------#
    def fit_predict(
            self, 
            Morph: pd.DataFrame, 
            Ephys: pd.DataFrame,
            low_var_thresh: float = 0.25, 
            corr_thresh: float = 0.95 
        ) -> np.ndarray:
        """
        Run consensus clustering on `Morph` and `Ephys` and get cluster assignments.

        Parameters
        ----------
        Morph : pd.DataFrame
            Morphological feature matrix.
        Ephys : pd.DataFrame
            Electrophysiological feature matrix.
        low_var_thresh : float, optional
            Threshold for coefficient of variation below which features are removed.
        corr_thresh : float, optional
            Correlation threshold above which highly correlated features are removed.

        Returns
        -------
        np.ndarray
            Consensus cluster labels after fitting.
        """
        self.fit(
            Morph=Morph, 
            Ephys=Ephys,
            low_var_thresh=low_var_thresh,
            corr_thresh=corr_thresh,
        )
        return self.labels_

##########################################################################################
    