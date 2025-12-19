##########################################################################################
### LOAD LIBRARIES
##########################################################################################
import random
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.base import clone
from typing import Iterable

##########################################################################################
### CLUSTER ANALYSIS HELPERS
##########################################################################################
def count_per_cluster(labels: Iterable[int | np.integer]) -> pd.DataFrame:
    """
    Count the number of observations assigned to each cluster.

    Parameters
    ----------
    labels : iterable of int
        Cluster label for each observation.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame where columns are cluster IDs and values are counts.
    """
    tmp = pd.DataFrame(
        {
            "Observation": range(len(labels)),
            "Cluster": labels,
        }
    )

    counts = tmp.groupby("Cluster")["Observation"].nunique()

    return counts.to_frame().T

#----------------------------------------------------------------------------------------#
def cluster_predictability(
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series | list,
        k: int = 5,
        classifiers: list[object] = None,
        scoring: str = "accuracy",
        shuffle: bool = True,
        random_state: int = 0
    ) -> pd.DataFrame:
    """
    Assess how well different classifiers predict cluster labels, optionally comparing to 
    shuffled labels as baseline.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : array-like
        True cluster labels.
    k : int
        Number of cross-validation folds.
    classifiers : list of objects, optional
        List of sklearn classifiers. If None, defaults are used.
    scoring : str, optional
        Scoring metric for evaluation (e.g., "accuracy", "f1").
    shuffle : bool, optional
        Whether to include a shuffled baseline for comparison.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Melted DataFrame with columns ["classifier", scoring, "class_labels"] showing CV 
        scores.
    """
    if classifiers is None:
        classifiers = get_default_classifiers()

    # Prepare label sets
    label_sets = [np.array(y)]
    label_types = ["original"] * k

    if shuffle:
        shuffled_y = np.array(y).copy()
        random.Random(random_state).shuffle(shuffled_y)
        label_sets.append(shuffled_y)
        label_types += ["shuffled"] * k

    results = pd.DataFrame(index=range(len(label_sets) * k))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class.*")
        for clf in classifiers:
            clf_copy = clone(clf)
            if hasattr(clf_copy, "random_state"):
                setattr(clf_copy, "random_state", random_state)

            scores = []
            for labels in label_sets:
                fold_scores = cross_val_score(clf_copy, X, labels, cv=k, scoring=scoring)
                scores.extend(fold_scores)

            clf_name = type(clf).__name__
            # Avoid duplicate column names
            while clf_name in results.columns:
                clf_name += "_dup"

            results[clf_name] = scores

    # Order classifiers by median score of original labels
    ordered_cols = results.columns[np.argsort(-results.iloc[:k].median())]

    # Melt for tidy format
    melted = pd.melt(results[ordered_cols], var_name="classifier", value_name=scoring)
    melted["class_labels"] = label_types * len(classifiers)

    return melted

#----------------------------------------------------------------------------------------#
def get_default_classifiers() -> list[object]:
    """
    Return a list of scikit-learn classifier instances with sensible default
    hyperparameters.

    The returned models span a range of learning paradigms (tree-based, kernel-based, 
    probabilistic, instance-based, and neural networks) and are intended for quick 
    benchmarking or baseline comparisons rather than exhaustive hyperparameter tuning.

    Returns
    -------
    list
        A list of instantiated sklearn classifier objects.
    """
    return [
        RandomForestClassifier(n_estimators=500),
        MLPClassifier(alpha=1, max_iter=2000),
        GaussianProcessClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        GaussianNB(),
        SVC(),
    ]

##########################################################################################