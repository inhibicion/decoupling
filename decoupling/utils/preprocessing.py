##########################################################################################
### LOAD LIBRARIES
##########################################################################################
import random
import numpy as np
import pandas as pd

from scipy.stats import variation, zscore

##########################################################################################
### PRE-PROCESSING HELPERS
##########################################################################################
def set_seed(seed: int | None = 0, verbose: bool = True) -> int:
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Seed value. If None, a random seed is generated.
    verbose : bool, optional
        Whether to print the seed value (default: True).

    Returns
    -------
    int
        The seed that was set.
    """
    if seed is None:
        # Use int32 max for compatibility across platforms
        seed = np.random.randint(0, np.iinfo(np.int32).max)

    random.seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"Random seed {seed} has been set.")

    return seed

#----------------------------------------------------------------------------------------#
def align_and_preprocess_features(
        Morph: pd.DataFrame, 
        Ephys: pd.DataFrame, 
        weight: float = 0.5, 
        low_var_thresh: float | None = 0.25, 
        corr_thresh: float | None = 0.95
    ) -> pd.DataFrame:
    """
    Align two DataFrames, z-score them independently, drop low-variance and highly 
    correlated features, apply weighting, and concatenate.

    Parameters
    ----------
    Morph : pd.DataFrame
        Morphology feature matrix.
    Ephys : pd.DataFrame
        Electrophysiology feature matrix.
    weight : float, optional
        Scaling factor applied to z-scored `Ephys`, by default 0.5.
    low_var_thresh : float or None, optional
        Threshold for coefficient of variation below which features are removed.
    corr_thresh : float or None, optional
        Correlation threshold above which highly correlated features are removed.

    Returns
    -------
    pd.DataFrame
        Processed, aligned feature matrix containing both `Morph` and weighted `Ephys`.
    """
    M_aligned, E_aligned, _ = align_inputs(Morph=Morph, Ephys=Ephys)

    M_z = preprocess_features(
        Features=M_aligned, 
        low_var_thresh=low_var_thresh, 
        corr_thresh=corr_thresh,
    )
    E_z = preprocess_features(
        Features=E_aligned, 
        low_var_thresh=low_var_thresh, 
        corr_thresh=corr_thresh,
    )

    return M_z.join(weight * E_z)

#----------------------------------------------------------------------------------------#
def align_inputs(
        Morph: pd.DataFrame, 
        Ephys: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Align two DataFrames on their index and ensure they share specimen IDs.

    Parameters
    ----------
    Morph : pd.DataFrame
        Morphological feature matrix.
    Ephys : pd.DataFrame
        Electrophysiological feature matrix.

    Returns
    -------
    pd.DataFrame
        Aligned morphology DataFrame.
    pd.DataFrame
        Aligned electrophysiology DataFrame.
    np.ndarray
        Array of shared specimen IDs.

    Raises
    ------
    ValueError
        If no common indices exist after alignment.
    """
    Morph, Ephys = Morph.dropna(), Ephys.dropna()
    if not Morph.index.equals(Ephys.index):
        Morph, Ephys = Morph.align(Ephys, join="inner", axis=0)

    if len(Morph.index) == 0:
        raise ValueError(
            "No overlapping specimen IDs found after alignment between Morph and Ephys."
        )

    return Morph, Ephys, Morph.index.values

#----------------------------------------------------------------------------------------#
def preprocess_features(
        Features: pd.DataFrame, 
        low_var_thresh: float | None = 0.25, 
        corr_thresh: float | None = 0.95
    ) -> pd.DataFrame:
    """
    Remove low-variance and highly correlated features, then z-score.

    Parameters
    ----------
    Features : pd.DataFrame
        Input feature matrix.
    low_var_thresh : float or None, optional
        Coefficient of variation threshold for dropping features. If None, low-variance 
        filtering is skipped.
    corr_thresh : float or None, optional
        Correlation threshold for dropping features. If None, correlation filtering is 
        skipped.

    Returns
    -------
    pd.DataFrame
        Z-scored filtered feature matrix.
    """
    X = Features.dropna()
    
    features_to_drop = set()

    if low_var_thresh is not None:
        low_var_features = get_low_variance_features(
            X=X, threshold=low_var_thresh,
        )
        features_to_drop.update(low_var_features)

    if corr_thresh is not None:
        high_corr_features = get_high_correlation_features(
            X=X, threshold=corr_thresh,
        )
        features_to_drop.update(high_corr_features)

    if features_to_drop:
        X = X.drop(columns=features_to_drop)

    return X.apply(zscore, nan_policy="omit")

#----------------------------------------------------------------------------------------#
def get_low_variance_features(
        X: pd.DataFrame, 
        threshold: float
    ) -> list[str]:
    """
    Identify features with coefficient of variation below the given threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    threshold : float
        Coefficient of variation threshold.

    Returns
    -------
    list
        Column names with low variance.
    """
    cv = np.abs(variation(X, axis=0, nan_policy="omit"))
    return X.columns[cv < threshold].tolist()

#----------------------------------------------------------------------------------------#
def get_high_correlation_features(
        X: pd.DataFrame, 
        threshold: float
    ) -> list[str]:
    """
    Identify highly correlated features exceeding the given correlation threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    threshold : float
        Correlation threshold above which a feature is removed.

    Returns
    -------
    list
        List of column names to drop.
    """
    corr_matrix = X.corr().abs()
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_corr = corr_matrix.where(upper_triangle)
    return [col for col in upper_corr.columns if any(upper_corr[col] > threshold)]

##########################################################################################