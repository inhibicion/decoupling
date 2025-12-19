##########################################################################################
### LOAD LIBRARIES
##########################################################################################
import numpy as np
import pandas as pd

from scipy.stats import t, sem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Iterable

##########################################################################################
### FEATURE ANALYSIS HELPERS
##########################################################################################
def variance_per_sample_size(
        X: pd.DataFrame,
        sample_sizes: Iterable[int],
        n_components: int = 3,
        n_shuffles: int = 10
    ) -> pd.DataFrame:
    """
    Compute the proportion of explained variance (PEV) for increasing sample sizes with 
    multiple shuffles, and summarize the results by computing the mean and 95% CI width.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix (rows = samples, columns = features).
    sample_sizes : Iterable[int]
        Iterable of sample sizes to evaluate.
    n_components : int, optional
        Number of PCA components to consider (default: 3).
    n_shuffles : int, optional
        Number of random subsamples per sample size (default: 10).

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row contains:
        - "sample_size" : int
        - "mean"        : float, mean variance explained (%)
        - "95% CI"      : float, half-width of the 95% confidence interval.
    """
    X = X.dropna()
    explained_var = {size: [] for size in sample_sizes}

    # Standardize once
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns,
    )

    for size in sample_sizes:
        for seed in range(n_shuffles):
            sample = df_scaled.sample(
                n=size, 
                replace=False, 
                random_state=seed,
            )
            pca = PCA(n_components=n_components).fit(sample)
            cum_var = pca.explained_variance_ratio_.cumsum()[0]
            explained_var[size].append(100 * cum_var)

    pev_dict = []
    for size, vals in explained_var.items():
        vals = np.array(vals)
        mean = vals.mean()
        scale = sem(vals)
        if scale == 0:
            ci_95 = 0
        else:
            ci_95_bounds = t.interval(0.95, len(vals) - 1, loc=mean, scale=scale)
            ci_95 = (ci_95_bounds[1] - ci_95_bounds[0]) / 2
        pev_dict.append(
            {
                "sample_size": size, 
                "mean": mean, 
                "ci_95": ci_95,
            }
        )

    return pd.DataFrame(pev_dict)

##########################################################################################