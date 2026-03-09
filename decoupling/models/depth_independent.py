##########################################################################################
### LOAD LIBRARIES
##########################################################################################
import operator
import numpy as np
import pandas as pd

from functools import reduce
from typing import Any, Iterable

from decoupling.models.base import BaseModel

##########################################################################################
### HELPER FUNCTIONS
##########################################################################################
def default_schema() -> dict[str, dict[str, Any]]:
    """
    Default rule-based schema for class assignments.

    Returns
    -------
    dict
        A dictionary mapping groups to rules applied on features.
    """
    return {
        "Group I": {
            "spike_frequency": (">", 0.15),
            "spike_frequency_adaptation": ("<", -0.35)
        },
        "Group IIIb": {
            "spike_frequency": ("<", -0.5),
            "dendrite_vertical_extent": ("<", -1)
        },
        "Group IIIa": {
            "axon_vertical_extent": (
                "<linear", 
                "dendrite_vertical_extent", 
                (0.15, 0.35),
            )
        },
        "Group II": {
            "spike_frequency": (
                "<linear", 
                "spike_frequency_adaptation", 
                (3, 0.05),
            )
        }
    }

#----------------------------------------------------------------------------------------#
def check_schema_features(schema: dict, features: Iterable[str]) -> None:
    """
    Ensure that all features referenced in the schema exist in the provided features list.

    Parameters
    ----------
    schema : dict
        Rule-based schema.
    features : iterable
        Iterable of feature names.

    Raises
    ------
    KeyError
        If any schema feature is missing in the provided features.
    """
    missing = set()
    for rules in schema.values():
        for feat in rules:
            # Skip meta-keys like _logic
            if feat.startswith("_"):  
                continue
            if feat not in features:
                missing.add(feat)
    if missing:
        raise KeyError(f"Schema references missing features: {sorted(missing)}")
    
#----------------------------------------------------------------------------------------#
def evaluate_rule(row: pd.Series, feature: str, rule: Any) -> bool:
    """
    Evaluate a single rule against a row of features.

    Parameters
    ----------
    row : pd.Series
        A row of cluster-averaged features.
    feature : str
        Feature name to evaluate.
    rule : tuple
        Rule definition.

    Returns
    -------
    bool
        True if the rule is satisfied, False otherwise.

    Raises
    ------
    ValueError
        If the rule operator is unknown.
    """
    if not isinstance(rule, tuple):
        raise ValueError(f"Rule must be a tuple, got {rule}")

    op = rule[0]

    if op in {">", "<"}:
        _, threshold = rule
        value = row[feature]
        return value > threshold if op == ">" else value < threshold

    if op == "<linear":
        _, ref_feature, (slope, intercept) = rule
        return row[feature] < slope * row[ref_feature] + intercept

    raise ValueError(f"Unknown rule operator: {op}")

##########################################################################################
### MODEL CLASS
##########################################################################################
class Decoupling(BaseModel):
    """
    Assign depth-independent classes to neurons using rule-based thresholds applied to 
    cluster-averaged features.

    The assignment is done at the cluster level and mapped back to individual samples.
    """
    def __init__(self, schema: dict[str, Any] = None):
        """
        Parameters
        ----------
        schema : dict, optional
            Rule-based schema for class assignment. Should have the form:
            {
                "Group 1": {"feature A": (">", value A), "feature B": ("<", value B)},
                ...
                "Group N": {"feature A": ("<linear", "feature B", (slope, intercept))},
            }
            If None, default thresholds are used.
        """
        self.schema = default_schema() if schema is None else schema
        self.labels_: np.ndarray = None    

    #------------------------------------------------------------------------------------#
    def fit(self, features: pd.DataFrame, local_grouping: np.ndarray) -> "Decoupling":
        """
        Run decoupling model on cluster-averaged features.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix with columns like 
                "dendrite_vertical_extent", 
                "axon_vertical_extent", 
                "spike_frequency", and 
                "spike_frequency_adaptation".
        local_grouping : np.ndarray
            Depth-dependent class assignments corresponding to each sample.

        Returns
        -------
        self : Decoupling
            The fitted decoupling model. Labels are stored in `self.labels_`.
        """
        # Compute cluster averages
        cluster_means = (
            features
            .groupby(local_grouping)
            .mean(numeric_only=True)
        )

        # Validate schema
        check_schema_features(self.schema, cluster_means.columns)

        # Set the penultimate key in schema as the default label
        default_label = list(self.schema.keys())[-2]

        # Assign cluster-level labels
        cluster_to_label = {}
        for clust_id, row in cluster_means.iterrows():
            assigned_label = None

            for label, rules in self.schema.items():
                logic = rules.get("_logic", "and")
                combine = operator.and_ if logic.lower() == "and" else operator.or_

                # Evaluate all rules for this group
                conditions = [
                    evaluate_rule(row, feat, rule)
                    for feat, rule in rules.items() if not feat.startswith("_")
                ]
                if reduce(combine, conditions):
                    assigned_label = label
                    break

            cluster_to_label[clust_id] = (
                assigned_label if assigned_label is not None else default_label
            )

        # Map back to sample-level labels
        self.labels_ = np.array([cluster_to_label[c] for c in local_grouping])

        return self

    #------------------------------------------------------------------------------------#
    def fit_predict(
            self, 
            features: pd.DataFrame, 
            local_grouping: np.ndarray
        ) -> np.ndarray:
        """
        Run decoupling model and get depth-independent class assignments based on 
        cluster-averaged features.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix with columns like 
                "dendrite_vertical_extent", 
                "axon_vertical_extent", 
                "spike_frequency", and 
                "spike_frequency_adaptation".
        local_grouping : np.ndarray
            Depth-dependent class assignments corresponding to each sample.

        Returns
        -------
        np.ndarray
            Depth-independent class assignments after fitting.
        """
        self.fit(
            features=features, 
            local_grouping=local_grouping,
        )
        return self.labels_
    
##########################################################################################
    