# decoupling/__init__.py

from decoupling.models.depth_independent import Decoupling
from decoupling.models.consensus import ConsensusClustering
from decoupling.models.hierarchical import HierarchicalClustering
from decoupling.metrics.clusters import cluster_predictability
from decoupling.metrics.features import variance_per_sample_size
from decoupling.utils.preprocessing import set_seed, align_inputs, preprocess_features, align_and_preprocess_features
from decoupling.utils.postprocessing import compute_molecular_profiles
from decoupling.utils.visualization import create_embedding, plot_embedding, plot_embedding_by_layer, plot_decoupling_schema, plot_features_by_layer, colorbar

__all__ = [
    "Decoupling",
    "ConsensusClustering",
    "HierarchicalClustering",
    "cluster_predictability",
    "variance_per_sample_size",
    "set_seed",
    "align_inputs",
    "preprocess_features",
    "align_and_preprocess_features",
    "compute_molecular_profiles",
    "create_embedding",
    "plot_embedding",
    "plot_embedding_by_layer",
    "plot_decoupling_schema",
    "plot_features_by_layer",
    "colorbar",
]
