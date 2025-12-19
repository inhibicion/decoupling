# decoupling/__init__.py

from .models.depth_independent import Decoupling
from .models.consensus import ConsensusClustering
from .models.hierarchical import HierarchicalClustering
from .metrics.clusters import cluster_predictability
from .metrics.features import variance_per_sample_size
from .utils.preprocessing import set_seed, align_inputs, preprocess_features, align_and_preprocess_features
from .utils.postprocessing import compute_molecular_profiles
from .utils.visualization import create_embedding, plot_embedding, plot_embedding_by_layer, plot_decoupling_schema, plot_features_by_layer, colorbar

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
