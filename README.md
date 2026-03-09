# Morphoelectric properties of inhibitory neurons shift gradually and regardless of cell type along the depth of the cerebral cortex

[![Python Version](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
![Continuous Integration](https://github.com/inhibicion/decoupling/actions/workflows/ci.yml/badge.svg)
[![Coverage](https://codecov.io/gh/inhibicion/decoupling/branch/main/graph/badge.svg)](https://codecov.io/gh/inhibicion/decoupling)

- We quantify morphological and electrophysiological properties of inhibitory neurons across cortical depth.

- These properties shift gradually regardless of cell type in rat barrel cortex, mouse visual and motor cortex, and human middle temporal gyrus.

- Simple morphoelectric relationships distinguish the four main molecular cell types at any depth.

- This reveals two sources of diversity: 
    1. intrinsic developmental programs (i.e., cell type-specific and depth-independent), and 
    2. extrinsic environmental modulation (i.e., depth-dependent and cell type-independent).

## Setup

To run the code, you need to clone and install this repository locally, e.g., in the command line, run: 
```shell
git clone https://github.com/inhibicion/decoupling.git
cd decoupling
pip install -e .
```

## Running Analyses

All scripts are in the `notebooks/` folder:

- `analysis1_morphoelectric_diversity.ipynb` – Characterization of inhibitory neurons across cortical depth.  
- `analysis2_depth-dependent_gradients.ipynb` – Gradual shifts of morphoelectric properties along cortical depth.  
- `analysis3_link_to_molecular_identity.ipynb` – Depth-dependent variations reveal the molecular identity of inhibitory neurons.  
- `analysis4_preservation_across_areas_and_species.ipynb` – Generalization across cortical areas and species.

## Attribution

If you use `decoupling` consider citing our manuscript.
```
@article{YanezEtAl_Decoupling,
    title   = {Morphoelectric properties of inhibitory neurons shift gradually and regardless of cell type along the depth of the cerebral cortex}, 
    author  = {Felipe Yáñez and Fernando Messore and Guanxiao Qi and Nima Dehghani and Hanno S. Meyer and Dirk Feldmeyer and Bert Sakmann and Marcel Oberlaender},
    journal = {bioRxiv},
    year    = {2026},
	doi     = {10.64898/2026.03.05.709819},
	URL     = {https://www.biorxiv.org/content/10.64898/2026.03.05.709819}
}
```