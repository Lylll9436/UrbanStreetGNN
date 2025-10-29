# UrbanStreetGNN

UrbanStreetGNN is a collection of PyTorch Geometric projects for analyzing urban street networks with graph neural networks. The repository centers on an unsupervised edge-embedding pipeline that scores the structural importance of street segments, and it also ships graph autoencoder baselines for learning city-level representations.

## Key capabilities
- **Edge importance scoring** – Encode node and edge attributes, propagate messages with configurable GNN layers (GCN/GAT/SAGE), and infer 0–1 structural scores for every street segment.
- **Graph autoencoders** – Train GAE, VGAE, and GraphMAE models for graph-level embeddings when you need whole-city descriptors.
- **Experiment orchestration** – Reproducible scripts and configs for data checks, training, evaluation, and post-hoc analysis with reports and visualisations.

## Repository layout
```
UrbanStreetGNN/
├── models/                     # Source code, configs, data caches, trained weights, analysis helpers
│   ├── scripts/                # Training/evaluation scripts for edge embedding and autoencoders
│   ├── config/                 # JSON/YAML configuration files
│   ├── data/                   # Expected location for processed ego-graphs and cached tensors
│   ├── gnn_models/             # Saved checkpoints and model weights
│   ├── outputs/                # Generated reports, plots, and embeddings
│   └── analysis/               # Post-processing utilities and notebooks
├── ego_graph_results/          # Sample ego-graph visualisations
├── data_for_test/              # Example inputs for quick smoke tests
├── *.ipynb                     # Exploratory notebooks documenting datasets and experiments
└── *.png                       # Reference figures describing graph statistics
```

Refer to `models/README.md` for a deep dive into the edge-embedding architecture and training workflow, and `models/README_GRAPH_AUTOENCODERS.md` for a side-by-side comparison of the autoencoder variants.

## Getting started
### Prerequisites
- Python 3.9+
- PyTorch with CUDA support if you plan to train on GPU
- Core libraries: `torch`, `torch-geometric`, `networkx`, `numpy`, `matplotlib`, `scikit-learn`, `seaborn`, `optuna`

Create a virtual environment and install dependencies manually:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torch-geometric networkx numpy matplotlib scikit-learn seaborn optuna
```

### Data expectations
Edge-embedding experiments expect pickled ego-graphs stored under `models/data/`. Each item should include:

```python
{
    "graph": networkx.Graph(),  # with node/edge attributes described below
    "id": "unique_graph_id"
}
```

- Node attributes: `pos` (longitude, latitude), `degree`, `centrality` (degree, betweenness, closeness)
- Edge attributes: `highway`, `width`, `length`, `geometry`

Ensure your processed data follows this schema before launching the training scripts.

## Running the pipelines
### Edge-embedding workflow
Run the full pipeline—data checks, training, evaluation, and reporting—with:
```bash
cd models/scripts
python run_experiment.py
```

You can also execute stages independently:
```bash
python train_edge_embedding.py     # Train the EdgeEmbeddingModel
python evaluate_model.py           # Generate metrics and diagnostic plots
python analyze_results.py          # Post-process embeddings and score distributions
```

### Graph autoencoders
Graph-level representation learning lives in the same `models/scripts` folder:
```bash
python train_gae.py
python train_vgae.py
python train_graphmae.py
```
Evaluation helpers and Optuna-based tuning scripts are provided as:
```bash
python evaluate_autoencoders.py
python optuna_tune_gae.py
python optuna_tune_vgae.py
python optuna_tune_graphmae.py
```
Check the autoencoder README for configuration details and recommended defaults.

### Configuration management
Hyperparameters, feature definitions, and training settings are stored in `models/config/`. When you add new features or adjust the data schema, update the corresponding config files so that training and evaluation scripts stay in sync.

## Outputs and analysis
Successful runs populate:
- `models/gnn_models/` with trained weights and checkpoints.
- `models/outputs/` with embedding tensors, structural importance reports, and visualisations.
- `models/analysis/` with helper scripts (e.g., score distribution inspection, ego-graph inspection) to interpret results.

Review the generated artifacts before customizing the pipeline to confirm the expected behaviour.

## Notebooks and figures
Top-level notebooks such as `RouteModel.ipynb` and `Visualization.ipynb` document exploratory analysis, while figures like `largest_graphs.png` and `nodes_edges_relationship.png` summarise dataset statistics. These resources are a good primer on the data characteristics feeding the GNN pipelines.

## Suggested next steps for newcomers
1. **Reproduce the baseline** – Run `python models/scripts/run_experiment.py`, then inspect the metrics and plots emitted under `models/outputs/` to understand the edge score distributions.
2. **Study the autoencoder README** – Cross-reference the conceptual diagrams with `train_gae.py`, `train_vgae.py`, and `train_graphmae.py` to see how each encoder/decoder is implemented.
3. **Explore sample data** – Load a pickle from `models/data/` in a notebook to get comfortable with the expected node/edge attribute schema.
4. **Review analysis scripts** – `models/analysis/analyze_results.py` and companions illustrate how to interpret embeddings and derive actionable insights.
5. **Tweak configurations** – Adjust files under `models/config/` when trying new features, message-passing layers, or Optuna sweeps.

## Contributing
Feel free to open issues or pull requests if you add new datasets, models, or analysis utilities. Please accompany substantial changes with documentation updates and, when possible, include sample outputs in `models/outputs/` for reviewers to replicate.

## License
No license information is currently provided. If you plan to distribute derived work, coordinate with the maintainers to clarify usage terms.
