What this project does

This is a Kosovo bank transaction fraud/AML analysis project built with two unsupervised models:

Isolation Forest

Autoencoder

It works even when the uploaded dataset has different column names, because the app includes automatic feature mapping (with fuzzy matching) and a manual override option.

You can train / retrain models from the Streamlit UI, review results, and export them.

Main features

Two unsupervised models (Isolation Forest + Autoencoder)

Train & retrain directly from the Streamlit UI

Auto-detect feature columns (fuzzy matching)

Manual mapping override (if auto-detection misses)

Configurable anomaly thresholds

KPI dashboard (live indicators)

Transaction network graph (NetworkX + Plotly)

Hover details for transactions in the graph

Model versioning + metadata

Training history logs

Download results as CSV

Multilingual UI (EN / AL)

Graceful error handling + progress indicators

File structure (core modules)

src/data/loader.py — data loading + feature mapping

src/data/preprocessing.py — preprocessing + feature engineering

src/models/isolation_forest.py — Isolation Forest model wrapper

src/models/autoencoder.py — Autoencoder model wrapper

src/models/train.py — training orchestration

src/models/retrain.py — retraining logic

src/viz/network_graph.py — interactive network visualization

src/analysis/bank_analysis.py — Kosovo bank-level analysis

src/ui/streamlit_app.py — Streamlit app entry point

__init__.py — package structure

Synthetic dataset included

The original dataset cannot be published because it contains sensitive financial data.
For this reason, this repository includes a synthetic (fake) dataset that follows the same schema and includes some suspicious-looking patterns so others can run and test the project safely.