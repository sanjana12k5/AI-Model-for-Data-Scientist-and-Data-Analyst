🚀 Project goals / vision

This repository is intended to be a single place to prototype an end-to-end fintech / e-commerce ML workflow:

Upload or generate datasets (synthetic generator included)

Flexible column-mapping UI so any CSV schema can be used

In-browser EDA and statistical testing

Train a pure deep-learning model (PyTorch) for fraud detection

Use learned merchant/user embeddings for simple recommendations

Profit & loss calculations and basic business analytics

Prediction UI to score single rows

Everything runs locally (no external calls)

✅ Features implemented (current)

Streamlit frontend (app.py)

Dataset upload + synthetic dataset generator

Full EDA (descriptive stats, histograms, correlation heatmap)

Basic statistical tests (t-test)

Column-mapping UI (choose which CSV columns correspond to canonical fields)

Robust preprocessing (type coercion, safe numeric conversion)

Training button that calls the trainer in src/train_deep.py

Prediction UI (uses saved model + metadata)

Recommendation UI (top-K merchants using embedding similarity)

Profit/Loss analysis UI (safe numeric conversion + user-selectable revenue column)

Debug info + full traceback on errors to aid development

Training script (src/train_deep.py)

Accepts df + mappings (actual column → canonical name)

Resets indices to avoid DataLoader indexing errors

Coerces types (IDs → strings, amount → float, label → 0/1)

Builds categorical maps and embeddings

StandardScaler for numeric features

Trains and saves models/model.pt, models/scaler.pkl, models/metadata.pkl

Returns best validation AUC

Robustness fixes:

Auto-detect / user-select revenue column for finance calculations

Safe numeric parsing (commas, strings)

Defensive UI to prevent typical user errors (non-numeric amount, missing label)

No external API dependencies — runs offline

🎯 Planned / Proposed features (roadmap)

Below are features planned, prioritized roughly top → bottom.

High priority / near-term

 Model monitoring: simple UI to display training curves, validation metrics, and model artifact versions.

 Stream training logs to Streamlit in real-time (progress bar + epoch logs).

 Model versioning & checkpointing (timestamped model saves with metadata).

 Auto-detection & suggestion of recommended columns (heuristics to suggest mappings to user).

 Input validation step that warns about extremely imbalanced labels or tiny dataset size.

 Hyperparameter UI for epochs, batch size, learning rate in Streamlit.

Medium term

 More models: add LSTM/Temporal CNN for time-series forecasting, classification ensembles, and an experimental LightGBM baseline for comparison (optional).

 Explainability: SHAP or integrated gradients for per-sample feature importance.

 Clustering & segmentation: customer segmentation dashboard + visualization.

 A/B test analysis helper: automatic calculation of sample size and experiment metrics.

Long term / nice-to-have

 Dockerization and one-command deployment

 Authentication & multi-user support for a shared lab environment

 Export reports: PDF/PowerPoint EDA and training summaries

 Feature store / persistence: versioned features for reproducible pipelines

 CI integration: tests for data schema, training smoke-tests, linter

📁 Recommended repository layout
.
├── app.py                      # Streamlit frontend
├── requirements.txt            # pinned dependencies
├── README.md
├── data/
│   ├── transactions.csv        # synthetic / uploaded data
│   └── uploaded.csv
├── models/
│   ├── model.pt
│   ├── scaler.pkl
│   └── metadata.pkl
├── outputs/
├── src/
│   └── train_deep.py           # training pipeline, called by app.py
└── .gitignore

🔧 Installation & quick start

Clone the repo:

git clone <repo-url>
cd <repo>


Create and activate a virtual environment (recommended):

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Recommended core packages: streamlit, pandas, numpy, matplotlib, seaborn, torch, scikit-learn, joblib, statsmodels.

Run Streamlit:

streamlit run app.py


Workflow

Data tab: upload your CSV or generate synthetic dataset.

Train tab: map your columns, inspect debug info, start training.

Predict / Recs / Profit tabs: use trained model and metadata.

📝 Data expectations / canonical schema

Trainer expects the canonical column names after mapping:

user_id (categorical — string or numeric)

merchant_id (categorical)

amount (numeric — transaction value)

country (categorical)

device_type (categorical)

label (binary 0/1 target)

The Streamlit UI provides a mapping step so your actual CSV column names can be remapped to these canonical names before training.

⚠️ Known issues & troubleshooting

Embedding size mismatch: If you train with one architecture and change embedding sizes in the app, model load will fail. Solution: use the matching architectures (the provided train_deep.py and app.py are aligned).

KeyError: numeric key like 6918: Caused by non-reset index before DataLoader. Fixed in trainer by resetting indices.

Type errors in profit tab: If revenue column contains strings like "$1,000" the UI coerces to numeric; if conversion fails values default to 0 — check raw data for bad formats.

Very imbalanced labels: Model might report poor AUC or degenerate accuracy; use sampling strategies or additional features.

🧪 Tests / Validation suggestions

Unit-test data mapping: ensure rename mapping produces required canonical columns.

Smoke test: run trainer on a small synthetic dataset to confirm training pipeline works.

Integration tests: simulate full flow (upload → map → train → predict).

👥 Contributing

Contributions are welcome! Suggested workflow:

Fork the repo

Create a branch for your feature/fix (feature/xxx or fix/yyy)

Add tests and update README where necessary

Open a PR describing your changes and why
