# MoneyMakingAlgorithm (MMA) — End-to-end Guide

This project cleans raw UFC data, engineers features, trains an XGBoost model, and makes predictions for upcoming fights.

Highlights:
- Automatically refreshes raw data from ufcstats.com (via Greco1899’s scraper) or falls back to local copies.
- Predicts 7 classes: KO win, Submission win, Decision win, Draw, Decision loss, Submission loss, KO loss.
- Uses repeated K-fold cross validation with Optuna for hyperparameter optimization.
- Optional symmetrization and SVD transformations; built-in feature selection.
- Creates probability distributions for each of the 7 classes (optionally aggregate to 3 classes by summing probabilities).
- Note: Some cleaning modules/feature sets may remain private and only summary statistics (e.g., mean probabilities) may be shared.

The sections below mirror the Jupyter notebook `mma_project_guide.ipynb` and include the corrected code snippets you can run directly.

## Quick setup (Windows/PowerShell)

```powershell
# From the repo root (this folder)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# If xgboost-cpu wheel fails on your CPU, try:
# pip install xgboost==2.1.1
```

In VS Code, select the `.venv` interpreter for the notebook: Ctrl+Shift+P → “Python: Select Interpreter” → choose `.venv`.

---

## 1) Cleaning data

Combines the 4 raw datasets and writes a single analysis-ready CSV `data/interim/clean_ufcstats-com_data.csv`.

Notes:
- Excludes events prior to UFC 31 (first standardized event format and unified rules).
- Keep `new_fights_only=False` (WIP if set True).
- If some names/sex cannot be inferred (edge-cases), the process will write a CSV to review and stop with a clear log message.

Python:

```python
from src.data_processing.clean_raw_data import process_all_data

# prefer_external=True tries to pull fresh CSVs from GitHub; falls back to local files if it fails
process_all_data(prefer_external=True, new_fights_only=False)
# Expected: data/interim/clean_ufcstats-com_data.csv
```

---

## 2) Constructing feature sets

Feature modules live in `src/feature_engineering` and are imported dynamically by `FeatureManager`.

Feature sets you can include:
- base_features (recommended: always include)
- elo_params (always include if using Elo-based features)
- wl_elos
- stat_elos_round_averages
- stat_elos_per_round (alternative to the above; generally not recommended)
- acc_elos_round_averages
- acc_elos_per_round (alternative to the above; generally not recommended)
- rock_paper_scissor (opponent-overlap features)

Python (corrected):

```python
from src.data_processing.feature_manager import FeatureManager
from src.feature_engineering.get_elo_params import set_elo_params

feature_sets = {}

# Parameters per feature module
base_features_params = {}
elo_params = {"d_params": set_elo_params()}  # provide Elo parameters
wl_elos_params = {"which_K": "log"}
stat_elos_round_averages_params = {
    "which_K": "log",
    "exact_score": True,
    "always_update": False,
}
stat_elos_per_round_params = {
    "which_K": "log",
    "exact_score": True,
    "always_update": False,
}
acc_elos_round_averages_params = {"which_K": "log"}
acc_elos_per_round_params = {}
rock_paper_scissor_params = {"intervals": [0, 2]}  # or [0, 2, 4]

# Choose final feature sets
feature_sets["base_features"] = base_features_params
feature_sets["elo_params"] = elo_params
feature_sets["wl_elos"] = wl_elos_params
feature_sets["stat_elos_round_averages"] = stat_elos_round_averages_params
# feature_sets["stat_elos_per_round"] = stat_elos_per_round_params
feature_sets["acc_elos_round_averages"] = acc_elos_round_averages_params
# feature_sets["acc_elos_per_round"] = acc_elos_per_round_params
# feature_sets["rock_paper_scissor"] = rock_paper_scissor_params

# Create feature sets (writes CSVs under data/features/)
FeatureManager(feature_sets, overwrite_all=True)
```

Expected outputs under `data/features/`: one CSV per enabled set, e.g., `base_features.csv`, `elo_params.csv`, `wl_elos.csv`, etc.

---

## 3) Final data processing

### 3.1 Merge and split train/valid

```python
from src.model_selection.trainvalpred import TrainValPred

TVP = TrainValPred(feature_sets)

# Merge all selected feature sets
TVP.merge_features(overwrite_feature_sets=False)

# Choose validation partition (recent years and sample fraction)
last_years = 1
sample_size = 0.15
TVP.split_trainval(last_years=last_years, sample_size=sample_size)
```

Expected outputs:
- `data/interim/chosen_features_merged.csv`
- `data/processed/train.csv`
- `data/processed/valid.csv`
- `data/interim/train_names.csv` and `data/interim/valid_names.csv`

### 3.2 Build prediction (upcoming fights) dataset

```python
from src.data_processing.scrape_pred import scrape_pred
from src.data_processing.clean_pred import clean_pred

scrape_pred()
clean_pred()
TVP.construct_pred()
```

Expected outputs:
- `data/raw/pred_raw.csv`
- `data/interim/pred_clean.csv`
- `data/processed/pred.csv`
- `data/interim/pred_names.csv`

### 3.3 Optional further processing: symm or SVD

```python
suffix = "symm"

if suffix == "symm":
    # write train_symm/valid_symm/pred_symm
    TVP.symmetrize(for_svd=False)
elif suffix == "svd":
    # write train_svd/valid_svd/pred_svd
    TVP.do_svd(k=204)
```

---

## 4) Training and predictions

### 4.1 Hyperparameter optimization (Optuna)

```python
from src.model_selection.cv import CrossValidation

xgb_params = {
    "max_depth": (3, 6),
    "learning_rate": (0.02, 0.03),
    "n_estimators": (300, 750),
    "min_child_weight": (0, 50),
    "gamma": (0, 10),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.8, 1.0),
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

data_params = {
    "suffix": "symm",          # "", "symm", or "svd"
    "k_selected": None,         # leave None during initial optimization
    "vv_seed": 6,
    "vv_size": 0,               # >0 to optimize on a validation split
    "vv_random_split": False,
    "save_as_n_classes": 3,
    "measure_calibration": False,
}

cv_params = {
    "fold_seed": 30,
    "n_folds": 5,
    "n_repeats": 1,
    "n_trials": 50,
}

CV = CrossValidation(data_params, cv_params, xgb_params)
CV.run_cv(select_features=False, predict=False)
# Outputs top-50 to data/output/metrics/param_optimization_symm.csv
```

What gets logged per run:
- Hyperparameters sampled by Optuna for the trial (plus run metadata).
- Metrics averaged over n_repeats * n_folds: accuracy, logloss, and macro-F1 for both 7-class and 3-class groupings on train, and on validation partitions (vt, and vv if vv_size > 0).
- Per trial, the target accuracy is logged to console: acc3_vt if vv_size == 0, otherwise acc3_vv.

```python
# Optionally measure calibration on validation
# data_params["measure_calibration"] = True

cv_params["n_repeats"] = 200

CV = CrossValidation(data_params, cv_params, None)
CV.run_cv(select_features=False, predict=True)
# Writes data/output/predictions/pred_symm_predictions.csv
```

### 4.2 Feature selection and re-train

```python
# Feature selection pass
CV = CrossValidation(data_params, cv_params, None)
CV.run_cv(select_features=True, predict=False)

# Then vary k_selected and optimize again
data_params["k_selected"] = (50, 200)
CV = CrossValidation(data_params, cv_params, xgb_params)
CV.run_cv(select_features=False, predict=False)
```

### 4.3 Predictions
- Automatically selects the best hyperparameters (and best k_selected if present in metrics) from the optimization file; xgb_params and data_params["k_selected"] do not need to be provided for predictions.
- Outputs distribution statistics per class across n_repeats*n_folds samples: mean, std, mean±2std, p5, p95, min, max.
- Optionally aggregate to 3 classes by summing the 7-class probabilities (set save_as_n_classes=3); training remains 7-class.
- Optionally store validation probabilities to assess calibration later (measure_calibration=True).
- Also handles debuting fighters; note only limited covariates (e.g., height, reach, age) may be available, which impacts accuracy.

```python
# Optionally look at only 3 classes in the output file
data_params["save_as_n_classes"] = 3

cv_params["n_repeats"] = 200

CV = CrossValidation(data_params, cv_params, None)
CV.run_cv(select_features=False, predict=True)
# Writes data/output/predictions/pred_symm_predictions.csv
```

---

## Sanity checks after each step

- After cleaning: `data/interim/clean_ufcstats-com_data.csv`
- After features: CSVs under `data/features/` (e.g., `base_features.csv`, `elo_params.csv`, ...)
- After merge/split: `processed/train.csv`, `processed/valid.csv` + `interim/*_names.csv`
- After pred build: `processed/pred.csv` + `interim/pred_names.csv`
- After symm/SVD: `processed/train_symm.csv`/`train_svd.csv` (and matching valid/pred)
- After Optuna: `output/metrics/param_optimization_<suffix>.csv`
- After predictions: `output/predictions/pred_<suffix>_predictions.csv`

## Troubleshooting

- Do not set `new_fights_only=True` during cleaning (WIP).
- If upcoming fight scraping fails, rerun the pred pipeline later; site structure can change.
- If you see `unknown_sex.csv`, follow the log instructions to add a simple mapping and rerun.
- XGBoost install issues? Use `pip install xgboost==2.1.1`.

---

## Run in your browser (GitHub Codespaces)

Run the notebook and view MLflow entirely in the browser using GitHub Codespaces.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/REPLACE_WITH_YOUR_GITHUB_USERNAME_OR_ORG/REPO_NAME?quickstart=1)

What it does
- Spins up a cloud dev environment with Python 3.11
- Auto-installs `requirements.txt` (devcontainer provided)
- Forwards MLflow UI on port 5000

Steps
1) Click the badge above (or Code → Codespaces → Create codespace on main)
2) Open `mma_project_guide.ipynb` and run cells
3) To view MLflow UI, open a terminal and run:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

The port will auto-forward; click the globe icon in the Ports panel to open the UI in a new tab.
# mma_money_making_algorithm
