# MoneyMakingAlgorithm (MMA) — End-to-end Guide

This README walks through the entire data pipeline from (from raw data to predictions) for predicting the win method of fights in the upcoming Ultimate Bouting Championship (UFC) event, a weekly or every-other-weekly Mixed Martial Arts (MMA) tournament.

Highlights 
- Retrieves raw data, cleans raw data, engineers features, trains a model and makes predictions for upcoming Ultimate Fighting Championship (UFC) fights. 
- Predicts 7 classes: KO win, Submission win, Decision win, Draw, Decision Loss, Submission Loss, KO loss. 
- Creates probability distributions for each of the 7 classes that can then be used for price estimation and risk analysis
- Makes predictions using a self-devised elo-based rating system and other features. 
- Machine Learning method used: Extreme Gradient Booster (`xgboost`)
- Minimizes hyperparameters using n-repeated m-fold cross valuation. 
- Includes singular value decomposition, feature selection and other optional data processing features.

Notes 
- This is a light version of the model with fewer feature sets, and is only about 60% accurate. Also, the data cleaning module is disabled. You can, however, still use this code this play around and make your own predictions with.
- The full pipeline is also contained in the notebook and in scripts/full_pipeline.py

DISCLAIMER 
- This is not to be used for gambling! The purpose of this project is to obtain statistics for price estimation and risk analysis. If this model does what it should, it would tell you sportsbook offer unfavorable prices!

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

## 1. Cleaning Data

Raw ufcstats.com datasets are loaded from Greco1899's scape_ufc_stats repository. These datasets are regularly updated by the corresponding scraper (https://github.com/Greco1899/scrape_ufc_stats). Second, datasets are cleaned and merged into a single data set with basic features are created (date, height, time format etc...) and made ready for further feature engineering. Output csv is saved as `data/interim/clean_ufcstats-com_data.csv`.

Key module: `src/data_processing/clean_raw_data.py`
- Main function: `process_all_data(prefer_external=True, new_fights_only=False)`
- Core class: `UFCDataProcessor`

Notes:
- Only events from UFC 31: Locked and Loaded (May 04, 2001) are included, because this is the first standardized UFC event (using the Unified Rules of MMA and has only 3 or 5 round fights. 
- Set `new_fights_only=False` (WIP otherwise).
- Not all names in ufc_fight_results.csv match with those in ufc_fighter_tott. In addition, some fighters in ufc_fighter_tott have the same name, or are mentioned double with (one with stats, the other without etc). These issues have been resolved for all retired and currently active fighters, but may arise again in the future for newly debuting fighters. In this case, the code terminates and user must follow the instructions in the log and comments to implement a simple hard-code fix. 
- If there will ever be a Catch Weight bout where both fighters are debuting, sex cannot be inferred. Code termination for manual appending sex to 'interim/unknown_sex.csv'. One could also implement a AI that recognizes male/female names to solve this.   

Python:

```python
from src.data_processing.clean_raw_data import process_all_data

# prefer_external=True tries to pull fresh CSVs from GitHub; falls back to local files if it fails
process_all_data(prefer_external=True, new_fights_only=False)
# Expected: data/interim/clean_ufcstats-com_data.csv
```

---

## 2. Constructing feature sets

+ We now construct both our desired feature sets, and the feature sets they depend on. 

Key module: `src/data_processing/feature_manager.py`
- Class: `FeatureManager(feature_set_names=None, feature_set_params=None, overwrite_all=True)`
- Feature modules live in `src/feature_engineering` and are imported dynamically (e.g., `get_base_features.py`).

Finished sets to choose from:
- base_features (always include)
- elo_params (always include for elo feature sets)
- wl_elos

Regarding elo_params 
- get_elo_params creates multiple K-parameters which can then be chosen by the feature model using 'which_K' ('cust' or 'log', per round or not per round). 
- I believe that currently K-parameters that are not used by the other elo feature set automatically discarded in the final model, but I have to double check. In any case, it may be worthwhile keeping them in, because the K-parameters acts as an experience measures for the fighters, so not only help the model understand the elo rating system but also directly help it understand the data. 

Notes:
- Keep the param `process_upcoming_fights=False` at this stage (otherwise handled separately).
Python (corrected):

```python
from src.data_processing.feature_manager import FeatureManager
from src.feature_engineering.get_elo_params import set_elo_params

feature_sets = {}

# Parameters per feature module
base_features_params = {}
elo_params = {"d_params": set_elo_params()}  # provide Elo parameters
wl_elos_params = {"which_K": "log"}

# Choose final feature sets
feature_sets["base_features"] = base_features_params
feature_sets["elo_params"] = elo_params
feature_sets["wl_elos"] = wl_elos_params

# Create feature sets (writes CSVs under data/features/)
FeatureManager(feature_sets, overwrite_all=True)
```

Expected outputs under `data/features/`: one CSV per enabled set, e.g., `base_features.csv`, `elo_params.csv`, `wl_elos.csv`, etc.

---

## 3) Creating training set, validation set and data-to-predict (TrainValPred) and further processing

+ Key module: `src/model_selection/trainvalpred.py`
- Class: `TrainValPred(feature_sets=None)`

### 3.1 Creating the training and validation data
+ Training data is created by merging feature sets and splitting off the validation data  
+ To validate the model on relevant data, we split the validation set based on recency, either by 
last `last_years` years, or the most recent `sample_size` (proportion) of the data. The value that represents the smallest portion of the data takes precedence! When based `sample_size`, you can choose to randomly sample them from the last `last_years` of fights by setting `if_on_size_then_randomly=True` (Default False). 
+ When   
+ The snippet below creates files `interim/chosen_features_merged`, `processed/train.csv` and `procssed/valid.csv`, +
```python
from src.model_selection.trainvalpred import TrainValPred

TVP = TrainValPred(feature_sets)

# Merge all selected feature sets
TVP.merge_features(overwrite_feature_sets=True)

# Choose validation partition (recent years and sample fraction; smallest takes precedence)
last_years = 1
sample_size = 0.15

TVP.split_trainval(last_years=last_years, sample_size=sample_size)
```

Expected outputs:
- `data/interim/chosen_features_merged.csv`
- `data/processed/train.csv`
- `data/processed/valid.csv`
- `data/interim/train_names.csv` and `data/interim/valid_names.csv`

### 3.2 Scraping, cleaning and processing features for data-to-predict. 

- The snippet below runs the entire prediction data pipeline from scraping `ufcstats.com`'s upcoming event data to creating all the features. 
- Creates files: `raw/pred_raw.csv`, `interim/pred_clean`, `processed/pred.csv`, +1
- Tip: rerun this snippet if any bouts get cancelled/replaced.s

```python
from src.data_processing.scrape_pred import scrape_pred
from src.data_processing.clean_pred import clean_pred

TVP.construct_pred(scrape_and_clean=True)
```

Expected outputs:
- `data/raw/pred_raw.csv`
- `data/interim/pred_clean.csv`
- `data/processed/pred.csv`
- `data/interim/pred_names.csv`
```

### 3.3 Optional further feature processing

There are now basically three options: 
1. No further processing and go straight to training (`suffix = "natty"`) 
2. Make (anti-)symmetric features, i.e. `fighter1_feautures -> (fighter1_features + fighter2_features)/sqrt(2)` and  `fighter2_feautures -> (fighter1_features - fighter2_features)/sqrt(2)`, and leave shared features be. In this case, set `suffix = "symm"`. 
3. Do a Singular Value Decompostion(SVD) on the data `suffix = "svd"` and transform to the Schmidt basis. 

Notes 
- The SVD path first standardizes the data and also makes (anti-)symmetric pairs. However, in contrast to `symmetrize(for_svd = False)`, one-hot encoded features will not be transformed to flags. This is done because one-hot encoded features could be favorable for the SVD, but otherwise waste xgb splits. 
- Ceates datasets `processed/train_{suffix}`, `processed/valid_{suffix}`, `processed/pred_{suffix}`

```python 
suffix = 'natty'

if suffix == 'symm': 
    TVP.symmetrize(for_svd = False) 

elif suffix == 'svd':
    # Because you probably wanna check where you truncate,
    # you may have to run the SVD twice. 
    # TVP.svd(k = 10e6, plot_sv = True)
    
    TVP.do_svd(k=204)  
elif suffix == 'natty': 
    TVP.go_natty() 

```

## 4) Training and predictions

### 4.1 Hyperparameter optimization (Optuna)
```python
suffix = 'symm'
CV = CrossValidationMain('symm') 

# Optionally, change default parameters (at any point in the pipeline)
valid_params = { 
    'vv_size': 0, 
    'vv_seed': 34, 
    'vv_random_split': False
}
cv_params = { 
    'n_repeats': 3,
    'n_folds': 5, 
    'fold_seed': 42
} 
# Chose either tuple or fixed value 
hyper_params = { 
    "max_depth": 5,
    "learning_rate": (0.02, 0.025),
    "n_estimators": (400,600),
    "min_child_weight": (0, 40),
    "gamma": (0, 2.5),
    "subsample": (0.7, 0.85),
    "colsample_bytree": 1,
    # Optional regularization
    "reg_alpha": 0.0,
    "reg_lambda": 1.0
}
CV.set_valid_params(valid_params) 
CV.set_cv_params(cv_params)
CV.set_hyper_params(hyper_params)

# Initial training 
CV.optimize(n_trials = 50)
```

### 4.2 Feature selection and re-training 

+ The following code snippet automatically selects the best hyperparameters from the output metrics file and starts feature selection. It outputs file `output/feature_selection/feature_frequency`.
+ After ranking all features by their importance and counting how many times they it starts optimizing hyperparameters again but this time also varying over a range of the k_selected-th most important features.
+ Because during HPO xgb random_state is fixed, we can set `rndstate_stability_check=True` to measure the stability of the model over different seeds. It will take the `top_n` parameter combinations with the best metrics and does `n_repeats` of cross validation, where inside each fold a different random seed is chosen. The seeds-averaged metrics will be stored in `data/output/metrics/{suffix}_stability_check.csv` and the best ones are automatically retrieved for further feature selection.

```python
# Optionally, change default stability check params 
stability_check_params = {
    'top_n': 1, 
    'n_repeats': 2
}
CV.set_stability_check_params(stability_check_params)

CV.select_features(rnd_state_stability_check = True)

select_by = 'frequency'   # or by index 

if select_by == 'frequency': 
    max_freq = CV.cv_params['n_repeats'] * CV.cv_params['n_folds'] 
    feature_range = (max_freq-2, max_freq)
elif select_by == 'index': 
    max_index = len(CV.Xt.columns)
    feature_range = (50, max_index - 50) 
CV.set_feature_params(
    select_by = select_by, 
    feature_range = feature_range
) 

CV.optimize(n_trials = 30)
```

### 4.3 Making predictions

+ Now we can start making predictions. Program automatically selects the best hyperparameters and the best k_selected most important columns and calculates probabilities for each of the 7 classes.
+ The output file `output/predictions/pred_{suffix}` contains averages, standard deviations, mean +/- 2std, 5perc, 95perc, min max for each of the 7 classes. These values define the probability distributions that are created by making predictions in each of the folds of the n_repeats unique 5-folds (so `n_repeats*n_folds` unique samples).  
+ In contrast to training, for each repeat and for each fold, a different xgb random state is chosen. 
+ Probability distributions are created for both 7 classes and 2 classes (win or lose). In case of two classes, a draw basically means money back so win probabilities are calculated as $$P_{win} = \frac{P_{KO} + P_{Sub} + P_{Dec}}{1-P_{Draw}}$$
+ Based on the validation set, the model also creates a plot to show how well it's calibrated.
+ Also makes predictions for debuting fighters, but the model does not take into account previous carreer stats. This means that, with luck, only height, reach and age are available. Take this into account when competing against other models and comparing accuracies. 
+ In stead of using the model parameters with the best metrics from the previous optimization step, you can also first do a rndstate_stability_check again and use the best paramaters of those.

```python
CV.change_param(n_repeats = 200) # Takes n_repeats * n_folds samples with as many different random states
CV.predict(rndstate_stability_check=True)
```

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

---

## Acknowledements 

Massive thanks to Erik Prjadka for his invaluable advice throughout, teaching me about data science principles and which machine learning method to use. Also to Sjoerd Visser for bringing me up to date on machine learning and coding standards and practices. 

## Sources 
- www.ufcstats.com
- greco1899/ufc_stats_scraper
- pandas
- xgboost 
- numpy 
- sklearn
