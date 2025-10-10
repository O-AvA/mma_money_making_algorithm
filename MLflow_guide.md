# MLflow Pipeline for the MMA Project

## Overview

This document describes the MLflow setup for this project. It uses a single experiment per pipeline run, a clear parent–child run hierarchy, consistent tagging, and centralized metadata. The structure matches your current modules and paths and is designed to drop into your existing workflow with minimal changes.

## Hierarchy

A small manager in `src/utils/mlflow_pipeline.py` centralizes MLflow usage and introduces a simple hierarchy:

```
Pipeline Run (parent)
├── Stage 1: Data Cleaning (child)
├── Stage 2: Feature Engineering (child)
│   ├── Merge Features (nested child)
│   ├── Split Train/Val (nested child)
│   ├── Build Prediction Set (nested child)
│   └── Transform Data (nested child)
└── Stage 3: Model Training (child)
    ├── Hyperparameter Optimization (nested child)
    ├── Feature Selection (nested child)
    └── Final Predictions & Calibration (nested child)
```

### Highlights

- One experiment per full pipeline run
- Parent–child runs for clear lineage
- Consistent tagging (stage, substage, pipeline, git, environment)
- Centralized parameters and metadata (pipeline-, stage-, and git-level)
- Registry-ready structure for when you enable the model registry
- Sensible error handling and tagging

## Usage

```python
from src.utils.mlflow_pipeline import MLflowPipelineManager, set_pipeline_manager

# Initialize
pipeline_manager = MLflowPipelineManager(
    pipeline_name="mma_prediction",
    experiment_name="mma_pipeline_production"
)
set_pipeline_manager(pipeline_manager)

# Full pipeline
with pipeline_manager.pipeline_run(
    suffix="symm",
    feature_sets=feature_sets,
    last_years=3,
    sample_size=0.15
):
    # Stage 1: Data Cleaning
    with pipeline_manager.stage_run("data_cleaning", stage_type="processing"):
        process_all_data(prefer_external=True, new_fights_only=False)

    # Stage 2: Feature Engineering
    with pipeline_manager.stage_run("feature_engineering", stage_type="processing"):
        with pipeline_manager.substage_run("merge_features"):
            TVP.merge_features(overwrite_feature_sets=False)

        with pipeline_manager.substage_run("split_trainval"):
            TVP.split_trainval(last_years=3, sample_size=0.15)

    # Stage 3: Model Training & Predictions
    with pipeline_manager.stage_run("model_training", stage_type="training"):
        # a) Hyperparameter Optimization (Optuna) over stratified CV
        with pipeline_manager.substage_run("hyperparameter_optimization"):
            CV.optimize(n_trials=50)

        # b) Optional Feature Selection via stability across folds
        with pipeline_manager.substage_run("feature_selection"):
            CV.select_features(top_idx=0)

        # c) Final Predictions + Calibration Distributions (7-class and 2-class)
        with pipeline_manager.substage_run("final_predictions"):
            CV.predict(top_idx=0)
```

## Why this helps

- Clear pipeline picture in the MLflow UI
- Easier debugging: failures are localized and tagged
- Reproducible runs: parameters and git context are captured
- Scales well as you add stages or compare pipelines
- Ready for model registry when you are

## Status

- Completed:
  - `src/utils/mlflow_pipeline.py`
  - `src/scripts/full_pipeline.py`
  - `src/data_processing/clean_raw_data.py`
  - Partial updates in TrainValPred
- In progress:
  - Finer-grained logging inside CV loops
  - Optional registry promotion flows

## Consistency checklist with your codebase

- Imports and API surface:
  - `src/utils/mlflow_pipeline.py` exports `MLflowPipelineManager` and `set_pipeline_manager`
  - Context managers exist: `pipeline_run`, `stage_run`, `substage_run`
- Functions/objects used in the example:
  - `process_all_data(prefer_external: bool, new_fights_only: bool)`
  - `TVP.merge_features(overwrite_feature_sets: bool)`
  - `TVP.split_trainval(last_years: int, sample_size: float)`
  - `CV.optimize(n_trials: int)`
  - `CV.select_features(top_idx: int)`
  - `CV.predict(top_idx: int)`
- Output locations and filenames (ensure code writes these or update to match):
  - `output/metrics/{suffix}_metrics.csv`
  - `output/feature_selection/{suffix}_feature_frequency.csv`
  - `output/calibration/{suffix}_preds7.csv`, `{suffix}_preds2.csv`
  - `output/predictions/{suffix}_preds7.csv`, `{suffix}_preds2.csv`
  - `output/calibration/{suffix}7.png`, `{suffix}2.png`
- MLflow setup:
  - Tracking URI reachable (env/config); experiment `mma_pipeline_production` created if missing
  - Tags present: `stage`, `substage`, `pipeline` (optional: `git_*`, `env`) with the same key names used by the manager
  - Parent/child runs use nested runs as shown
- Environment and dependencies:
  - Optuna installed; XGBoost (or your chosen model) available for feature importance
  - Write access to `output/` directories; plotting libs available if used for calibration
- Git context:
  - Repo available (or manager handles absence gracefully)

## Quick validation

- Run a small slice:
  - `python src/scripts/full_pipeline.py`
- Check MLflow UI:
  - One parent run, nested children for each stage/substage
  - Tags `stage` and `substage` set correctly
- Check filesystem and artifacts:
  - CSVs and plots exist at the paths listed above
  - Artifacts are logged under the corresponding runs
- If names/paths differ:
  - Prefer updating code to write to the documented paths, or update this doc to reflect your actual outputs—keep them consistent.

## Where to adjust if naming differs

- If you use different objects (e.g., `TrainValPred` instead of `TVP`) or different filenames, update either the usage example or your code to match. Consistency matters more than the exact names.

## Stage 3: Model training and predictions

All training work happens under one stage with three substages. Outputs stay together and are easy to compare.

- Substage: hyperparameter_optimization
  - Runs Optuna over stratified CV on the training folds
  - Logs best params and objective
  - Aggregates CV metrics (7-class and derived 2-class) to:
    - `output/metrics/{suffix}_metrics.csv`

- Substage: feature_selection (optional)
  - Trains XGB across folds, aggregates importances, ranks by stability
  - Writes frequency table to:
    - `output/feature_selection/{suffix}_feature_frequency.csv`

- Substage: final_predictions
  - Retrains per fold with chosen hyperparameters
  - Produces probability distributions for:
    - Validation (for calibration): `output/calibration/{suffix}_preds7.csv`, `{suffix}_preds2.csv`
    - Upcoming fights (predictions): `output/predictions/{suffix}_preds7.csv`, `{suffix}_preds2.csv`
  - Saves calibration plots:
    - `output/calibration/{suffix}7.png`, `{suffix}2.png`
  - Everything is logged under the same nested run for side-by-side comparison

Logging details in Stage 3:
- Tags: `stage=model_training`, `substage` in {hyperparameter_optimization, feature_selection, final_predictions}
- Params: CV config, best hyperparameters, and (if used) selected top-k features
- Metrics: Aggregated CV metrics for train_valid, valid_train, and valid_valid (with prefixes)
- Artifacts: metrics CSVs, feature frequency CSV, prediction distributions, calibration plots

## Migration

1. Use the full pipeline script:
   ```bash
   python src/scripts/full_pipeline.py
   ```
2. Update module imports to use the new manager where needed
3. Modules auto-detect pipeline context and fall back if not present

## MLflow UI at a glance

- Experiments: one per end-to-end pipeline run
- Runs: nested by stage and substage
- Metrics: consistently prefixed
- Params: pipeline- and stage-level
- Artifacts: organized by stage/substage
- Tags: consistent and filterable

## Next steps

1. Test the pipeline on a small slice
2. Migrate remaining modules incrementally
3. Enable model registry when ready
4. Add A/B pipelines if you want to compare strategies