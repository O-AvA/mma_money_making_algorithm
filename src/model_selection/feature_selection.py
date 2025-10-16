import pandas as pd
import numpy as np 
from loguru import logger
from src.utils.general import store_csv 
from src.utils.mlflow_pipeline import get_pipeline_manager
import os

class FeatureSelector:

    def __init__(self, CVmain):
        self.CVmain = CVmain 

    def _select_features(self, params): 
        """
        THank you he chatgpt
        Little bit more than I asked but sure man it works 

        Stability selection via CV:
          - Run n_repeats * n_folds XGB fits on train set
                    - In each fit, select the top-m features by importance (m capped, e.g., 50)
                    - Count how often each feature is selected (frequency); also compute mean importance and rank
                    - Final ranking sorted by: freq desc, mean_rank asc, mean_importance desc
          - CV-evaluate selected features on train folds and full valid set

        Returns:
          {
            "selected_features": List[str],
            "freq_df": pd.DataFrame,
            "cv": Dict[str, float],      # averaged across folds on train-only CV
            "valid": Dict[str, float],   # metrics on provided valid set
            "total_models": int
          }
        """
        Xt = self.CVmain.Xt
        yt = self.CVmain.yt 
        feat_names = np.array(Xt.columns)
        n_features = len(feat_names)
        n_repeats = self.CVmain.cv_params['n_repeats']
        suffix = self.CVmain.suffix
        all_folds = self.CVmain.folds

        model = self.CVmain._xgb_factory(params, xgb_seed = self.CVmain.xgb_seed)

        counts = np.zeros(n_features, dtype=int)
        imp_sum = np.zeros(n_features, dtype=float)
        rank_sum = np.zeros(n_features, dtype=float)
        total_models = 0

        logger.info('Starting cross validation for feature selection') 
        for rep in range(n_repeats):
            for k, (tr_idx, va_idx) in enumerate(all_folds[rep]): 
                X_tr, X_va = Xt.iloc[tr_idx], Xt.iloc[va_idx]
                y_tr, y_va = yt.iloc[tr_idx], yt.iloc[va_idx]

                # Early stopping happens on the fold's validation set
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

                # Try XGB built-in importances; fallback to booster gain if needed
                imps = getattr(model, "feature_importances_", None)
                if imps is None or np.all(np.isnan(imps)) or np.sum(imps) == 0:
                    try:
                        booster = model.get_booster()
                        gain = booster.get_score(importance_type="gain")
                        imps = np.array([gain.get(f"f{i}", 0.0) for i in range(n_features)], dtype=float)
                    except Exception:
                        imps = np.zeros(n_features, dtype=float)

                # Select top-m for this fold (cap m to avoid counting all features)
                order = np.argsort(-imps)  # descending importance
                m = max(1, min(150, n_features))
                top_idx = order[:m]
                counts[top_idx] += 1
                imp_sum += imps

                # Rank: 1 = best importance, larger = worse; stable tiebreaker
                ranks = np.empty(n_features, dtype=float)
                ranks[order] = np.arange(1, n_features + 1)
                rank_sum += ranks

                total_models += 1
                logger.info(f'Repeat {rep+1}/{n_repeats}, fold {k+1}/{len(all_folds[rep])} complete.')

        if total_models == 0:
            logger.error("No CV models were trained; check data shapes/labels.")
            return {"selected_features": [], "freq_df": pd.DataFrame(), "cv": {}, "valid": {}, "total_models": 0}

        freq_df = pd.DataFrame(
            {
                "feature": feat_names,
                "count": counts,
                "freq": counts / total_models,
                "mean_importance": imp_sum / total_models,
                "mean_rank": rank_sum / total_models,
            }
        ).sort_values(["freq", "mean_rank", "mean_importance"], ascending=[False, True, False]).reset_index(drop=True)

        # Write outputs
        store_csv(freq_df, self.CVmain.feature_path)

        # Log artifact in MLflow when running under pipeline
        try:
            pipeline_manager = get_pipeline_manager()
            if pipeline_manager is not None and os.path.exists(self.CVmain.feature_path):
                import mlflow
                mlflow.log_artifact(str(self.CVmain.feature_path))
        except Exception as e:
            logger.warning(f"Could not log feature frequency artifact: {e}")

