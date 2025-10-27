
"""
Modern MMA Pipeline with Hierarchical MLflow Tracking

This script demonstrates the new MLflow structure that properly organizes
the 3-stage pipeline with parent-child run relationships.
"""
from loguru import logger

from src.utils.mlflow_pipeline import MLflowPipelineManager, set_pipeline_manager
#from src.data_processing.clean_raw_data import process_all_data
from src.data_processing.feature_manager import FeatureManager
from src.feature_engineering.get_elo_params import set_elo_params
from src.model_selection.trainvalpred import TrainValPred
from src.model_selection.cv_main import CVMain

def run_full_pipeline(skip_data_cleaning=False, skip_feature_engineering=False):
    """
    Run the complete MMA prediction pipeline with proper MLflow tracking.
    
    Args:
        skip_data_cleaning (bool): If True, skip the data cleaning stage (Stage 1)
        skip_feature_engineering (bool): If True, skip the feature engineering stage (Stage 2)
    """
    
    # Initialize the pipeline manager
    pipeline_manager = MLflowPipelineManager(
        pipeline_name="mma_prediction",
        experiment_name="mma_pipeline_production"
    )
    set_pipeline_manager(pipeline_manager)
    
    # Define feature sets
    feature_sets = {
        'base_features': {},
        'elo_params': {'d_params': set_elo_params()},
        'wl_elos': {'which_K': 'log'}
    }
    
    # Validation set parameters   # FROM THE NEXT LINE ONWARDS PASTE TO MMA_LIGHT
    suffix = 'natty'
    last_years = 5
    sample_size = 0.08
    if_on_size_then_randomly = False

    # Start the complete pipeline
    with pipeline_manager.pipeline_run(
        suffix=suffix,
        feature_sets=feature_sets,
        last_years=last_years,
        sample_size=sample_size,
        skip_data_cleaning=skip_data_cleaning,
        skip_feature_engineering=skip_feature_engineering
    ):
        
        # Stage 1: Data Cleaning (conditional)
        if not skip_data_cleaning:
            with pipeline_manager.stage_run(
                "data_cleaning", 
                stage_type="processing",
                prefer_external=True,
                new_fights_only=False
            ):
                process_all_data(prefer_external=True, new_fights_only=False)
        else:
            logger.info("Skipping data cleaning stage...")
        
        # Stage 2: Feature Engineering & Dataset Preparation (conditional)
        if not skip_feature_engineering:
            with pipeline_manager.stage_run(
                "feature_engineering",
                stage_type="processing", 
                feature_sets=feature_sets
            ):
                # Note: Substages are handled inside called modules to avoid duplicate MLflow runs
                FeatureManager(feature_sets, overwrite_all=False)

                TVP = TrainValPred(feature_sets)
                TVP.merge_features(overwrite_feature_sets=False)
                cfm = TVP.open_merged_features()
                cfm = cfm[col for col in cfm.columns if 'overlap' not in col]
                store_csv(csm, get_data_path('interim') / 'chosen_features_merged.csv')
                TVP.split_trainval(
                        last_years=last_years, 
                        sample_size=sample_size, 
                        if_on_size_then_randomly = if_on_size_then_randomly
                ) 
                TVP.construct_pred(scrape_and_clean=False)

                if suffix == 'symm':
                    TVP.symmetrize(for_svd=False)
                elif suffix == 'svd':
                    TVP.do_svd(k=225)
                elif suffix == 'natty': 
                    TVP.go_natty() 
        else:
            logger.info("Skipping feature engineering stage...")
            # Still need TVP for model training stage
            TVP = TrainValPred(feature_sets)
            
        # Stage 3: Model Training & Prediction
        with pipeline_manager.stage_run(
            "model_training",
            stage_type="training",
            suffix=suffix
        ):
            # Configure CV
            CV = CVMain(suffix)

            valid_params = {
                'vv_size': 0,
                'vv_seed': 34,
                'vv_random_split': False
            }
            cv_params = {
                'n_repeats': 1,
                'n_folds': 5,
                'fold_seed': 42
            }


            ######################  INITIAL TRAINING
            hyper_params = {
                "max_depth": (3,7),
                "learning_rate": (0.018,0.03),
                "n_estimators": (600,800),
                "min_child_weight": 10,
                "gamma": 1,
                "subsample": (0.75,0.85),
                "colsample_bytree": 1,
                # Optional regularization
                "reg_alpha": 0.0,
                "reg_lambda": 1.0
            }
            CV.set_valid_params(valid_params)
            CV.set_cv_params(cv_params)
            CV.set_hyper_params(hyper_params)

            # Hyperparameter optimization → Feature selection → Final predictions
            CV.optimize(n_trials=60)

            ##################### FEATURE SELECTION
            stability_check_params = {
                    'top_n': 5, 
                    'n_repeats': 2
            }
            CV.set_stability_check_params(stability_check_params)
            CV.change_cv_param('n_repeats', 3)
            CV.select_features(rndstate_stability_check=True)

            ##################### RETRAINING
            select_by = 'index'   # by index or frequency

            if select_by == 'frequency': 
                max_freq = CV.cv_params['n_repeats'] * CV.cv_params['n_folds'] 
                #feature_range = (max(max_freq-2, 1), max_freq)
                feature_range = (12, 15) 
            elif select_by == 'index': 
                #max_index = len(CV.Xt.columns)
                #feature_range = (50, max_index - 50) 
                feature_range = (60,180)
            CV.set_feature_params(
                    select_by = select_by, 
                    feature_range = feature_range
            ) 

            #CV.change_cv_param('n_repeats', 2) 
            CV.optimize(n_trials=60)

            ##################### PREDICTING

            CV.change_cv_param('n_repeats', 70)
            CV.change_cv_param('fold_seed', None)
            CV.predict(rndstate_stability_check=True)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MMA prediction pipeline")
    parser.add_argument("--skip-data-cleaning", action="store_true", 
                       help="Skip the data cleaning stage (Stage 1)")
    parser.add_argument("--skip-feature-engineering", action="store_true",
                       help="Skip the feature engineering stage (Stage 2)")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        skip_data_cleaning=args.skip_data_cleaning,
        skip_feature_engineering=args.skip_feature_engineering
    )
