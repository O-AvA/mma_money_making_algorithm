import numpy as np 
import pandas as pd 
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, log_loss, accuracy_score
from xgboost import XGBClassifier
import time 
import optuna
from loguru import logger 

from src.utils.general import get_data_path, ensure_dir, open_csv, creaopen_file, store_csv
from src.model_selection.cv_params import CVParams as CVparams
from src.model_selection.cv_dataloader import CVDataloader as DataLoader
from src.model_selection.process_predictions import PredictionProcessor as PrPr
from src.model_selection.feature_selection import FeatureSelector as FS
from src.utils.mlflow_pipeline import get_pipeline_manager, substage_run


class CVMain: 
    """
    Main data pipeline for model training and making predictions and 
    encompasses four stages stages. 

    (1) Finding optimal parameters for full model 
    (2) Feature selection based on optimal parameters
    (3) Retraining with reduced features
    (4) Making predictions based on optimal parameters and 
        simultaneously measuring the calibration of the model. 

    Utilities: 
    - Hyperparameter optimization / model training using Optuna
    - Feature selection 
    - Measuring model calibration 
    - Predicting upcoming UFC fights

    All of which are done using a stratified cross validation. 
    """

    def __init__(self, suffix):
        """
        Initializes Cross Validation instance. 

        Args: 
            suffix str: 
                Select which model you want from the previous feature
                processing step: '', 'symm' or 'svd' 

        """
        self.suffix = suffix 

        output_path = get_data_path('output') 
        metrics_path = output_path / 'metrics'
        feature_selection_path = output_path / 'feature_selection'
        predictions_path = output_path / 'predictions'
        calibration_path = output_path / 'calibration'

        ensure_dir(output_path) 
        ensure_dir(metrics_path) 
        ensure_dir(feature_selection_path) 
        ensure_dir(predictions_path) 
        ensure_dir(calibration_path) 

        self.metrics_path = metrics_path / f'{suffix}_metrics.csv' 
        self.feature_path = feature_selection_path / f'{suffix}_feature_frequency.csv' 
        self.preds_path = predictions_path / f'{suffix}_preds.csv'
        self.cal_preds_path = calibration_path / f'{suffix}_preds.csv' 
        self.cal_plot_path = calibration_path / f'{suffix}.png' 

        #self.pipeline_stage = 0 

        self.set_valid_params = CVparams(self).set_valid_params
        self.set_cv_params = CVparams(self).set_cv_params
        self.set_feature_params = CVparams(self).set_feature_params
        self.set_hyper_params = CVparams(self).set_hyper_params
        self.set_stability_check_params = CVparams(self).set_stability_check_params 

        self.change_cv_param = CVparams(self).change_cv_param

        self.xgb_seed = 2025
   
    def optimize(
            self, 
            n_trials, 
            sort_by = 'logloss7'
    ):
        """
        Optuna hyperparameter optimization using a stratified 
        cross validation. 
        
        Args: 
            n_trials int: 
                Number of Optuna trials 
            sort_by str: 
                Remove this. 
                Which metric to sort output metrics file on. 
                Options: 
                logloss7, logloss2, f1macro2 or f1macro7. 
                2 and 7 denote the number of classes. 
                NOTE: The model will always train on 7 classes 
                and mlogloss!! 
        """
        DataLoader(self)._load_train() 
        DataLoader(self)._load_valid() 

        # Wrap optuna optimization as a substage when in pipeline context
        pipeline_manager = get_pipeline_manager()
        in_pipeline = pipeline_manager is not None

        def _run_optuna():
            sampler = optuna.samplers.TPESampler(seed=42)
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
            # Unique study name to avoid MLflow param conflicts across runs
            study_name = f"xgb_{self.suffix}_{int(time.time())}"
            study = optuna.create_study(
                direction='minimize',
                sampler=sampler,
                pruner=pruner,
                study_name=study_name
            )

            # Optional per-trial logging
            pipeline_mgr = get_pipeline_manager()
            def _callback(study, trial):
                if pipeline_mgr is not None:
                    try:
                        # Log each trial's params and value under a trial-specific namespace
                        pipeline_mgr.log_params_safe(trial.params, prefix=f"optuna.trial{trial.number}")
                        pipeline_mgr.log_metrics_safe({"objective": trial.value}, prefix="optuna", step=trial.number)
                    except Exception:
                        pass

            study.optimize(self._objective, n_trials=n_trials, callbacks=[_callback])
            return study

        if in_pipeline:
            with substage_run(
                "hyperparameter_optimization",
                n_trials=n_trials,
                cv_params=self.cv_params,
                valid_params=self.valid_params
            ):
                study = _run_optuna()
                # Log best params/metric with study-scoped namespace
                try:
                    best_params = study.best_trial.params if study.best_trial else {}
                    pipeline_manager.log_params_safe(best_params, prefix=f"optuna.{study.study_name}.best")
                    pipeline_manager.log_metrics_safe({"optuna.best_value": study.best_value})
                except Exception:
                    pass
        else:
            study = _run_optuna()

        for_which = 'valid_train' if self.valid_params['vv_size'] == 0 else 'train' 
        logger.info(f'Lowest logloss found for {for_which}: {study.best_trial.value}') 

        #self.pipeline_stage = 1 if self.pipeline_stage == 0 else 3

    def predict(
            self, 
            top_idx = 0,
            rndstate_stability_check = True
    ):
        """
        Makes predictions and measures calibration.

        Args:
            n_repeats:
                Number of repeats of the crossvalidation to obtain
                n_repeats x n_folds samples.
            top_idx:
                Which row to pick from the output metrics file.
            rndstate_stability_check bool: 
                During HPO and FS, xgb random state is fixed. Therefore, 
                before predicting, first do a stability check on the top_n best 
                metrics and predict on the best one
            rndstate_stability_check bool: 
                During HPO, xgb random state is fixed. Therefore, 
                before predicting, first do a stability check on the top_n best 
                metrics and predict on the best one

        Returns:
            Writes the following files:
            - calibration/{suffix}_preds2.csv
            - calibration/{suffix}_preds7.csv
            - predictions/{suffix}_preds2.csv
            - predictions/{suffix}_preds7.csv

        Notes:
            Probability matrices shape expected by PredictionProcessor:
            rows = (n_rows_in_X) * 7, flattened row-major (for each row, classes 0..6),
            cols = number of model samples produced.
        """
        # Retrieve best hyperparameters and (optionally) number of features
        # from the stored metrics.
        if rndstate_stability_check: 
            top_n = self.stability_check_params['top_n'] 
            n_repeats = self.stability_check_params['n_repeats']
            # If running under MLflow pipeline, wrap stability check as its own substage
            pipeline_manager = get_pipeline_manager()
            in_pipeline = pipeline_manager is not None
            ssc_metrics_path = str(self.metrics_path).replace('.csv', '_rndstate_stability_check.csv')
            if in_pipeline:
                with substage_run("rndstate_stability_check", top_n=top_n, n_repeats=n_repeats):
                    self._xgb_seed_stability_check()#top_n=top_n, n_repeats=n_repeats)
                    # Log the stability-check CSV if present
                    try:
                        pipeline_manager.log_data_artifact(ssc_metrics_path, artifact_type="stability")
                    except Exception:
                        pass
            else:
                self._xgb_seed_stability_check()#top_n=top_n, n_repeats=n_repeats)
            metrics_path = ssc_metrics_path
        else:
            metrics_path = self.metrics_path
        metrics = open_csv(metrics_path).iloc[top_idx]
        params = {k: metrics.get(k) for k in self.hyper_params.keys()}
        params['top_k_feats'] = metrics.get('top_k_feats')

        n_repeats = self.cv_params['n_repeats']

        # Reload train to get folds and prediction matrices
        #if not hasattr(self, 'folds'):
        DataLoader(self)._load_train()
        folds_list = self.folds

        # Set to 1 to do incrementally
        CVparams(self).change_cv_param('n_repeats', 1)

        if not hasattr(self, 'Xvt'):
            DataLoader(self)._load_valid()

        # If Xvt has been used to minimize logloss on, we can't use it
        # for model calibration.
        Xv = self.Xvt if self.valid_params['vv_size'] == 0 else self.Xvv
        yv = self.yvt if self.valid_params['vv_size'] == 0 else self.yvv

        # Load pred so that _cross_validate() can call it
        DataLoader(self)._load_pred()

        total_samples = len(folds_list) * self.cv_params['n_folds']
        v_probs7 = np.empty((len(Xv) * 7, total_samples))
        p_probs7 = np.empty((len(self.Xp) * 7, total_samples))

        logger.info('Starting CV for making predictions.')

        og_ss = params['subsample']
        og_cst = params['colsample_bytree']
        var_range = 0.05

        #ss_range = (min(0, og_ss-var_range), min(og_ss+var_range,1))
        #cst_range = (min(0, og_cst-var_range), min(og_cst+var_range,1))

        for k, folds in enumerate(folds_list):
            self.folds = [folds]
            for j in range(self.cv_params['n_folds']):
                self.folds = [[folds[j]]]

                #params['subsample'] = np.round(np.random.uniform(ss_range[0], ss_range[1]), 5) 
                #params['colsample_bytree'] = np.round(np.random.uniform(cst_range[0], cst_range[1]), 5)
                # Varying over subsample and colsample_bytree
                params['subsample'] = max(0, min(np.random.normal(og_ss, var_range), 1))
                params['colsample_bytree'] = max(0, min(np.random.normal(og_cst, var_range), 1))

                xgb_seed = np.random.randint(0,10e6)
                model = self._cross_validate(params, xgb_seeds=[xgb_seed], mlflow_prefix=None)

                # We have to reload Xv because _cross_validation changes columns 
                Xv = self.Xvt if self.valid_params['vv_size'] == 0 else self.Xvv

                yv_prob7 = model.predict_proba(Xv)
                yp_prob7 = model.predict_proba(self.Xp)

                idx = k * self.cv_params['n_folds'] + j
                v_probs7[:, idx] = yv_prob7.reshape(-1)
                p_probs7[:, idx] = yp_prob7.reshape(-1)

                logger.info(f'Sample {idx}/{len(folds_list)*self.cv_params["n_folds"]} predicted.')

        # Save validation (with ytrue) and prediction distributions for 7-class and 2-class.
        pipeline_manager = get_pipeline_manager()
        in_pipeline = pipeline_manager is not None
        if in_pipeline:
            with substage_run("final_predictions", top_idx=top_idx, total_samples=total_samples):
                PrPr(self)._save_probas(v_probs7, ytrue=yv)
                PrPr(self)._save_probas(p_probs7)
                # Log artifacts
                try:
                    from pathlib import Path
                    cal7 = Path(str(self.cal_preds_path).replace('preds', 'preds7'))
                    cal2 = Path(str(self.cal_preds_path).replace('preds', 'preds2'))
                    p7 = Path(str(self.preds_path).replace('preds', 'preds7'))
                    p2 = Path(str(self.preds_path).replace('preds', 'preds2'))
                    plot7 = Path(str(self.cal_plot_path).replace(self.suffix, f"{self.suffix}7"))
                    plot2 = Path(str(self.cal_plot_path).replace(self.suffix, f"{self.suffix}2"))
                    for fp in [cal7, cal2, p7, p2, plot7, plot2]:
                        if fp.exists():
                            import mlflow
                            mlflow.log_artifact(str(fp))
                except Exception:
                    pass
        else:
            PrPr(self)._save_probas(v_probs7, ytrue=yv)
            PrPr(self)._save_probas(p_probs7)
        #PrPr(self)._save_probas(v_probs7, save_as_n_classes=2, ytrue=yv)
        #PrPr(self)._save_probas(p_probs7, save_as_n_classes=2)

    def select_features(self, rndstate_stability_check=True,top_idx = 0): 
        """
        Performs feature selection, ranks columns by their importance

        Args: 
            rndstate_stability_check bool: 
                During HPO, xgb random state is fixed. Therefore, 
                before predicting, first do a stability check on the top_n best 
                metrics and predict on the best one
            topValidation set 
                Chose the hyperparams with top_idx'th best metrics 
        """
        if rndstate_stability_check: 
            top_n = self.stability_check_params['top_n'] 
            n_repeats = self.stability_check_params['n_repeats']
            # If running under MLflow pipeline, wrap stability check as its own substage
            pipeline_manager = get_pipeline_manager()
            in_pipeline = pipeline_manager is not None
            ssc_metrics_path = str(self.metrics_path).replace('.csv', '_rndstate_stability_check.csv')
            if in_pipeline:
                with substage_run("rndstate_stability_check", top_n=top_n, n_repeats=n_repeats):
                    self._xgb_seed_stability_check()#top_n=top_n, n_repeats=n_repeats)
                    # Log the stability-check CSV if present
                    try:
                        pipeline_manager.log_data_artifact(ssc_metrics_path, artifact_type="stability")
                    except Exception:
                        pass
            else:
                self._xgb_seed_stability_check()#top_n=top_n, n_repeats=n_repeats)
            metrics_path = ssc_metrics_path
        else:
            metrics_path = self.metrics_path
        metrics = open_csv(metrics_path).iloc[top_idx] 
        params = {k: metrics.get(k) for k in self.hyper_params.keys()} 
        log_params = {
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in params.items()
        }

        DataLoader(self)._load_train() 
        logger.info(f'Starting feature selection for {log_params}') 

        pipeline_manager = get_pipeline_manager()
        in_pipeline = pipeline_manager is not None
        if in_pipeline:
            with substage_run("feature_selection", **log_params):
                FS(self)._select_features(params)
        else:
            FS(self)._select_features(params)

        #self.pipeline_stage = 2

    def _xgb_seed_stability_check(self): #, top_n = 5, n_repeats=2):
        """
        HPO and feature selection will all be done on the same xgb_seed. 
        Therefore, before making predictions, it may be worthwhile 
        checking stability on xgb_seed. 

        It will take the top_n best metrics and tests them and then uses 
        the best one for making predictions.

        Args
            top_n int: 
                Do stability check for the top_n best metrics and pick 
                the best one 
            n_repeats int: 
                Overrides self.cv_params['n_repeats'] 
        """
        if not hasattr(self, 'stability_check_params'):
            CVparams(self).set_stability_check_params()

        top_n = self.stability_check_params['top_n']
        n_repeats = self.stability_check_params['n_repeats']

        xgb_seeds = np.random.randint(0,10000,size=n_repeats*self.cv_params['n_folds'])

        # Saving current n_repeats and setting to custom one
        og_n_repeats = self.cv_params['n_repeats'] 
        CVparams(self).change_cv_param('n_repeats', n_repeats)

        # Reload X to get folds corresponding to new n_repeats
        DataLoader(self)._load_train()
        if not hasattr(self, 'Xvt'):
            DataLoader(self)._load_valid()

        logger.info(f'Checking xgb random_state stability.')

        ssc_metrics_path = str(self.metrics_path).replace('.csv', '_rndstate_stability_check.csv')
        df_ssc = creaopen_file(ssc_metrics_path)

        for j in range(top_n):
            dfm = open_csv(self.metrics_path)
            metrics = dfm.iloc[j]
            params = {k: metrics.get(k) for k in self.hyper_params.keys()}
            if 'All' not in metrics.get('top_k_feats'):
                params['top_k_feats'] = metrics.get('top_k_feats')
            else: 
                params['top_k_feats'] = len(self.Xt.columns)


            new_metrics = self._cross_validate(params, xgb_seeds=xgb_seeds) 
            new_metrics = {k + ' seed_avg': v for k, v in new_metrics.items()}
            new_metrics = {k: metrics.get(k) for k in dfm.columns} | new_metrics
            new_metrics = {'rnd state samples': n_repeats * self.cv_params['n_folds']} | new_metrics
            new_metrics = pd.DataFrame({k: [v] for k, v in new_metrics.items()})

            df_ssc = pd.concat([df_ssc, new_metrics])  
            logloss = 'll7_tv seed_avg' if self.valid_params['vv_size'] == 0 else 'll7_vt seed_avg' 
            f1score = 'f1macro7_tv seed_avg' if self.valid_params['vv_size'] == 0 else 'f1macro7_vt seed_avg' 
            df_ssc = df_ssc.sort_values(by=[logloss, f1score], 
                                  ascending=[True, False]
            ) 
            store_csv(df_ssc, ssc_metrics_path)

            logger.info(f'Checked stability for {j+1}/{top_n} metrics')
            try:
                row = new_metrics.iloc[0]
                if "ll7_tv" in row and "ll7_tv seed_avg" in row:
                    logger.info(f'Difference logloss 7 valid_train: {float(row["ll7_tv"]) - float(row["ll7_tv seed_avg"]) }')
                if "ll2_tv" in row and "ll2_tv seed_avg" in row:
                    logger.info(f'Difference logloss 2 valid_train: {float(row["ll2_tv"]) - float(row["ll2_tv seed_avg"]) }')
            except Exception:
                pass

            # When running inside MLflow, log per-candidate deltas for visibility
            pipeline_manager = get_pipeline_manager()
            if pipeline_manager is not None:
                try:
                    row = new_metrics.iloc[0]
                    metrics_payload = {}
                    if "ll7_tv" in row and "ll7_tv seed_avg" in row:
                        metrics_payload["ll7_tv_delta"] = float(row["ll7_tv"]) - float(row["ll7_tv seed_avg"]) 
                    if "ll2_tv" in row and "ll2_tv seed_avg" in row:
                        metrics_payload["ll2_tv_delta"] = float(row["ll2_tv"]) - float(row["ll2_tv seed_avg"]) 
                    if metrics_payload:
                        pipeline_manager.log_metrics_safe(metrics_payload, prefix="stability_check", step=j)
                except Exception:
                    pass




        # Change back to original n_repeats
        CVparams(self).change_cv_param('n_repeats', og_n_repeats)



    def _cross_validate(self, hyperparams, xgb_seeds = None, mlflow_prefix="model"):
        params = hyperparams.copy() 

        Xt, yt = self.Xt, self.yt 
        all_folds = self.folds

        # if self.pipeline_stage >= 2:
        if hasattr(self, 'feature_params'): 
            # Let optuna decide the number of features
            top_k = params['top_k_feats'] 
            params.pop('top_k_feats') 
            df_ff = open_csv(self.feature_path)
            
            if top_k == 'All': 
                features = Xt.columns
                logger.info(f'Selected all {len(features)} features for CV.')
            elif self.feature_params['select_by'] == 'frequency': 
                top_k = int(top_k) if isinstance(top_k, str) else top_k
                features = df_ff[df_ff['count'] >= top_k]['feature'].values
                logger.info(f'Selected {len(features)} that were chosen {top_k} times or more.')
            else: # self.feature_params['select_by'] == 'index':
                top_k = int(top_k) if isinstance(top_k, str) else top_k
                features = df_ff.iloc[:int(top_k)]['feature'].values
                logger.info(f'Selected top {top_k} features for CV: {len(features)}')
            Xt = Xt[features] 
        Xvt, yvt = self.Xvt[Xt.columns], self.yvt
        Xvv = self.Xvv
        Xvv = Xvv[Xt.columns] if Xvv is not None else Xvv 
        yvv = self.yvv
        if hasattr(self, 'Xp'): 
            self.Xp = self.Xp[Xt.columns]
            self.Xvv = Xvv
            self.Xvt = Xvt

        n_repeats = self.cv_params['n_repeats'] 
        n_folds = self.cv_params['n_folds'] 
        
        tv_metrics_list = []
        vv_metrics_list = []
        vt_metrics_list = [] 

        xgb_seed = self.xgb_seed

        if not hasattr(self, 'Xp'): 
            log_params = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in hyperparams.items()
            }

            logger.info(f'Starting CV for {log_params}') 

        for j in range(n_repeats):
            for k, (train_idx, val_idx) in enumerate(all_folds[j]): 
                Xtt, Xtv = Xt.iloc[train_idx], Xt.iloc[val_idx]
                ytt, ytv = yt.iloc[train_idx], yt.iloc[val_idx]

    
                xgb_seed = xgb_seeds[j*n_folds + k] if xgb_seeds is not None else xgb_seed
                model = self._xgb_factory(params, xgb_seed=xgb_seed)

                model.fit(Xtt, ytt, eval_set=[(Xtv, ytv)], verbose=False)

                if hasattr(self, 'Xp'):
                    print('huh') 
                    return model 
                
                yt_proba = model.predict_proba(Xtv)
                metrics = self._get_metrics(ytv, yt_proba, params) 
                tv_metrics_list.append(metrics)

                yvt_proba = model.predict_proba(Xvt) 
                vt_metrics = self._get_metrics(yvt, yvt_proba, params) 
                vt_metrics_list.append(vt_metrics) 

                if Xvv is not None and yvv is not None: 
                    yvv_proba = model.predict_proba(Xvv) 
                    vv_metrics = self._get_metrics(yvv, yvv_proba, params)
                    vv_metrics_list.append(vv_metrics) 

                logger.info(f'Repeat {j+1}/{n_repeats}, fold {k+1}/{len(all_folds[j])} complete.')

        def _avg(metrics_seq, 
                 label 
        ): 
            if not metrics_seq:
                return {}
            keys = metrics_seq[0].keys()
            return {f'{k}_{label}': round(float(np.mean([m[k] for m in metrics_seq])),8) for k in keys}

        metrics = {
            "train_valid": _avg(tv_metrics_list, 'tv'),
            "valid_train": _avg(vt_metrics_list, 'vt'),
            "valid_valid": _avg(vv_metrics_list, 'vv')
        }

        logger.info(f'Accuracy on valid_train: {metrics["valid_train"]["acc2_vt"]}')

        # Validation set parameters CV metrics to MLflow when available
        pipeline_manager = get_pipeline_manager()
        if pipeline_manager is not None:
            try:
                pipeline_manager.log_metrics_safe(metrics.get("train_valid", {}), prefix="cv")
                pipeline_manager.log_metrics_safe(metrics.get("valid_train", {}), prefix="cv")
                pipeline_manager.log_metrics_safe(metrics.get("valid_valid", {}), prefix="cv")
                # Only log params if a prefix is provided (avoids conflicts in Optuna/predict)
                if mlflow_prefix:
                    pipeline_manager.log_params_safe(params, prefix=mlflow_prefix)
            except Exception:
                pass

        # Save to file 
        
        params['top_k_feats'] = 'All' if not hasattr(self,'feature_params') else top_k 
        params['n_features'] = f'All ({len(Xt.columns)})' if not hasattr(self,'feature_params') else len(features)
        valid_fraction = (len(self.Xvt) + self.valid_params['vv_size']) / (len(self.Xvt) + self.valid_params['vv_size'] + len(self.Xt)) 
        model_d = {'suffix': self.suffix, 'valid_size': valid_fraction}
        output_d = model_d | self.cv_params | self.valid_params | params 
        metrics_d = metrics['train_valid'] | metrics['valid_train'] | metrics['valid_valid'] 
        if xgb_seeds is not None: # and self.Xp is None: 
            # This is for the seed stability check pipeline
            return metrics_d
        output_d = output_d | metrics_d 
        output_d = {k: [v] for k, v in output_d.items()}
        dfm = creaopen_file(self.metrics_path)
        dfm = pd.concat([dfm, pd.DataFrame(output_d)])
        logloss = 'll7_tv' if self.valid_params['vv_size'] == 0 else 'll7_vt' 
        f1score = 'f1macro7_tv' if self.valid_params['vv_size'] == 0 else 'f1macro7_vt' 
        dfm = dfm.sort_values(by=[logloss, f1score], 
                              ascending=[True, False]
        )
        store_csv(dfm, self.metrics_path)

        time.sleep(2.5) 

        # For Optuna 
        if self.valid_params['vv_size'] > 0: 
            return metrics['valid_train']['ll7_vt']
        return metrics['train_valid']['ll7_tv'] 
        
    def _objective(self, trial): 
        if not hasattr(self, 'hyper_params'): 
            CVparams(self).set_hyper_params() 
        hyper_params = self.hyper_params.copy()
        param_ranges = {} 

        #if self.pipeline_stage == 2: 
        if hasattr(self, 'feature_params'): 
            #CVparams(self).set_feature_params() 
            hyper_params['top_k_feats'] = self.feature_params['feature_range']
            logger.info(f'Found attribute "feature_params". Assuming re-training with reduced features.') 

            top_n_trials = 10
            dfm = open_csv(self.metrics_path).iloc[:top_n_trials]
            for k in hyper_params.keys():
                if k == 'top_k_feats': 
                    # At this point only full feature model has been considered
                    # and top_k features are selected manually. 
                    continue 

                min_prm = 0.8*dfm[k].min() 
                max_prm = 1.2*dfm[k].max()
                max_prm = min(max_prm, 1) if k in ['subsample', 'colsample_bytree'] else max_prm

                int_params = ['n_estimators','top_k_feats','max_depth'] 
                #float_params = [k for k in hyper_params.keys() if k not in int_params] 
                min_prm = round(min_prm) if k in int_params else float(min_prm) 
                max_prm = round(max_prm) if k in int_params else float(max_prm) 
                max_prm = 

                if min_prm != max_prm: 
                    hyper_params[k] = (min_prm, max_prm)
                else: 
                    hyper_params[k] = min_prm 
                    
        for k, v in hyper_params.items(): 
            if isinstance(v, float) or isinstance(v, int): 
                param_ranges[k] = v 
            elif isinstance(v, (list, tuple)): 
                if isinstance(v[0], int) and isinstance(v[1], int): 
                    param_ranges[k] = trial.suggest_int(k, v[0], v[1]) 
                else: 
                    param_ranges[k] = trial.suggest_float(k, v[0], v[1]) 

        # Do not log params from _cross_validate here; per-trial params already logged in callback
        return float(self._cross_validate(param_ranges, mlflow_prefix=None))

    def _get_metrics(
            self, 
            y_true7, 
            y_proba7,
            params
    ):
        # Ensure shapes align: y_proba7 should be (n_samples, 7)
        y_true7 = np.asarray(y_true7).ravel()
        y_proba7 = np.asarray(y_proba7)
        if y_proba7.ndim == 2:
            if y_proba7.shape[0] != y_true7.shape[0] and y_proba7.shape[1] == y_true7.shape[0]:
                # Some upstream code may have provided a transposed matrix (7, n_samples)
                y_proba7 = y_proba7.T
                logger.debug(f"Transposed y_proba7 to align shapes: y_true7={y_true7.shape}, y_proba7={y_proba7.shape}")
        elif y_proba7.ndim == 1:
            # Single-sample case or malformed input; try to coerce
            if y_proba7.shape[0] == 7:
                y_proba7 = y_proba7.reshape(1, -1)
            else:
                raise ValueError(f"y_proba7 has unexpected shape {y_proba7.shape}; expected (*, 7)")

        y_pred7 = np.argmax(y_proba7, axis=1)
        f1_7 = f1_score(y_true7, y_pred7, average='macro') 
        ll_7 = log_loss(y_true7, y_proba7, labels = range(7)) 
        acc_7 = accuracy_score(y_true7, y_pred7)
        
        # Check for Draws 
        draw_idx = np.argmax(y_proba7, axis=1) == 3 
        if np.any(y_true7[draw_idx] == 3): 
            logger.info("Holy darn it correctly predicted a Draw")

        y_proba7 = y_proba7[y_true7 != 3] 
        y_true7 = y_true7[y_true7 != 3] 

        y_true2 = pd.Series(y_true7).map({0:0,1:0,2:0,4:1,5:1,6:1}).to_numpy()

        win = y_proba7[:,:3].sum(axis=1) / (1 - y_proba7[:, 3]) 
        los = y_proba7[:,4:].sum(axis=1) / (1 - y_proba7[:, 3]) 
        y_proba2 = np.column_stack((win, los))

        y_pred2 = np.argmax(y_proba2, axis=1)
        f1_2 = f1_score(y_true2, y_pred2, average='macro') 
        ll_2 = log_loss(y_true2, y_proba2, labels = range(2)) 
        acc_2 = accuracy_score(y_true2, y_pred2)

        return {"ll7": ll_7, "f1macro7": f1_7, "acc7": acc_7, 
                  "ll2": ll_2, "f1macro2": f1_2, "acc2": acc_2}

    def _xgb_factory(self, params, xgb_seed): 
        base = dict(
            objective="multi:softprob",
            num_class=7,
            n_jobs=-2,
            early_stopping_rounds=25,
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
            tree_method="hist",
            random_state=xgb_seed
        )
        base.update(params)
        return XGBClassifier(**base)



if __name__ == '__main__':

    suffix = 'symm'
    CV = CVMain('symm') 
    
    # Optionally, change default values 
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
    hyper_params = { 
        "max_depth": 5,
        "learning_rate": (0.02, 0.025),
        "n_estimators": (50,60),
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
    CV.optimize(n_trials = 1) 

    # Feature selection 
    #CV.change_cv_param('n_repeats', 1) 
    #CV.select_features() 

    # Training again 
    #CV.change_cv_param('n_repeats', 1)
    #CV.optimize(n_trials = 1) 
        
    # Predicting and calibrating
    # CV.change_cv_param('n_repeats', 1)
    # CV.change_cv_param('n_folds', 2) 
    # CV.predict()

    # Finito


