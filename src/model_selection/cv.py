import pandas as pd 
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, log_loss, accuracy_score
from loguru import logger
from typing import Callable, Dict, List, Tuple, Optional
from xgboost import XGBClassifier
import time 
import optuna

from src.utils.general import open_csv, store_csv, get_data_path, creaopen_file
import mlflow
from src.utils.mlflow_utils import (
    setup_mlflow,
    log_params_dict,
    log_metrics_dict,
    log_artifact_safe,
    start_run_with_tags,
    pick_experiment_by_suffix,
)
from src.model_selection.trainvalpred import TrainValPred
from src.data_processing.cleaned_data import CleanedFights 

class CrossValidation: 
    """
    Initializes a K-repeated cross validation. 

    Utilities:
    - Optimize hyperparameters   
    - Most important features selector
    - Making predictions on unseen data. 

    CrossValidation is initiated by a 3 sets of parameters. 
    data_params dict[str, ...]
        suffix str: 
            the suffix of the dataset you have created using trainvalpred. 
            So: 'svd', 'symm' or ""
        k_selected int: 
            after feature selection, set this value to desired number of 
            most important columns. Set to None when only optimizing 
            hyperparameters.  
        skip_params bool:
            whether you wanna skip parameter combinations you already 
            tried
        vv_size float [0,1): 
            In stead of minimizing on the training data, you can 
            minimize on a portion of the validation data ('valid_train')
            using the remainder as your actual validation data ('valid_valid'). 
            This way, the model trains on a more relevant part of the data, 
            namely, the most recent fights.
            If you choose to minize a part of the validation data,
            set 1 > vv_size > 0. 
            I don't know. So far I am noticing a huge difference in 
            the metrics of valid_valid and valid_train. 
        vv_random_split bool: 
            When you choose vv_size > 0, you can either split 'valid_train' 
            and 'valid_valid' randomnly or based on date. When choosing the 
            former, set vv_random_split to True. 
        vv_seed int: 
            When vv_random_split = True, set your seed.
        3class_feature_selection bool: 
            Whether to select features based on a 3-class model or a 7-class model.
            Not recommended. Training is for now always done on 7 classes so....

            
    cv_params dict[str, int]:
        n_repeats:
            The number of times you wanna repeat the cross validation
            (for n_repeats different folds)
        n_folds: 
            The number of folds. So the model gets evaluated
            n_repeats * n_folds times. 
        fold_seed: 
            Random seed for folding the training data. 
            Note, when n_repeats > 1, the seed get extended
            as fold_seed, fold_seed + 1, fold_seed + 2, ... 

    xgb_params: 
        xgboost model parameters.
             
    """

    def __init__(self, data_params = {
                           'suffix': 'svd', 
                           'k_selected': None,
                           'skip_params': False, 
                           'vv_seed': 69, 
                           'vv_random_split': False,
                           'vv_size': 0.5,
                           'save_as_n_classes': 3,
                           'measure_calibration': False,
                           '3class_feature_selection': False
                       }, 
                       cv_params = {
                           'fold_seed': 42, 
                           'n_folds' : 5, 
                           'n_repeats': 1,
                           'n_trials': 50
                       },
                       xgb_params = {
                           'n_estimators': 500, 
                           'max_depth': 3
                        }
    ): 

        
        self.data_params = data_params 
        # For Optuna-based optimization, we don't precompute a grid.
        self.param_grid = None 
        self.cv_params = cv_params 
        self.xgb_params = xgb_params 
        # Keep a cached, ordered list of features (for dynamic k selection)
        self._feat_order: Optional[List[str]] = None

    def run_cv(self, 
                 predict=False, 
                 select_features=False,
                 random_grid_search=False): 
        """
        Perform a repeated K-fold Cross Validation over a grid of hyperparameters.

        Can be used for 
        - Hyperparameter optimization  (predict = False, select_features = False) 
        - Most important features selection (predict = False, select_features=True) 
        - Making predictions on unseen data (predict = True, select_features = False)  

        output: 
            output/metrics/param_optimization_{suffix}.csv
                CSV file with metrics for each hyperparamter
                combination. 
            output/feature_selection/feature_frequency_{suffix}.csv
                File with mean importances and frequencies of features 
                counted in each fold for each repeat. 
            output/predictions/pred_{suffix}.csv
                Final predictions on upcoming fights. 
        """

        cv_params = self.cv_params.copy()   
        data_params = self.data_params.copy()
        param_grid = self.param_grid

        fold_seed = cv_params['fold_seed'] 
        n_folds = cv_params['n_folds'] 
        n_repeats = cv_params['n_repeats'] 
        suffix = data_params['suffix'] 

        dft, folds = TrainValPred().get_folds(suffix, n_repeats = n_repeats, 
                                            n_folds = n_folds, first_seed = fold_seed)
        n_features = len(dft.columns)-1 

        # Prepare a stable feature order for dynamic k selection; do not subselect here
        self._feat_order = self._get_feature_order(
            list(dft.drop(columns=['result']).columns), suffix
        )
        # Default k when not optimizing it explicitly
        if data_params.get('k_selected') is None:
            data_params['k_selected'] = n_features

        dfv = TrainValPred().open('valid', suffix)

        Xt, yt = dft.drop(columns=['result']), dft['result']
        Xv, yv = dfv[Xt.columns], dfv['result']
        Xvt, yvt = Xv.copy(), yv.copy() 
        Xvv, yvv = None, None 

        if predict:
            # MLflow setup per suffix
            pick_experiment_by_suffix(suffix)
            # Checking for inconsistent arguments  
            if any([isinstance(self.xgb_params[k], (list, tuple)) and len(self.xgb_params[k]) == 2 for k in self.xgb_params.keys()]): 
                logger.warning('Provided hyperparam grid. Sure you wanna make predictions?')    
            if select_features: 
                logger.warning("'predict' and 'select_features' both set to True. Skipping feature selection.")  
            # Opening the data to predict file 
            dfp = TrainValPred().open('pred', suffix)
            Xp = dfp[Xt.columns]
            with start_run_with_tags(stage="predict", suffix=suffix, run_type="predict"):
                log_params_dict({"mode": "predict", "suffix": suffix})
                log_params_dict(self.data_params, prefix="data")
                log_params_dict(self.cv_params, prefix="cv")
                # Resolve params and run
                best_override = self._get_best_xgb_params()
                if best_override is None:
                    logger.warning('No best params file found; falling back to provided xgb_params.')
                    best_override = self.xgb_params
                fixed_params = self._resolve_fixed_params(best_override)

                if 'k_selected' not in fixed_params:
                    fixed_params['k_selected'] = len(self._feat_order) if self._feat_order is not None else Xt.shape[1]
                    logger.info(f"No k_selected in best params; using all features: k_selected={fixed_params['k_selected']}")

                new_metric = self._train_xgb(Xt, yt, fixed_params, folds, Xvt, yvt, Xvv, yvv, Xp)
                log_params_dict(fixed_params, prefix="xgb_best")
                log_metrics_dict(new_metric.get("train", {}), prefix="train")
                log_metrics_dict(new_metric.get("vtrain", {}), prefix="vtrain")
                log_metrics_dict(new_metric.get("vvalid", {}), prefix="vvalid")
                preds_file = get_data_path('output') / 'predictions' / f"pred_{suffix}_predictions.csv"
                log_artifact_safe(preds_file)
                logger.info(f'CV metrics:')
                logger.info(new_metric["vtrain"])            
                return 0
        elif select_features:
            # Beginning feature selection  
            pick_experiment_by_suffix(suffix)
            with start_run_with_tags(stage="feature_selection", suffix=suffix, run_type="train"):
                log_params_dict({"mode": "feature_selection", "suffix": suffix})
                log_params_dict(self.data_params, prefix="data")
                log_params_dict(self.cv_params, prefix="cv")
                out_fs = self._feature_selection(Xt, yt, folds) 
                # Log artifact and meta
                ff_file = get_data_path('output') / 'feature_selection' / f"feature_frequency_{suffix}.csv"
                log_artifact_safe(ff_file)
                if isinstance(out_fs, dict):
                    log_params_dict({"total_models": out_fs.get("total_models", 0)}, prefix="fs")
            return 0  
        else: 
            # You wanna optimize hyperparameters 
            Xp = None
            vsize = len(Xv) / (len(Xv) + len(Xt)) 
            out1 = data_params | cv_params | {'n_features': n_features} 
            # Safe pops
            for _k in ['skip_params','save_as_n_classes','measure_calibration','3class_feature_selection','n_trials']:
                if _k in out1:
                    out1.pop(_k)

            opt_path = get_data_path('output')/'metrics'/f'param_optimization_{suffix}.csv'
            df_opt = creaopen_file(opt_path)
            # skip_params is ignored in Optuna mode (we keep top-50 file up to date)

            vv_size = self.data_params['vv_size'] 
            random_split = self.data_params['vv_random_split'] 
            if vv_size > 0:  
                # In stead of minimizing on training data you can minimize on 
                # a part of the validation data.  
                vv_seed = self.data_params['vv_seed'] 

                vnames_path = get_data_path('interim') / 'valid_names.csv' 
                f_ids_v = open_csv(vnames_path)['temp_f_id'] 
                Xv = pd.concat([Xv, f_ids_v], axis=1) 

                if random_split: 
                    rng = np.random.default_rng(seed = vv_seed) 
                    f_ids_vv = rng.choice(f_ids_v.unique(), 
                                        size = int(vv_size*len(f_ids_v.unique())), 
                                        replace=False) 
                else:
                    f_ids_vv = f_ids_v[f_ids_v > (1-vv_size)*f_ids_v.max()] 
                    

                Xvv = Xv[Xv['temp_f_id'].isin(f_ids_vv)]
                yvv = yv.iloc[Xvv.index] 

                Xvt = Xv[~Xv['temp_f_id'].isin(f_ids_vv)]
                yvt = yv.iloc[Xvt.index] 

                #self.data_params['vv_index'] = Xvv.index 

                Xvv = Xvv.drop(columns=['temp_f_id']).reset_index(drop=True)  
                Xvt = Xvt.drop(columns=['temp_f_id']).reset_index(drop=True) 


                logger.info(f'Length valid_valid: {len(yvv)}')
            logger.info(f'Length valid_train: {len(yvt)}') 
            logger.info('Starting iteration over parameter grid.') 
        # Modes
        if select_features:
            return 0

        # Optimization with Optuna (random_grid_search is ignored)
        minimize_target = 'll7_vt' if self.data_params['vv_size'] > 0 else 'll7_train'
        # Target accuracy to log per parameter combination
        acc_target_key = 'acc3_vv' if self.data_params['vv_size'] > 0 else 'acc3_vt'

        # Combine XGB params with optional k_selected search space
        search_space = self.xgb_params.copy()
        ks = self.data_params.get('k_selected')
        if isinstance(ks, (list, tuple)) and len(ks) == 2:
            # interpret as integer range
            search_space['k_selected'] = (int(ks[0]), int(ks[1]))
        elif isinstance(ks, int):
            search_space['k_selected'] = int(ks)

        pick_experiment_by_suffix(suffix)
        with start_run_with_tags(stage="cv_optuna", suffix=suffix, run_type="train"):
            # Log context on parent run
            log_params_dict({"mode": "optimize", "suffix": suffix})
            log_params_dict(self.data_params, prefix="data")
            log_params_dict(self.cv_params, prefix="cv")
            # Log search space summary (flatten fixed or ranges)
            search_log = {}
            for k, v in search_space.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    search_log[k] = f"{v[0]}..{v[1]}"
                else:
                    search_log[k] = v
            log_params_dict(search_log, prefix="xgb_space")

            logger.info('Starting Optuna study.')
            study = optuna.create_study(direction='minimize')

            def _objective(trial):
                params = self._suggest_params(trial, search_space)
                with start_run_with_tags(stage="cv_trial", suffix=suffix, run_type="train", nested=True, run_name_extra=f"trial_{trial.number}", tags={"trial_number": str(trial.number)}):
                    log_params_dict(params, prefix="xgb")
                    metrics = self._train_xgb(Xt, yt, params, folds, Xvt, yvt, Xvv, yvv, None)
                    log_metrics_dict(metrics.get('train', {}), prefix='train', step=trial.number)
                    log_metrics_dict(metrics.get('vtrain', {}), prefix='vtrain', step=trial.number)
                    log_metrics_dict(metrics.get('vvalid', {}), prefix='vvalid', step=trial.number)

                    # update top-50 table and log as artifact
                    row = out1 | params | metrics['train'] | metrics['vtrain'] | metrics['vvalid']
                    df_row = pd.DataFrame({k: [row[k]] for k in row.keys()})
                    nonlocal df_opt
                    df_opt = pd.concat([df_opt, df_row], ignore_index=True)
                    sort_by = minimize_target
                    if sort_by in df_opt.columns:
                        df_opt = df_opt.sort_values(by=[sort_by], ascending=True, kind='mergesort').head(50)
                    store_csv(df_opt, opt_path)
                    log_artifact_safe(opt_path)

                    # Return objective
                    return float(metrics['vtrain'][minimize_target] if minimize_target == 'll7_vt' else metrics['train'][minimize_target])

            study.optimize(_objective, n_trials=self.cv_params.get('n_trials', 50))
            logger.info(f'Best trial value: {study.best_value}')
            logger.info(f'Best params: {study.best_trial.params}')
            mlflow.log_metric("best_value", float(study.best_value))
            mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})

            # Log vt accuracy for the best params
            try:
                best_params = study.best_trial.params
                best_metrics = self._train_xgb(Xt, yt, best_params, folds, Xvt, yvt, Xvv, yvv, None)
                acc7 = best_metrics.get('vtrain', {}).get('acc7_vt')
                acc3 = best_metrics.get('vtrain', {}).get('acc3_vt')
                if acc7 is not None:
                    logger.info(f'Best vt acc7 (at min ll7): {acc7}')
                if acc3 is not None:
                    logger.info(f'Best vt acc3 (at min ll7): {acc3}')
                log_metrics_dict(best_metrics.get('vtrain', {}), prefix='best_vtrain')
                log_metrics_dict(best_metrics.get('vvalid', {}), prefix='best_vvalid')
            except Exception as e:
                logger.warning(f'Could not compute vt accuracy for best params: {e}')
        return 0

             
    def _train_xgb(self, 
            Xt: pd.DataFrame,
            yt: pd.Series,
            params: dict[str, float],
            all_folds: list[list[int]],  
            Xvt: Optional[pd.DataFrame] = None,
            yvt: Optional[pd.Series] = None,
            Xvv: Optional[pd.DataFrame] = None,
            yvv: Optional[pd.Series] = None,
            Xp: Optional[pd.DataFrame] = None
    ): 
        """
        Performs n_repeats of an n_folds CV for 1 set of hyperparameter combos and 
        validates the model in each fold on a validation set Xvt, and optionally, 
        a validation set Xvv. One can then minimize on Xvt in stead of Xt, since 
        with current setting Xv contains most recent fights. 
        """
        log_params = params.copy() 
        log_params = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in log_params.items()
        }
        logger.info(f'Doing CV for hyperparameters {log_params}') 
        
        # Select top-k features here (supports Optuna varying k_selected)
        feat_order = self._feat_order if self._feat_order is not None else list(Xt.columns)
        k = params.get('k_selected', self.data_params.get('k_selected', len(feat_order)))
        try:
            k = int(k)
        except Exception:
            k = len(feat_order)
        k = max(1, min(k, len(feat_order)))
        selected = [c for c in feat_order if c in Xt.columns][:k]

        # Subset all matrices consistently
        Xt = Xt[selected]
        if Xvt is not None:
            Xvt = Xvt[[c for c in selected if c in Xvt.columns]]
        if Xvv is not None:
            Xvv = Xvv[[c for c in selected if c in Xvv.columns]]
        if Xp is not None:
            Xp = Xp[[c for c in selected if c in Xp.columns]]

        logger.info(f'Using k_selected={len(selected)} features.')

        n_repeats = self.cv_params['n_repeats']
        n_folds = self.cv_params['n_folds'] 

        if Xp is not None:  
            measure_cal = self.data_params.get('measure_calibration', False) 
            final_n_classes = self.data_params.get('save_as_n_classes', 3) 
            p_probas = np.empty((len(Xp)*final_n_classes, n_repeats*n_folds))
            """
            if Xv is not None: 
                # No need to split anymore
                vv_index = self.data_params['vv_index'] 
                Xv = pd.DataFrame(index = range(len(Xvt) + len(Xvv))) 
                Xv.loc[~Xv.index.isin(vv_index), Xvt.columns] = Xvt.values 
                Xv.loc[vv_index, Xvv.columns] = Xvv.values

                yv = pd.Series(index=range(len(Xv))) 
                yv.loc[~yv.index.isin(vv_index)] = yvt.values 
                yv.loc[vv_index] = Xvv.values
            """
            if measure_cal:
                v_probas = np.empty((len(Xvt)*final_n_classes, n_repeats*n_folds))


        # 
        model = self._xgb_factory(params)

        # metrics 
        metrics_list = []
        vv_metrics_list = []
        vt_metrics_list = [] 
        p_metrics_list = [] 



        logger.info('Starting CV.') 
        for j in range(n_repeats):
            for k, (train_idx, val_idx) in enumerate(all_folds[j]): 
                Xtt, Xtv = Xt.iloc[train_idx], Xt.iloc[val_idx]
                ytt, ytv = yt.iloc[train_idx], yt.iloc[val_idx]

                model.fit(Xtt, ytt, eval_set=[(Xtv, ytv)], verbose=False)

                yt_proba = model.predict_proba(Xtv)
                metrics = self._get_metrics(ytv, yt_proba) 
                metrics_list.append(metrics)

                if Xvt is not None and yvt is not None:

                    yvt_proba = model.predict_proba(Xvt) 
                    vt_metrics = self._get_metrics(yvt, yvt_proba)
                    vt_metrics_list.append(vt_metrics) 

                    if Xvv is not None and yvv is not None: 
                        yvv_proba = model.predict_proba(Xvv) 
                        vv_metrics = self._get_metrics(yvv, yvv_proba)
                        vv_metrics_list.append(vv_metrics) 

                if Xp is not None:
                    col_idx = j*n_folds + k 
                    yp_proba = model.predict_proba(Xp)
                    yp_proba = self._3class_probas(yp_proba) if final_n_classes == 3 else yp_proba 
                    yp_proba = self._2class_probas(yp_proba) if final_n_classes == 2 else yp_proba 
                    p_probas[:,col_idx] =  yp_proba.reshape(-1)

                    if measure_cal: 
                        yvt_proba = self._3class_probas(yvt_proba) if final_n_classes == 3 else yvt_proba 
                        yvt_proba = self._2class_probas(yvt_proba) if final_n_classes == 2 else yvt_proba 
                        v_probas[:,col_idx] = yvt_proba.reshape(-1) 


                logger.info(f'Repeat {j+1}/{n_repeats}, fold {k+1}/{len(all_folds[j])} complete.')
        logger.info('CV complete. Processing metrics.') 

        if Xp is not None: 
            self._save_probas(p_probas, 'pred') 
            if measure_cal: 
                self._save_probas(v_probas, 'valid') 


        def _avg(metrics_seq: List[Dict[str, float]],
                 label: str
        ) -> Dict[str, float]:
            if not metrics_seq:
                return {}
            keys = metrics_seq[0].keys()
            return {f'{k}_{label}': round(float(np.mean([m[k] for m in metrics_seq])),5) for k in keys}

        return {
            "train": _avg(metrics_list, 'train'),
            "vtrain": _avg(vt_metrics_list, 'vt'),
            "vvalid": _avg(vv_metrics_list, 'vv')
        }

    def _save_probas(self, probas, save_for: str = 'pred'): 

        name_path = get_data_path('interim') / f'{save_for}_names.csv' 
        df_names = open_csv(name_path)

        n_classes = self.data_params['save_as_n_classes'] 
        if n_classes == 7: 
            cols = ['Win KO','Win Sub','Win Dec','Draw','Loss Dec','Loss Sub','Loss KO'] 
        elif n_classes == 3:  
            cols = ['Win', 'Draw', 'Loss']
        if n_classes == 3:  
            cols = ['Win', 'Loss']
        else: 
            logger.warning('Please set "save_as_n_classes to 2, 3 or 7. No probabilities stored.') 
            return 0

        sample_cols = [str(i) for i in range(probas.shape[-1])] 
        df_prob = pd.DataFrame(columns=sample_cols, data=probas) 
        df_names = df_names.merge(pd.DataFrame({'outcome':cols}), how='cross') 
        df_prob = pd.concat([df_names, df_prob], axis=1)


        df_prob1 = df_prob[df_prob['upp_low']==0].copy().sort_values(by=['temp_f_id','outcome'], 
                                                                     ascending=[True, False]
        ) 
        df_prob2 = df_prob[df_prob['upp_low']==1].copy().sort_values(by=['temp_f_id','outcome']) 

        df_prob1.reset_index(drop=True,inplace=True) 
        df_prob2.reset_index(drop=True,inplace=True)  

        probas = np.hstack((df_prob1[sample_cols].values, 
                            df_prob2[sample_cols].values)) 

        avg = np.mean(probas, axis=1)
        std = np.std(probas, axis=1) 
        perc5 = np.percentile(probas, q = 5, axis=1) 
        perc95 = np.percentile(probas, q = 95, axis=1) 
        mean2stdp = avg + 2*std 
        mean2stdm = avg - 2*std
        minp = np.min(probas,axis=1) 
        maxp = np.max(probas,axis=1) 
        
        df_preds = pd.DataFrame(columns = ['avg','std','perc5','perc95','mean2stdp','mean2stdm','min','max'],
                                data = np.array([avg, std, perc5, perc95, mean2stdp, mean2stdm, minp, maxp]).T,
        ) 


        df_preds = pd.concat([df_prob1[['name f1', 'name f2','outcome']], df_preds], axis = 1) 

        subfolder = 'predictions'
        file_name = f'{save_for}_{self.data_params["suffix"]}_predictions.csv' 
        preds_path = get_data_path('output') / subfolder / file_name 
        store_csv(df_preds, preds_path) 

         
    def _get_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Compute standard metrics for 7-class problem plus 3-grouped (win/draw/lose)."""
        y_pred = np.argmax(y_proba, axis=1)
        f1 = f1_score(y_true, y_pred, average="macro")
        ll = log_loss(y_true, y_proba, labels = range(7)) 
        acc7 = accuracy_score(y_true, y_pred)

        y3 = self._3class_probas(y_proba) 
        y_true3 = pd.Series(y_true).map({0:0,1:0,2:0,3:1,4:2,5:2,6:2}).to_numpy()

        y3_pred = np.argmax(y3, axis=1) 
        y3_draw = y3_pred[y3_pred == 1] 
        if y3_draw.size > 0: 
            logger.warning('Model predicts draws. Exclude from metrics?') 

        acc3 = accuracy_score(y_true3, np.argmax(y3, axis=1))
        f1_3 = f1_score(y_true3, np.argmax(y3, axis=1), average="macro")
        ll_3 = log_loss(y_true3, y3, labels=range(3)) 

        return {"ll7": ll, "f1_7": f1, "acc7": acc7, 
                "ll3": ll_3, "f1_3": f1_3, "acc3": acc3}

    def _feature_selection(self, Xt, yt, all_folds
    ) -> Dict:
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
        feat_names = np.array(Xt.columns)
        n_features = len(feat_names)
        n_repeats = self.cv_params['n_repeats']
        xgb_params = self._resolve_fixed_params(self.xgb_params)
        suffix = self.data_params['suffix']

        yt = yt.map({0:0, 1:0, 2:0, 3:1, 4:2, 5:2, 6:2}) if self.data_params['3class_feature_selection']  else yt

        # Prefer best params from CSV when doing feature selection; fallback to provided xgb_params
        best_override = self._get_best_xgb_params()
        xgb_params = self._resolve_fixed_params(best_override if best_override is not None else self.xgb_params)

        # Model factory and base params (centered on your prior choices)
        model = self._xgb_factory(xgb_params)  

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
        out_dir = get_data_path("output") / 'feature_selection'
        store_csv(freq_df, out_dir / f"feature_frequency_{suffix}.csv")

        return {
            "freq_df": freq_df,
            "total_models": total_models,
        }

    def _resolve_fixed_params(self, params_in: Dict[str, float | int | List | Tuple]) -> Dict[str, float | int]:
        out = {}
        for k, v in params_in.items():
            if isinstance(v, (list, tuple)):
                if len(v) == 2:
                    val = v[0]
                elif len(v) == 1:
                    val = v[0]
                else:
                    # fallback
                    val = v[0]
            else:
                val = v
            # cast to int for int params
            if k in ['max_depth','n_estimators','min_child_weight','k_selected']:
                val = int(val)
            else:
                val = float(val) if isinstance(val, (int, float)) else val
            out[k] = val
        return out

    def _get_best_xgb_params(self) -> Optional[Dict[str, float | int | List | Tuple]]:
        """
        Load best XGB params from output/metrics/param_optimization_{suffix}.csv.
        Returns a dict merged onto self.xgb_params, also including k_selected when present.
        """
        suffix = self.data_params.get('suffix', '')
        opt_path = get_data_path('output') / 'metrics' / f'param_optimization_{suffix}.csv'
        try:
            df = open_csv(opt_path)
            if df is None or df.empty:
                return None
            # Prefer minimizing columns in this order
            for sort_col in ['ll7_vt', 'll7_train', 'll3_vt', 'll3_train']:
                if sort_col in df.columns:
                    df = df.sort_values(by=[sort_col], ascending=True, kind='mergesort')
                    break
            row = df.iloc[0].to_dict()
            # Keep known hyperparameters; also include k_selected if present
            picked = {k: row[k] for k in self.xgb_params.keys() if k in row and pd.notna(row[k])}
            if 'k_selected' in row and pd.notna(row['k_selected']):
                picked['k_selected'] = row['k_selected']
            if not picked:
                return None
            merged = self.xgb_params.copy()
            merged.update(picked)
            return merged
        except Exception as e:
            logger.warning(f'Could not read best params from {opt_path}: {e}')
            return None

    def _suggest_params(self, trial, space: Dict[str, float | int | List | Tuple]) -> Dict[str, float | int]:
        params = {}
        for k, v in space.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                lo, hi = v
                if k in ['max_depth','n_estimators','min_child_weight','k_selected']:
                    params[k] = trial.suggest_int(k, int(lo), int(hi))
                else:
                    params[k] = trial.suggest_float(k, float(lo), float(hi))
            else:
                # fixed value
                params[k] = int(v) if k in ['max_depth','n_estimators','min_child_weight','k_selected'] else float(v)
        return params

    def _xgb_factory(self, params, xgb_seed=69): 
        base = dict(
            objective="multi:softprob",
            num_class=7, 
            early_stopping_rounds=25,
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=xgb_seed
        )
        base.update(params)
        return XGBClassifier(**base)

    def _3class_probas(self, y_proba): 
        win = y_proba[:, :3].sum(axis=1)
        draw = y_proba[:, 3]
        lose = y_proba[:, 4:].sum(axis=1)
        y3 = np.vstack([win, draw, lose]).T
        return y3 

    def _2class_probas(self, y_proba): 
        # collapse 7-class probs to 2 classes: Win vs Lose (split draw evenly)
        win = y_proba[:, :3].sum(axis=1) + y_proba[:, 3] / 2.0
        lose = y_proba[:, 4:].sum(axis=1) + y_proba[:, 3] / 2.0
        y2 = np.vstack([win, lose]).T
        return y2 

    def _get_feature_order(self, feat_cols: List[str], suffix: str) -> List[str]:
        """
        Load ordered features from feature_frequency_{suffix}.csv if available.
        Fallback to given feat_cols order. Guarantees output is a permutation/subset of feat_cols.
        """
        try:
            cc_path = f'feature_frequency_{suffix}.csv' if suffix else 'feature_frequency.csv'
            df_cc = open_csv(get_data_path('output') / 'feature_selection' / cc_path)
            if df_cc is None or df_cc.empty or 'feature' not in df_cc.columns:
                return feat_cols
            ranked = [c for c in df_cc['feature'].astype(str).tolist() if c in feat_cols]
            # append any features not in the ranking to preserve availability
            remaining = [c for c in feat_cols if c not in ranked]
            return ranked + remaining
        except Exception as e:
            logger.warning(f'Falling back to original feature order; could not load ranking: {e}')
            return feat_cols

if '__main__' == __name__: 
    xgb_params = { 
        "max_depth": (5, 5),
        "learning_rate": (0.02, 0.025),
        "n_estimators": (500,600),
        "min_child_weight": (0, 40),
        "gamma": (0, 2.5),
        "subsample": (0.7, 0.85),
        "colsample_bytree": (0.95, 1.0),
        # Optional regularization
        "reg_alpha": 0.0,
        "reg_lambda": 1.0
    }
    data_params = {
        'suffix': 'symm', 
        'k_selected': None,
        'skip_params': True, 
        'vv_seed': 6, 
        'vv_size': 0,
        'vv_random_split': False,   
        'save_as_n_classes': 3, 
        'measure_calibration': False, 
        '3class_feature_selection': False # Fuck that keep False 
    } 
    cv_params = {
        'fold_seed': 30, 
        'n_folds' : 5, 
        'n_repeats': 1,
        'n_trials': 50  
    }
    
    for k in [50,100,150,200]: 
        data_params['k_selected'] = k 
        CV = CrossValidation(data_params, cv_params, xgb_params)
        CV.run_cv(select_features=False, predict=False)







