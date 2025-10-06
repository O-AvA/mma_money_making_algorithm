import pandas as pd 
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, log_loss, accuracy_score
from loguru import logger
from typing import Callable, Dict, List, Tuple, Optional
from xgboost import XGBClassifier

from src.utils.general import open_csv, store_csv, get_data_path, creaopen_file
from src.model_selection.trainvalpred import TrainValPred
from src.data_processing.cleaned_data import CleanedFights 

class CrossValidation:



    def __init__(self, data_params = {
                           'suffix': 'svd', 
                           'k_selected': None,
                           'skip_params': False, 
                           'vv_seed': 69, 
                           'vv_size': 0.5
                       }, 
                       cv_params = {
                           'fold_seed': 42, 
                           'n_folds' : 5, 
                           'n_repeats': 1
                       },
                       xgb_params = {
                           'n_estimators': 500, 
                           'max_depth': 3
                        }
    ): 

        
        self.data_params = data_params 
        self.param_grid = ParameterGrid(xgb_params) 
        self.cv_params = cv_params 
        self.xgb_params = xgb_params 


    def run_cv(self, 
                 predict=False, 
                 select_features=False,
                 random_grid_search=False): 
        """
        Perform a repeated K-fold Cross Validation over a grid of hyperparameters.
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
        logger.warning('Random empty E1, E2 columns in dft. What are there source?') 

        # Select K mostly chosen features in feature_selection 
        k_selected = data_params['k_selected'] 
        if k_selected is not None: 
            cc_path = f'feature_frequency_{suffix}.csv' if suffix else 'feature_frequency.csv'
            df_cc = open_csv(get_data_path('output') / 'feature_selection' / cc_path) 
            logger.warning('Please fix training on temp_f_id') 

            selected_cols = df_cc.iloc[:k_selected]['feature'].values.tolist() 
            dft = dft[selected_cols + ['result']]
        else: 
            k_selected = np.nan 

        dfv = TrainValPred().open('valid', suffix)

        Xt, yt = dft.drop(columns=['result']), dft['result']
        Xv, yv = dfv[Xt.columns], dfv['result']

        if predict:
            # Checking for inconsistent arguments  
            if any([len(xgb_params[k]) > 1 for k in xgb_params.keys()]): 
                logger.warning('Provided hyperparam grid. Sure you wanna make predictions?')    
            if select_features: 
                logger.warning("'predict' and 'select_features' both set to True. Skipping feature selection.")  
            # Opening the data to predict file 
            dfp = TrainValPred().open('pred', suffix)
            Xp = dfp[Xt.columns]
            Xvt, yvt = Xv, yv 
            Xvv, yvv = None, None
        elif select_features:
           # Beginning feature selection  
           self._feature_selection(Xt, yt, folds) 
           return 0  
        else: 
            # You wanna optimize hyperparameters 
            Xp = None
            vsize = len(Xv) / (len(Xv) + len(Xt)) 
            out1 = data_params | cv_params | {'n_features': n_features} 
            out1.pop('skip_params') 
            out1.pop('save_as_n_classes') 
            out1.pop('measure_calibration') 

            opt_path = get_data_path('output')/'metrics'/f'param_optimization_{suffix}.csv'
            df_opt = creaopen_file(opt_path)
            if data_params['skip_params']: 
                if not df_opt.empty: 
                    logger.info('Checking redundant parameter combinations.') 
                    new_param_grid = [] 
                    for params in param_grid:
                        out2 = {**out1, **params}
                        skip = (df_opt[list(out2)] == pd.Series(out2)).all(axis=1).any() 
                        if not skip:    
                            new_param_grid.append(params) 
                    self.param_grid = new_param_grid 
                    if len(param_grid) == 0: 
                        logger.warning('All param combinations already checked') 
                        return 0 
                    logger.info(f'Going to skip {len(param_grid) - len(self.param_grid)} parameter combination(s).') 

            vv_size = self.data_params['vv_size'] 
            random_split = False
            print(Xvt) 
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

        for params in self.param_grid:

            new_metric = self._train_xgb(Xt, yt, params, folds, Xvt, yvt, Xvv, yvv, Xp)
            logger.info(f'CV metrics:')
            logger.info(new_metric["vtrain"])

            if not predict:
                row = out1 | params | new_metric['train'] | new_metric['vtrain'] | new_metric['vvalid']  
                row = {k: [row[k]] for k in row.keys()} 
                df_opt = pd.concat([df_opt, pd.DataFrame(row)], ignore_index=True) 
                df_opt = df_opt.sort_values(by=['ll3_vt','ll3_train','n_estimators']) 
                store_csv(df_opt, opt_path)

             
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
        logger.info(f'Doing CV for hyperparameters {params}') 
        
        n_repeats = self.cv_params['n_repeats']
        n_folds = self.cv_params['n_folds'] 

        if Xp is not None:  
            measure_cal = self.data_params['measure_calibration'] 
            final_n_classes = self.data_params['save_as_n_classes'] 
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
                    p_probas[:,col_idx] =  yp_proba.reshape(-1)

                    if measure_cal: 
                        
                        yvt_proba = self._3class_probas(yvt_proba) if final_n_classes == 3 else yvt_proba 
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
            return {f'{k}_{label}': float(np.mean([m[k] for m in metrics_seq])) for k in keys}

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
            cols = ['KO win','Sub win','Dec win','Draw','Dec loss','Sub loss','KO loss'] 
        elif n_classes == 3: 
            cols = ['win', 'draw', 'loss'] 
        else: 
            logger.warning('Brother 3 or 7 classes please. No probabilities stored.') 
            return 0

        print(34*3,2, '=', probas.shape) # Correct!  
        sample_cols = [str(i) for i in range(probas.shape[-1])] 
        df_prob = pd.DataFrame(columns=sample_cols, data=probas) 
        print(df_names)  
        df_names = df_names.merge(pd.DataFrame({'outcome':cols}), how='cross') 
        df_prob = pd.concat([df_names, df_prob], axis=1)
        print(34*3, 3+3, '=', df_prob.shape) # Correct!  

        #df_prob['_fi'] = df_prob.groupby(['temp_f_id', 'name f1']).cumcount() 
        df_prob1 = df_prob[df_prob['name f1'] < df_prob['name f2']].sort_values(by=['temp_f_id']).reset_index(drop=True)  
        df_prob2 = df_prob[df_prob['name f1'] > df_prob['name f2']].sort_values(by=['temp_f_id']).reset_index(drop=True) 
       

        print(df_prob1.iloc[0], df_prob2.iloc[2]) # These two need to be aligned in probas (*) 
        probas2 = df_prob2[sample_cols].values

        # orig shape: (boutsxn_classes) x (n_repeats x n_folds)  
        # -> bouts x n_classes x (n_foldsxn_repeats)   
        probas2 = probas2.reshape(-1, n_classes, probas2.shape[-1]) 
        # -> n_classes x bouts x (n_foldsxn_repeats)   
        probas2 = probas2.transpose(1,0,2) 
        # Swap classes win-> lose  
        for i in range(int((n_classes-1)/2)): 
            probas2[i], probas2[n_classes-1-i] = probas2[n_classes-1-i], probas2[i] 
        if pd.isna(probas2).any(): 
            print(5/0) 
        # nclasses x n_bouts x -1 -> n_bouts x n_classes x -1 -> (n_bouts x n_classes) x -1 
        probas2 = probas2.transpose(1,0,2).reshape(-1, probas2.shape[-1]) 
        probas1 = df_prob1[sample_cols].values
        probas = np.vstack((probas1, probas2))
        print(probas[0])  # (*) Herre, but this is not the case. 

        if pd.isna(probas).any(): 
            print(5/0) 

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
        file_name = f'{save_for}_{self.data_params["suffix"]}.csv' 
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
          - In each fit, select the top_k features by importance
          - Count how often each feature is selected; also compute mean importance and rank
          - Final selection = top_k by count (tie-break by mean importance)
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
        xgb_params = self.xgb_params.copy()
        xgb_params = {k: xgb_params[k][0] for k in xgb_params.keys()}
        suffix = self.data_params['suffix']

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

                # Select top_k for this fold (if top_k > num features, clip)
                top_idx = np.argsort(imps)[::-1]
                counts[top_idx] += 1
                imp_sum += imps

                # Rank: 1 = best importance, larger = worse; stable tiebreaker
                order = np.argsort(-imps)  # descending importance
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
        ).sort_values(["count", "mean_importance"], ascending=[False, False]).reset_index(drop=True)

        # Write outputs
        out_dir = get_data_path("output") / 'feature_selection'
        store_csv(freq_df, out_dir / f"feature_frequency_{suffix}.csv")

        return {
            "freq_df": freq_df,
            "total_models": total_models,
        }

    def _xgb_factory(self, params, xgb_seed=42): 
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

if '__main__' == __name__: 
    xgb_params = { 
        "max_depth": [3],
        "learning_rate": [0.04], 
        "n_estimators": [600],
        "min_child_weight": [0],
        "gamma": [0],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        # Optional regularization
        "reg_alpha": [0.0],
        "reg_lambda": [1.0]
    }
    data_params = {
        'suffix': 'stan_symm', 
        'k_selected': None,
        'skip_params': True, 
        'vv_seed': 69, 
        'vv_size': 1/3, 
        'save_as_n_classes': 3, 
        'measure_calibration': True
    } 
    cv_params = {
        'fold_seed': 42, 
        'n_folds' : 2, 
        'n_repeats': 5
    }
    
    for k in [95]: 
        data_params['k_selected'] = k 
        CV = CrossValidation(data_params, cv_params, xgb_params)
        CV.run_cv(select_features=False, predict=True)  






         

             
                    
