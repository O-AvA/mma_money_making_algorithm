import pandas as pd 
import numpy as np 
from loguru import logger
import matplotlib.pyplot as plt 

from src.utils.general import (open_csv, get_data_path, store_csv, get_elo_feature_names)
from src.data_processing.feature_manager import FeatureManager 
from src.data_processing.cleaned_data import CleanedFights 
import mlflow
from src.utils.mlflow_utils import (
    setup_mlflow,
    log_params_dict,
    log_metrics_dict,
    log_artifact_safe,
    start_run_with_tags,
    pick_experiment_by_suffix,
)
#from src.data_processing.upcoming_fights import UpcomingFights

class TrainValPred: 

    def __init__(self, feature_sets=None): 

        if feature_sets is not None: 
            self.feature_sets = feature_sets
        else: 
            self.feature_sets = {}

        proc_path = get_data_path('processed')
        self.proc_path = proc_path

        self.train_path = proc_path / 'train.csv' 
        self.valid_path = proc_path / 'valid.csv' 
        self.pred_path = proc_path / 'pred.csv' 

    def merge_features(self, overwrite_feature_sets=False):
        logger.info('Merging features...') 
        pick_experiment_by_suffix("")
        with start_run_with_tags(stage="merge_features", suffix="", run_type="train"):
            # Log chosen feature sets and knobs
            try:
                log_params_dict({"feature_sets": list(self.feature_sets.keys())})
                log_params_dict(self.feature_sets, prefix="features")
            except Exception as e:
                logger.warning(f"Could not log feature set params: {e}")

            fmgr = FeatureManager(self.feature_sets, overwrite_feature_sets)
            df_merged = fmgr.merge_features()
            # Metrics and artifacts
            try:
                group_counts = {f"cols.{name}": int(df.shape[1]) for name, df in getattr(fmgr, "fsets", {}).items()}
                total_cols = int(df_merged.shape[1])
                total_rows = int(df_merged.shape[0])
                nan_rate = float(df_merged.isna().mean().mean()) if total_cols > 0 else 0.0
                log_metrics_dict({"rows": total_rows, "cols": total_cols, "nan_rate": round(nan_rate, 6)})
                if group_counts:
                    log_metrics_dict(group_counts)
                merged_path = get_data_path('interim') / 'chosen_features_merged.csv'
                log_artifact_safe(merged_path)
                # Log schema and small head sample
                mlflow.log_dict({
                    "columns": df_merged.columns.tolist()[:500],
                    "n_columns": total_cols,
                    "n_rows": total_rows
                }, "merged_features_schema.json")
                mlflow.log_text(df_merged.head(5).to_csv(index=False), "merged_features_head.csv")
            except Exception as e:
                logger.warning(f"Could not log merged features metrics/artifacts: {e}")
    logger.info('Features merged.') 

    def open_merged_features(self): 
        df_tv = FeatureManager().merged_features  
        return df_tv

    def construct_pred(self):
        logger.info('Constructing dataset to predict...') 
        pick_experiment_by_suffix("")
        run = start_run_with_tags(stage="construct_pred", suffix="", run_type="predict")
        try:
            try:
                log_params_dict(self.feature_sets, prefix="features")
            except Exception as e:
                logger.warning(f"Could not log pred feature set params: {e}")

            # Remove file if already exists 
            proc_pred_path = get_data_path('processed') / 'pred.csv'
            if proc_pred_path.exists():
                proc_pred_path.unlink()

            for fname, fparams in self.feature_sets.items():
                logger.info(f'Creating {fname} for pred...') 
                FeatureManager.create_feature_set(fname, fparams, upcoming_fights=True)
                logger.info(f'Appended {fname} to pred.')
            logger.info('All feature sets added to pred.') 
            
            # UpcomingFights() stores pred as 1 sample per bout so 
            # we need to duplicate it. 
            dfp = open_csv(self.pred_path)
            dfp = CleanedFights(dfp).duplicate(base_cols=dfp.columns.tolist(), add_labels=True) 
            dfp = dfp.drop(columns=[col for col in dfp.columns if col.endswith('E f2')], errors='ignore') 
            E1_keys = [col for col in dfp.columns if col.endswith('E f1')] 
            new_E_keys = [col.replace(' f1','') for col in E1_keys]
            dfp.rename(columns = dict(zip(E1_keys, new_E_keys)), inplace=True) 

            # Store names and f_ids
            dfp_names = dfp[['upp_low', 'temp_f_id', 'name f1', 'name f2']] 
            dfp = dfp.drop(columns= dfp_names.columns) 
            name_p = get_data_path('interim') / 'pred_names.csv'
            dfp_p = get_data_path('processed') / 'pred.csv'
            store_csv(dfp,dfp_p)
            store_csv(dfp_names, name_p) 

            # Log metrics and artifacts (schema/head to avoid large artifacts)
            try:
                log_metrics_dict({
                    "pred.rows": int(dfp.shape[0]),
                    "pred.cols": int(dfp.shape[1]),
                    "pred.nan_rate": float(dfp.isna().mean().mean() if dfp.shape[1] > 0 else 0.0)
                })
                log_artifact_safe(name_p)
                mlflow.log_dict({
                    "columns": dfp.columns.tolist()[:500],
                    "n_columns": int(dfp.shape[1]),
                    "n_rows": int(dfp.shape[0])
                }, "pred_schema.json")
                mlflow.log_text(dfp.head(5).to_csv(index=False), "pred_head.csv")
            except Exception as e:
                logger.warning(f"Could not log pred artifacts: {e}")
        finally:
            mlflow.end_run()

        


    def get_path(self, dataset, after1=None): # after2=None): 
        # after1, after2 in ['  svd', 'selector']  
        suffix = f'_{after1}' if after1 is not None else ""
        #suffix = f'{suffix}_{after2}' if after2 is not None else ""
        data_path = get_data_path('processed') / f'{dataset}{suffix}.csv'
        return data_path

    

    def show_correlations(self, df, simplify_result=True, duplicate=True): 
        if 'result' not in df.columns: 
            df = CleanedFights(df)
            df['result'] = CleanedFights()['result'] 

        df = df[df['result'] >= 0]
        if simplify_result: 
            result_map = {0:0, 1:0, 2:0, 3:1, 4:2, 5:2, 6:2} 
            df['result'] = df['result'].map(result_map)  


        if duplicate: 
            df = CleanedFights(df).duplicate(base_cols = df.columns.tolist()) 
        df = df.drop(columns=['name f1', 'name f2', 'tau'],errors='ignore')
        df.drop(columns=df.columns[df.columns.isin(['tau','upp_low','temp_f_id'])], inplace=True)
        correlations = df.corr()['result'].abs().sort_values(ascending=False)
        for col, corr in correlations[1:].items():
            print(f"{col}: {corr}")

    def split_trainval(self, last_years=2, sample_size=0.15): 
        """
        Samples validation data and saves both it and the remainder 
        of the training data. 
        It also removes categorical data. 
        
        Args: 
            last_years int: 
                Only sample from the last last_years years. 
            sample_size float: 
                float between [0,1), the fraction of samples 
                to take from the training data

        Returns:
            valid.csv, a DataFrame of the validation set 
            train.csv, a DataFrame of the training set 
        """ 
        pick_experiment_by_suffix("")
        start_run_with_tags(stage="split_trainval", suffix="", run_type="train")
        try:
            log_params_dict({"last_years": last_years, "sample_size": sample_size})
            dff = self.open_merged_features()
            dff = dff[dff['result'] >= 0]
            dff = dff.drop(columns=[col for col in dff.columns if '_REMOVE_' in col])  

            dff['f_id'] = dff.index 
            if pd.isna(dff['female']).any(): 
                print('in features...', 5/0) 

            # Delete columns for elo K-value that is not used 
            unused_K_cols = [] 
            params_dict = self.feature_sets or {}
            param_vals = [v for pdict in params_dict.values() for v in pdict.values()]
            if 'cust' not in param_vals: 
                Kcols = [f'Kcustr{ri} f{fi}' for ri in range(2,6) for fi in [1,2]] 
                Kcols.append(['Kcust f1', 'Kcust f2']) 
            if 'log' not in param_vals: 
                Kcols = [f'Klogr{ri} f{fi}' for ri in range(2,6) for fi in [1,2]] 
                Kcols.append(['Klog f1', 'Klog f2']) 

            ly = last_years
            ss = sample_size 

            dff['ya'] = (dff['tau'].max()-dff['tau'])/52
            dfv = dff[dff['ya'] < ly]  
            v_size = len(dfv)/len(dff) 
            if v_size < ss: 
                logger.warning(
                    f'Parameter sample_size={ss} too large for parameter last_years = {ly}. sample_size drops to {v_size}'
                )
                dft = dff[dff['ya'] >= ly] 
            else:  
                n = int(ss*len(dff))
                f_ids = dfv['f_id']
                f_ids_v = np.random.choice(f_ids, size=n, replace=False) 
                dfv = dfv[dfv['f_id'].isin(f_ids_v)] 
                dft = dff[~dff['f_id'].isin(f_ids_v)]

            dfv = CleanedFights(dfv).duplicate(base_cols=dfv.columns.tolist()) 
            dft = CleanedFights(dft).duplicate(base_cols=dft.columns.tolist()) 

            # Now we can drop the expectancies for fighter 2.
            # We kept them to make duplicating slightly easier even though its slightly
            # more inefficient.
            dft = dft.drop(columns=[col for col in dft.columns if col.endswith('E f2')], errors='ignore') 
            dfv = dfv.drop(columns=[col for col in dfv.columns if col.endswith('E f2')], errors='ignore') 
            E1_keys = [col for col in dft.columns if col.endswith('E f1')] 
            new_E_keys = [col.replace(' f1','') for col in E1_keys]
            dft.rename(columns = dict(zip(E1_keys, new_E_keys)), inplace=True) 
            dfv.rename(columns = dict(zip(E1_keys, new_E_keys)), inplace=True) 

            # Making the samples anonymous 
            p_alt = get_data_path('interim') 
            p_pro = get_data_path('processed') 

            name_cols = ['upp_low', 'temp_f_id', 'name f1','name f2'] 

            dfvmap = dfv[name_cols] 
            dftmap = dft[name_cols]

            p_map_v = p_alt / 'valid_names.csv' 
            p_map_f = p_alt / 'train_names.csv' 

            store_csv(dfvmap,p_map_v)
            store_csv(dftmap,p_map_f)

            del_cols = name_cols + ['ya', 'f_id'] 
            del_cols.remove('temp_f_id') 
            # For dft we'll use temp_f_id for folding 
            dft.drop(columns=del_cols,inplace=True,errors='ignore') 
            dfv.drop(columns=del_cols + ['temp_f_id'],inplace=True,errors='ignore')

            store_csv(dft, p_pro / 'train.csv') 
            store_csv(dfv, p_pro / 'valid.csv')
            if pd.isna(dff['female']).any(): 
                print('after get_trainval', 5/0) 

            # log metrics and artifacts
            try:
                metrics = {
                    "train.rows": int(dft.shape[0]),
                    "train.cols": int(dft.shape[1]),
                    "valid.rows": int(dfv.shape[0]),
                    "valid.cols": int(dfv.shape[1]),
                    "train.nan_rate": float(dft.isna().mean().mean() if dft.shape[1] > 0 else 0.0),
                    "valid.nan_rate": float(dfv.isna().mean().mean() if dfv.shape[1] > 0 else 0.0),
                }
                if 'tau' in dft.columns:
                    metrics.update({
                        "train.tau_min": float(dft['tau'].min()),
                        "train.tau_max": float(dft['tau'].max())
                    })
                if 'tau' in dfv.columns:
                    metrics.update({
                        "valid.tau_min": float(dfv['tau'].min()),
                        "valid.tau_max": float(dfv['tau'].max())
                    })
                if 'result' in dft.columns:
                    for k, v in dft['result'].value_counts().to_dict().items():
                        metrics[f"train.result_{k}"] = int(v)
                if 'result' in dfv.columns:
                    for k, v in dfv['result'].value_counts().to_dict().items():
                        metrics[f"valid.result_{k}"] = int(v)
                log_metrics_dict(metrics)
                log_artifact_safe(p_map_f)
                log_artifact_safe(p_map_v)
            except Exception as e:
                logger.warning(f"Could not log split_trainval metrics/artifacts: {e}")
        finally:
            mlflow.end_run()

    def get_folds(self, suffix, n_repeats, n_folds = 5, first_seed=None):   
        suffix = f'_{suffix}' if suffix != '' else suffix 

        dft = self.open(f'train{suffix}') 
        f_ids = dft['temp_f_id'].unique()

        if first_seed is None: 
            seeds = np.random.randint(low=0, high=100, size=n_repeats) 
        else: 
            seeds = range(first_seed, first_seed + n_repeats)   
        
       
        all_folds = [] 
        for j in range(n_repeats): 
            rng = np.random.default_rng(seed=seeds[j])
            f_ids = rng.permutation(dft['temp_f_id'].unique()) 
            f_id_subs = np.array_split(f_ids,n_folds) 

            train_ids = [] 
            val_ids = []
            for z in range(n_folds): 
                train_ids.append(dft[~dft['temp_f_id'].isin(f_id_subs[z])].index) 
                val_ids.append(dft[dft['temp_f_id'].isin(f_id_subs[z])].index) 
            folds = list(zip(train_ids,val_ids)) 
            all_folds.append(folds)

        dft = dft.drop(columns=['temp_f_id'])

        logger.info(f'Created {n_repeats} unique 5-folds') 
        if len(dft[pd.isna(dft['result'])]) > 0: 
            print('earlier', 5/0) 

        return dft, all_folds

    def get_trainvalpred(self, which=""): 
        which = f'_{which}' if which != "" else which

        proc_path = get_data_path('processed') 

        data = [open_csv(proc_path /f'train{which}.csv'),
                open_csv(proc_path /f'valid{which}.csv'), 
                open_csv(proc_path / f'pred{which}.csv') 
        ] 

        return data  
    
    def get_1h_columns(self):
        proc_path = get_data_path('interim') 
        dff = open_csv(proc_path / 'chosen_features_merged.csv') 
        one_hot_cols = ['male','female',
                        'title bout', 'normal bout',
                        '3 rounds', '5 rounds'
        ]
        wcs = ['Catch ', 'Straw', 'Fly', 'Bantam', 'Feather', 'Light', 'Welter', 
               'Middle', 'Light Heavy', 'Heavy'
        ]
        one_hot_cols.extend([f'{wc}weight' for wc in wcs]) 
        has_not_cols = [col for col in dff.columns if 'has not' in col]
        has_cols = [col.replace('has not', 'has') for col in has_not_cols]
        one_hot_cols.extend(has_not_cols) 
        one_hot_cols.extend(has_cols)

        return one_hot_cols 

    def get_path(self, tvp='train',which1=""): 
        proc_path = get_data_path('processed') 
        which1 = f'_{which1}' if which1 != "" else which1

        file_name = f'{tvp}{which1}.csv'

        return proc_path / file_name 


    def symmetrize(self, for_svd, pred_only=False):
        """
        (Anti-)symmetrizes the dataset. I.e., it makes antilinear
        superpositions of fighter 1 and fighter 2 columns. 
        df_f1 -> (df_f1 + df_f2)/sqrt(2) 
        df_f2 -> (df_f1 - df_f2)/sqrt(2)

        If the next step is an SVD, set for_svd to True. Otherwise
        one-hot encoded columns will be dropped and data will not 
        be standardized. 
        """

        # pred_only is only meaningful for the non-SVD path; SVD requires train/valid/pred
        if for_svd and pred_only:
            raise ValueError("pred_only=True is not supported when for_svd=True; SVD requires train/valid/pred.")

        suffix = 'svd' if for_svd else 'symm'
        pick_experiment_by_suffix(suffix)
        start_run_with_tags(stage="symmetrize", suffix=suffix, run_type="train")
        try:
            log_params_dict({"for_svd": bool(for_svd), "pred_only": bool(pred_only)})

            if not for_svd:
                data = self.get_trainvalpred(which="")
                del_1h = ['male', 'normal bout', '3 rounds']
                del_1h.extend([col for col in data[0].columns if 'weight' in col])
                del_1h.extend([col for col in data[0].columns if 'has not' in col])
                logger.info('Going to drop one-hot encoded columns and not standardize.')
            else:
                data = self._standardize()

            # Reduce to only prediction set when requested on the non-SVD path
            data = [data[2]] if (pred_only and not for_svd) else data

            logger.info('Beginning (anti-)symmetrization')

            # Configure optional swap lists when acc features present
            A2_labels, D2_labels = [], []
            if self.feature_sets and any('acc' in fset_name for fset_name in self.feature_sets.keys()):
                rounds = any('acc_elos_per_round' in fset_name for fset_name in self.feature_sets.keys())
                avgs = not rounds
                A2_labels = get_elo_feature_names(which=['A'], fighters=[2], avgs=avgs, rounds=rounds, extras=False)
                D2_labels = get_elo_feature_names(which=['D'], fighters=[2], avgs=avgs, rounds=rounds, extras=False)

            # oh_cols contains one_hot_columns that can also figure as flags, but del_1h only contains pure one-hot
            oh_cols = self.get_1h_columns()
            cols1 = [col for col in data[0].columns if col.endswith('f1') and col not in oh_cols]
            cols2 = [col.replace('f1','f2') for col in cols1]

            before_cols = [data[i].shape[1] for i in range(len(data))]
            for k in range(len(data)):
                if A2_labels and D2_labels:
                    data[k][A2_labels], data[k][D2_labels] = data[k][D2_labels], data[k][A2_labels]

                # Set nans to 0 when (anti-)symmetrizing for the svd.
                dfk = data[k].fillna(0) if for_svd else data[k]

                dfk[cols1], dfk[cols2] = (
                    (dfk[cols1].values + dfk[cols2].values) / np.sqrt(2),
                    (dfk[cols1].values - dfk[cols2].values) / np.sqrt(2),
                )

                # Fix this in the feature engineering module!!
                E1cols = [col for col in dfk.columns if col.endswith('Ef1')]
                Ecols = [col.replace('f1','') for col in E1cols]
                dfk.rename(columns=dict(zip(E1cols, Ecols)), inplace=True)
                dfk.drop(columns=[col for col in dfk.columns if col.endswith('Ef2')], inplace=True)

                if not for_svd:
                    dfk['weightclass'] = dfk.loc[:, 'Catch weight':'Heavyweight'].to_numpy().argmax(axis=1)
                    dfk = dfk.drop(columns=del_1h)

                data[k] = dfk

            logger.info('Dataset (anti-)symmetrized.')

            if not for_svd:
                if pred_only:
                    out_p = self.get_path(tvp='pred', which1='symm')
                    store_csv(data[0], out_p)
                    log_artifact_safe(out_p)
                else:
                    out_t = self.get_path(tvp='train', which1='symm')
                    out_v = self.get_path(tvp='valid', which1='symm')
                    out_p = self.get_path(tvp='pred', which1='symm')
                    store_csv(data[0], out_t)
                    store_csv(data[1], out_v)
                    store_csv(data[2], out_p)
                    try:
                        mlflow.log_dict({"columns": data[0].columns.tolist(), "n_rows": int(data[0].shape[0])}, "train_symm_schema.json")
                        mlflow.log_dict({"columns": data[1].columns.tolist(), "n_rows": int(data[1].shape[0])}, "valid_symm_schema.json")
                        mlflow.log_dict({"columns": data[2].columns.tolist(), "n_rows": int(data[2].shape[0])}, "pred_symm_schema.json")
                    except Exception as e:
                        logger.warning(f"Could not log symm schemas: {e}")
            else:
                try:
                    metrics = {}
                    for i, label in enumerate(["train", "valid", "pred"][:len(data)]):
                        metrics[f"{label}.cols_before"] = int(before_cols[i])
                        metrics[f"{label}.cols_after"] = int(data[i].shape[1])
                        metrics[f"{label}.nan_rate"] = float(data[i].isna().mean().mean() if data[i].shape[1] > 0 else 0.0)
                    log_metrics_dict(metrics)
                except Exception as e:
                    logger.warning(f"Could not log symmetrize metrics: {e}")
                return data
        finally:
            mlflow.end_run()
        
    
    def _standardize(self,which=''):
        logger.info('Standardizing data...') 
        start_run_with_tags(stage="standardize", suffix="svd", run_type="train")
        try:
            dft = self.open('train')
            dfv = self.open('valid')
            dfp = self.open('pred')

            dft['temp_col'] = [0]*len(dft)
            dfv['temp_col'] = [1]*len(dfv)
            dfp['temp_col'] = [2]*len(dfp)

            df_tvp = pd.concat([dft, dfv, dfp], ignore_index=True)

            one_hot_cols = self.get_1h_columns() + ['temp_col', 'result', 'temp_f_id', 'weightclass']
            one_hot_or_flag = [col for col in one_hot_cols if col in df_tvp.columns]

            df_stan = df_tvp[[col for col in df_tvp.columns if col not in one_hot_or_flag]]
            df_stan = ((df_stan - df_stan.mean(skipna=True)) / df_stan.std(skipna=True, ddof=0))

            df_stan = pd.concat([df_stan, df_tvp[one_hot_or_flag]], axis=1)

            dft = df_stan[df_stan['temp_col'] == 0].drop(columns=['temp_col'])
            dfv = df_stan[df_stan['temp_col'] == 1].drop(columns=['temp_col'])
            dfp = df_stan[df_stan['temp_col'] == 2].drop(columns=['temp_col'])

            if 'temp_f_id' not in dft.columns:
                print('stan', 5/0)

            logger.info('Standardization complete.')
            try:
                log_metrics_dict({
                    "std.train.cols": int(dft.shape[1]),
                    "std.valid.cols": int(dfv.shape[1]),
                    "std.pred.cols": int(dfp.shape[1])
                })
            except Exception as e:
                logger.warning(f"Could not log standardization metrics: {e}")

            #store_csv(dft, str(self.get_path(tvp ='train', which1=which)).replace('.csv','_stan.csv'))
            #store_csv(dfv, str(self.get_path(tvp ='valid', which1=which)).replace('.csv','_stan.csv'))
            #store_csv(dfp, str(self.get_path(tvp = 'pred', which1=which)).replace('.csv','_stan.csv'))

            return [dft, dfv, dft]
        finally:
            mlflow.end_run()

    def do_svd(self, k = 204, plot_sv = False): 
        pick_experiment_by_suffix("svd")
        start_run_with_tags(stage="svd", suffix="svd", run_type="train")
        try:
            log_params_dict({"k": int(k), "plot_sv": bool(plot_sv)})
            data = self.symmetrize(for_svd=True)
            dft = data[0]

            if 'temp_f_id' not in dft.columns:
                print('before svd', 5/0)

            exclude_cols = ['result', 'xp score_f1','xp_score_f2', 'temp_f_id']
            exclude_cols.extend([col for col in dft.columns if 'Kcust' in col])
            exclude_cols.extend([col for col in dft.columns if 'Klog' in col])
            exclude_cols = [exc_col for exc_col in exclude_cols if exc_col in dft.columns]

            for col in dft.columns:
                if pd.isna(dft[col]).any():
                    print(col)
            arrt = dft[[col for col in dft.columns if col not in exclude_cols]].values

            U, s, V = np.linalg.svd(arrt, full_matrices=False)
            for i, si in enumerate(s):
                print(i, si)
            s_trunc = s[:k]
            arrt = (arrt @ V.T)[:,:k]
            dft_svd = pd.DataFrame(arrt, columns=[f's{si}' for si in range(s_trunc.shape[0])])
            dft_svd[exclude_cols] = dft.copy()[exclude_cols].values

            dfv = data[1]
            dfp = data[2]

            dfv = dfv[dft.columns]
            dfp = dfv[dfp.columns]

            resultv = dfv['result']
            resultp = dfv['result']

            arrv = dfv[[col for col in dfv.columns if col not in exclude_cols]].values
            arrp = dfp[[col for col in dfp.columns if col not in exclude_cols]].values

            arrv = (arrv @ V.T)[:,:k]
            arrp = (arrp @ V.T)[:,:k]

            dfv_svd = pd.DataFrame(arrv, columns=[f's{si}' for si in range(s_trunc.shape[0])])
            dfv_svd[exclude_cols] = dfv.copy()[exclude_cols].values
            dfp_svd = pd.DataFrame(arrp, columns=[f's{si}' for si in range(s_trunc.shape[0])])
            dfp_svd[exclude_cols] = dfp.copy()[exclude_cols].values

            if 'temp_f_id' not in dft_svd.columns:
                print('before svd', 5/0)
            store_csv(dft_svd, self.proc_path /'train_svd.csv')
            store_csv(dfv_svd, self.proc_path /'valid_svd.csv')
            store_csv(dfp_svd, self.proc_path /'pred_svd.csv')

            # Log SVD metrics and schemas
            try:
                total_energy = float(np.sum(s**2)) if s.size else 0.0
                k_energy = float(np.sum(s_trunc**2)) if s_trunc.size else 0.0
                explained_ratio = float(k_energy/total_energy) if total_energy > 0 else 0.0
                log_metrics_dict({
                    "s.total_singular": int(s.size),
                    "s.top_value": float(s[0]) if s.size else 0.0,
                    "s.k_value": float(s[k-1]) if s.size and k <= s.size else 0.0,
                    "s.explained_ratio_k": explained_ratio
                })
                mlflow.log_dict({"columns": dft_svd.columns.tolist(), "n_rows": int(dft_svd.shape[0])}, "train_svd_schema.json")
                mlflow.log_dict({"columns": dfv_svd.columns.tolist(), "n_rows": int(dfv_svd.shape[0])}, "valid_svd_schema.json")
                mlflow.log_dict({"columns": dfp_svd.columns.tolist(), "n_rows": int(dfp_svd.shape[0])}, "pred_svd_schema.json")
            except Exception as e:
                logger.warning(f"Could not log SVD metadata: {e}")

            if plot_sv:
                plt.plot(range(len(s)), np.log(s))
                plt.xlabel('Singular value index')
                plt.ylabel('Log singular values')
                plt.scatter([k]*50, np.linspace(np.log(s)[-1], np.log(s)[0],50), s=1, color='red')
                plt.scatter(np.linspace(0,len(s),100), [0]*100, s=1, color='black')
                plt.grid()
                try:
                    fig_path = get_data_path('output') / 'metrics' / 'svd_singular_values.png'
                    fig_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                    log_artifact_safe(fig_path)
                except Exception as e:
                    logger.warning(f"Could not save/log singular value plot: {e}")
                finally:
                    plt.show()
        finally:
            mlflow.end_run()
    
    def open(self, data_set, suffix=None): 
        if suffix is None: 
            df = open_csv(self.proc_path / f'{data_set}.csv') 
        else: 
            suffix = f'_{suffix}' if suffix != '' else suffix
            df = open_csv(self.proc_path / f'{data_set}{suffix}.csv') 
        return df 
        



        




if __name__ == '__main__': 
    from src.feature_engineering.get_elo_params import set_elo_params 

    overwrite_all = True

    feature_sets = {
        'base_features': {},
        'elo_params': {'d_params': set_elo_params()},
        'wl_elos': {'which_K': 'log'},
        'stat_elos_round_averages': {'which_K': 'log', 'always_update': False, 'exact_score': True},
        'acc_elos_round_averages': {'which_K': 'log'},
        'rock_paper_scissor': {'intervals': [0, 2]},
    }

    TVP = TrainValPred(feature_sets)

    from src.data_processing.scrape_pred import scrape_pred 
    from src.data_processing.clean_pred import clean_pred 

    #scrape_pred() 
    clean_pred() 
    TVP.construct_pred() 

    #FeatureManager({'rock_paper_scissor': feature_sets['rock_paper_scissor']}, overwrite_all=True) 
    #TVP.merge_features(overwrite_feature_sets=False) 
    #TVP.show_correlations(TVP.open_merged_features()) 
    #TVP.construct_pred()
    #TVP.split_trainval(last_years = 1.5, sample_size = 0.15) 
    #TVP.symmetrize(for_svd=False) 
    #TVP.do_svd(plot_sv = True)  

    #data = TVP.get_trainvalpred(which='svd') 
    #dftv = pd.concat([data[0], data[1]], ignore_index=True)
    #TVP.show_correlations(dftv, duplicate=False) 






         
        
# Define a reasonable default search space for XGBoost multi-class
