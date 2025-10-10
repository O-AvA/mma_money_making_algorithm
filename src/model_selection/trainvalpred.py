import os
import pandas as pd 
import numpy as np 
from loguru import logger
import matplotlib.pyplot as plt 

from src.utils.general import (open_csv, get_data_path, store_csv, creaopen_file)
from src.data_processing.feature_manager import FeatureManager 
from src.data_processing.cleaned_data import CleanedFights 
import mlflow
from mlflow.tracking import MlflowClient

def _safe_log_params(params: dict, prefix: str = "") -> None:
    """Safely log parameters with error handling."""
    try:
        flat_params = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat_params[f"{key}.{sub_k}"] = sub_v
            else:
                flat_params[key] = v
        mlflow.log_params(flat_params)
    except Exception as e:
        logger.warning(f"Could not log params: {e}")

def _safe_log_metrics(metrics: dict, prefix: str = "") -> None:
    """Safely log metrics with error handling."""
    try:
        flat_metrics = {}
        for k, v in metrics.items():
            key = f"{prefix}.{k}" if prefix else k
            flat_metrics[key] = v
        mlflow.log_metrics(flat_metrics)
    except Exception as e:
        logger.warning(f"Could not log metrics: {e}")

def _safe_log_artifact(path) -> None:
    """Safely log artifact with error handling."""
    try:
        from pathlib import Path
        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p))
    except Exception as e:
        logger.warning(f"Could not log artifact {path}: {e}")

def _check_pipeline_context():
    """Check if we're running in a pipeline context."""
    try:
        from src.utils.mlflow_pipeline import get_pipeline_manager
        return get_pipeline_manager() is not None
    except ImportError:
        return False

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
        
        # Check if we're in a pipeline context
        try:
            from src.utils.mlflow_pipeline import get_pipeline_manager, substage_run
            pipeline_manager = get_pipeline_manager()
            in_pipeline = pipeline_manager is not None
        except ImportError:
            in_pipeline = False
        
        if in_pipeline:
            # Use pipeline substage
            with substage_run("merge_features",
                            feature_sets=list(self.feature_sets.keys()),
                            overwrite_feature_sets=overwrite_feature_sets):
                self._do_merge_features(overwrite_feature_sets)
        else:
            # Standalone execution - create our own MLflow run
            try:
                mlflow.set_experiment("feature_engineering_standalone")
            except Exception as e:
                logger.warning(f"MLflow setup failed: {e}")
            
            with mlflow.start_run(run_name="merge_features_standalone"):
                self._do_merge_features(overwrite_feature_sets)
    
    def _do_merge_features(self, overwrite_feature_sets=False):
        """Internal method to perform feature merging with MLflow logging."""
        # Log chosen feature sets and knobs
        try:
            _safe_log_params({"feature_sets": list(self.feature_sets.keys())})
            _safe_log_params(self.feature_sets, prefix="features")
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
            _safe_log_metrics({"rows": total_rows, "cols": total_cols, "nan_rate": round(nan_rate, 6)})
            if group_counts:
                _safe_log_metrics(group_counts)
            merged_path = get_data_path('interim') / 'chosen_features_merged.csv'
            _safe_log_artifact(merged_path)
            # Log schema and small head sample
            try:
                mlflow.log_dict({
                    "columns": df_merged.columns.tolist()[:500],
                    "n_columns": total_cols,
                    "n_rows": total_rows
                }, "merged_features_schema.json")
                mlflow.log_text(df_merged.head(5).to_csv(index=False), "merged_features_head.csv")
            except Exception as e:
                logger.warning(f"Could not log MLflow dict/text: {e}")
        except Exception as e:
            logger.warning(f"Could not log merged features metrics/artifacts: {e}")
        logger.info('Features merged.') 

    def open_merged_features(self): 
        df_tv = FeatureManager().merged_features  
        return df_tv

    def construct_pred(self, scrape_and_clean = True):
        logger.info('Constructing dataset to predict...') 

        
        # Check if we're in a pipeline context
        try:
            from src.utils.mlflow_pipeline import get_pipeline_manager, substage_run
            pipeline_manager = get_pipeline_manager()
            in_pipeline = pipeline_manager is not None
        except ImportError:
            in_pipeline = False
        
        if in_pipeline:
            # Use pipeline substage
            with substage_run("build_prediction_data",
                            feature_sets=list(self.feature_sets.keys())):
                self._do_construct_pred()
        else:
            # Standalone execution
            try:
                mlflow.set_experiment("prediction_data_standalone")
            except Exception as e:
                logger.warning(f"MLflow setup failed: {e}")
            
            with mlflow.start_run(run_name="construct_pred_standalone"):
                self._do_construct_pred()
    
    def _do_construct_pred(self):
        """Internal method to construct prediction dataset with MLflow logging."""
        try:
            _safe_log_params(self.feature_sets, prefix="features")
        except Exception as e:
            logger.warning(f"Could not log pred feature set params: {e}")
        # Main logic (always run regardless of logging success)
        # Remove file if already exists
        proc_pred_path = get_data_path('processed') / 'pred.csv'
        if proc_pred_path.exists():
            proc_pred_path.unlink()

        for fname, fparams in self.feature_sets.items():
            logger.info(f'Creating {fname} for pred...')
            FeatureManager.create_feature_set(fname, fparams, upcoming_fights=True)
            logger.info(f'Appended {fname} to pred.')
        logger.info('All feature sets added to pred.')

        # UpcomingFights() stores pred as 1 sample per bout so we need to duplicate it.
        dfp = open_csv(self.pred_path)
        dfp = CleanedFights(dfp).duplicate(base_cols=dfp.columns.tolist(), add_labels=True)

        dfp = dfp.drop(columns=[col for col in dfp.columns if col.endswith('Ef2')], errors='ignore')
        E1_keys = [col for col in dfp.columns if col.endswith('Ef1')]
        new_E_keys = [col.replace('f1','') for col in E1_keys]
        dfp.rename(columns=dict(zip(E1_keys, new_E_keys)), inplace=True)

        # Store names and f_ids
        dfp_names = dfp[['upp_low', 'temp_f_id', 'name f1', 'name f2']]
        dfp = dfp.drop(columns=dfp_names.columns)
        dfp_names['tau'] = dfp['tau'] 
        name_p = get_data_path('interim') / 'pred_names.csv'
        dfp_p = get_data_path('processed') / 'pred.csv'
        store_csv(dfp, dfp_p)
        store_csv(dfp_names, name_p)

        # Log metrics and artifacts (schema/head to avoid large artifacts)
        try:
            _safe_log_metrics({
                "pred.rows": int(dfp.shape[0]),
                "pred.cols": int(dfp.shape[1]),
                "pred.nan_rate": float(dfp.isna().mean().mean() if dfp.shape[1] > 0 else 0.0)
            })
            _safe_log_artifact(name_p)
            try:
                mlflow.log_dict({
                    "columns": dfp.columns.tolist()[:500],
                    "n_columns": int(dfp.shape[1]),
                    "n_rows": int(dfp.shape[0])
                }, "pred_schema.json")
                mlflow.log_text(dfp.head(5).to_csv(index=False), "pred_head.csv")
            except Exception as e:
                logger.warning(f"Could not log MLflow dict/text: {e}")
        except Exception as e:
            logger.warning(f"Could not log pred artifacts: {e}")

        


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

    def split_trainval(
            self, 
            last_years=2, 
            sample_size=0.15,
            if_on_size_then_randomly = False
    ): 
        """
        Split the validation set from the training data. To get a good indication of
        how well the model performs, we choose this based on recency. Optionally,
        if the validation size exceeds the desired fraction, you can sample randomly
        instead of using a strict recency split.
        """

        # Detect MLflow pipeline context correctly (outside of the docstring)
        try:
            from src.utils.mlflow_pipeline import get_pipeline_manager, substage_run
            pipeline_manager = get_pipeline_manager()
            in_pipeline = pipeline_manager is not None
        except ImportError:
            in_pipeline = False

        if in_pipeline:
            with substage_run("split_trainval",
                            last_years=last_years,
                            sample_size=sample_size,
                            if_on_size_then_randomly=if_on_size_then_randomly):
                self._do_split_trainval(last_years, sample_size, if_on_size_then_randomly)
        else:
            try:
                mlflow.set_experiment("data_splitting_standalone")
            except Exception as e:
                logger.warning(f"MLflow setup failed: {e}")
            
            with mlflow.start_run(run_name="split_trainval_standalone"):
                self._do_split_trainval(last_years, sample_size, if_on_size_then_randomly)
    
    def _do_split_trainval(self, last_years, sample_size, if_on_size_then_randomly=False):
        """Internal method to split train/val with MLflow logging."""
        try:
            _safe_log_params({
                "last_years": last_years,
                "sample_size": sample_size,
                "if_on_size_then_randomly": bool(if_on_size_then_randomly)
            })
        except Exception as e:
            logger.warning(f"Could not log split params: {e}")
        # Main logic (always run regardless of logging success)
        dff = self.open_merged_features()
        dff = dff[dff['result'] >= 0]
        dff = dff.drop(columns=[col for col in dff.columns if '_REMOVE_' in col])

        dff['f_id'] = dff.index
        if pd.isna(dff['female']).any():
            print('in features...', 5/0)

        # Delete columns for elo K-value that is not used (currently computed but not dropped)
        unused_K_cols = []
        params_dict = self.feature_sets or {}
        param_vals = [v for pdict in params_dict.values() for v in pdict.values()]
        if 'cust' not in param_vals:
            unused_K_cols = [f'Kcustr{ri} f{fi}' for ri in range(2, 6) for fi in [1, 2]]
            unused_K_cols.extend(['Kcust f1', 'Kcust f2'])
        if 'log' not in param_vals:
            unused_K_cols.extend([f'Klogr{ri} f{fi}' for ri in range(2, 6) for fi in [1, 2]])
            unused_K_cols.extend(['Klog f1', 'Klog f2'])
        # Optionally drop if present
        # dff = dff.drop(columns=[c for c in unused_K_cols if c in dff.columns], errors='ignore')

        ly = last_years
        ss = sample_size

        dfp = creaopen_file(get_data_path('interim')/'pred_clean')
        if not dfp.empty:
            curr_tau = dfp['tau'].values[0] 
        else:
            # Could be two week hiatus, but doesnt matter for the model. 
            curr_tau = dff['tau'].max() + 1

        dff['ya'] = (curr_tau - dff['tau']) / 52
        dfv = dff[dff['ya'] < ly]
        v_size = len(dfv) / len(dff) if len(dff) else 0.0
        if v_size < ss:
            ss = v_size
            logger.info(
                f'Validation set contains last {ly} years of fights, representing {round(ss*100)}% of data.'
            )

            dft = dff[dff['ya'] >= ly]
        else:
            if if_on_size_then_randomly:
                # Selecting on f_id is not really necessary here because data has not been duplicated yet
                n = int(ss * len(dff))
                f_ids = dfv['f_id']
                f_ids_v = np.random.choice(f_ids, size=n, replace=False)
                dfv = dfv[dfv['f_id'].isin(f_ids_v)]
             
                dft = dff[~dff['f_id'].isin(f_ids_v)]


                logger.info(
                        f'Validation set {round(ss*100,1)}% of total all data, randomly drawn from last {ly} years of fights'
                )
            else:
                # Fallback: keep recency-based selection, log bounds safely
                dff = dff.sort_values(by=['tau']).reset_index(drop=True) 
                v_idx = round((1-ss)*len(dff))
                dfv = dff.iloc[v_idx:] 
                dft = dff.iloc[:v_idx] 
                ly = dfv['ya'].max() 
                logger.info(
                    f'Validation set contains last {round(ly,1)} years of fights, representing {round(ss*100)}% of data.'
                )


        dfv = CleanedFights(dfv).duplicate(base_cols=dfv.columns.tolist())
        dft = CleanedFights(dft).duplicate(base_cols=dft.columns.tolist())

        # Now we can drop the expectancies for fighter 2.
        dft = dft.drop(columns=[col for col in dft.columns if col.endswith('Ef2')], errors='ignore')
        dfv = dfv.drop(columns=[col for col in dfv.columns if col.endswith('Ef2')], errors='ignore')
        E1_keys = [col for col in dft.columns if col.endswith('Ef1')]
        new_E_keys = [col.replace('f1', '') for col in E1_keys]
        dft.rename(columns=dict(zip(E1_keys, new_E_keys)), inplace=True)
        dfv.rename(columns=dict(zip(E1_keys, new_E_keys)), inplace=True)

        # Making the samples anonymous
        p_alt = get_data_path('interim')
        p_pro = get_data_path('processed')

        name_cols = ['upp_low', 'temp_f_id', 'name f1', 'name f2','tau']

        dfvmap = dfv[name_cols]
        dftmap = dft[name_cols]

        p_map_v = p_alt / 'valid_names.csv'
        p_map_f = p_alt / 'train_names.csv'

        store_csv(dfvmap, p_map_v)
        store_csv(dftmap, p_map_f)

        name_cols.remove('tau') 

        del_cols = name_cols + ['ya', 'f_id']
        del_cols.remove('temp_f_id')
        # For dft we'll use temp_f_id for folding
        dft.drop(columns=del_cols, inplace=True, errors='ignore')
        dfv.drop(columns=del_cols + ['temp_f_id'], inplace=True, errors='ignore')
    
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
            _safe_log_metrics(metrics)
            _safe_log_artifact(p_map_f)
            _safe_log_artifact(p_map_v)
        except Exception as e:
            logger.warning(f"Could not log split_trainval metrics/artifacts: {e}")

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
        """
        Note: This also contains one-hot columns that simultaneously 
        serve as a flag. Deleting these columns therefore means also 
        deleting all flags. 
        """

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
        """Symmetrize the dataset with MLflow logging."""
        if for_svd and pred_only:
            raise ValueError("pred_only=True is not supported when for_svd=True; SVD requires train/valid/pred.")

        suffix = 'svd' if for_svd else 'symm'
        
        try:
            from src.utils.mlflow_pipeline import get_pipeline_manager, substage_run
            pipeline_manager = get_pipeline_manager()
            in_pipeline = pipeline_manager is not None
        except ImportError:
            in_pipeline = False
        
        if in_pipeline:
            with substage_run("data_transformation",
                            transformation_type="symmetrize",
                            for_svd=for_svd,
                            pred_only=pred_only):
                return self._do_symmetrize(for_svd, pred_only, suffix)
        else:
            try:
                mlflow.set_experiment(f"symmetrize_{suffix}_standalone")
            except Exception as e:
                logger.warning(f"MLflow setup failed: {e}")
            
            with mlflow.start_run(run_name=f"symmetrize_{suffix}_standalone"):
                return self._do_symmetrize(for_svd, pred_only, suffix)
    
    def _do_symmetrize(self, for_svd, pred_only, suffix):
        """Internal method to perform symmetrization with MLflow logging."""
        try:
            _safe_log_params({"for_svd": bool(for_svd), "pred_only": bool(pred_only)})
        except Exception as e:
            logger.warning(f"Could not log symmetrize params: {e}")
        # Main logic (always run regardless of logging success)
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

        # oh_cols contains one_hot_columns that can also figure as flags, but del_1h only contains pure one-hot
        oh_cols = self.get_1h_columns()
        cols1 = [col for col in data[0].columns if col.endswith('f1') and col not in oh_cols]
        cols2 = [col.replace('f1', 'f2') for col in cols1]

        before_cols = [data[i].shape[1] for i in range(len(data))]
        for k in range(len(data)):
            
            # Set nans to 0 when (anti-)symmetrizing for the svd.
            dfk = data[k].fillna(0) if for_svd else data[k]

            dfk[cols1], dfk[cols2] = (
                (dfk[cols1].values + dfk[cols2].values) / np.sqrt(2),
                (dfk[cols1].values - dfk[cols2].values) / np.sqrt(2),
            )

            if not for_svd:
                dfk['weightclass'] = dfk.loc[:, 'Catch weight':'Heavyweight'].to_numpy().argmax(axis=1)
                dfk = dfk.drop(columns=del_1h)

            data[k] = dfk

        logger.info('Dataset (anti-)symmetrized.')

        if not for_svd:
            if pred_only:
                out_p = self.get_path(tvp='pred', which1='symm')
                store_csv(data[0], out_p)
                _safe_log_artifact(out_p)
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
                _safe_log_metrics(metrics)
            except Exception as e:
                logger.warning(f"Could not log symmetrize metrics: {e}")
            return data
        
    
    def _standardize(self,which=''):
        logger.info('Standardizing data...') 
        in_pipeline = _check_pipeline_context()
        
        if not in_pipeline:
            try:
                mlflow.set_experiment("standardize_standalone")
            except Exception as e:
                logger.warning(f"MLflow setup failed: {e}")
            
            with mlflow.start_run(run_name="standardize_standalone"):
                return self._do_standardize(which)
        else:
            return self._do_standardize(which)
    
    def _do_standardize(self, which=''):
        """Internal method to perform standardization with MLflow logging."""
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
            logger.warning('temp_f_id not found in standardized training data')

        logger.info('Standardization complete.')
        try:
            _safe_log_metrics({
                "std.train.rows": int(dft.shape[0]),
                "std.train.cols": int(dft.shape[1]),
                "std.valid.rows": int(dfv.shape[0]),
                "std.valid.cols": int(dfv.shape[1]),
                "std.pred.rows": int(dfp.shape[0]),
                "std.pred.cols": int(dfp.shape[1])
            })
        except Exception as e:
            logger.warning(f"Could not log standardization metrics: {e}")

        return [dft, dfv, dfp]

    def do_svd(self, k = 204, plot_sv = False): 
        try:
            from src.utils.mlflow_pipeline import get_pipeline_manager, substage_run
            pipeline_manager = get_pipeline_manager()
            in_pipeline = pipeline_manager is not None
        except ImportError:
            in_pipeline = False
        
        if in_pipeline:
            with substage_run("data_transformation",
                            transformation_type="svd",
                            k=k,
                            plot_sv=plot_sv):
                self._do_svd(k, plot_sv)
        else:
            try:
                mlflow.set_experiment("svd_standalone")
            except Exception as e:
                logger.warning(f"MLflow setup failed: {e}")
            
            with mlflow.start_run(run_name="svd_standalone"):
                self._do_svd(k, plot_sv)
    
    def _do_svd(self, k, plot_sv):
        """Internal method to perform SVD with MLflow logging."""
        try:
            _safe_log_params({"k": int(k), "plot_sv": bool(plot_sv)})
        except Exception as e:
            logger.warning(f"Could not log SVD params: {e}")
        # Main logic (always run regardless of logging success)
        data = self.symmetrize(for_svd=True)
        dft = data[0]

        if 'temp_f_id' not in dft.columns:
            print('before svd', 5/0)

        exclude_cols = ['result', 'xp score_f1', 'xp_score_f2', 'temp_f_id']
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
        arrt = (arrt @ V.T)[:, :k]
        dft_svd = pd.DataFrame(arrt, columns=[f's{si}' for si in range(s_trunc.shape[0])])
        dft_svd[exclude_cols] = dft.copy()[exclude_cols].values

        dfv = data[1]
        dfp = data[2]

        # Align columns of valid and pred to train
        dfv = dfv[dft.columns]
        dfp = dfp[dft.columns]

        arrv = dfv[[col for col in dfv.columns if col not in exclude_cols]].values
        arrp = dfp[[col for col in dfp.columns if col not in exclude_cols]].values

        arrv = (arrv @ V.T)[:, :k]
        arrp = (arrp @ V.T)[:, :k]

        dfv_svd = pd.DataFrame(arrv, columns=[f's{si}' for si in range(s_trunc.shape[0])])
        dfv_svd[exclude_cols] = dfv.copy()[exclude_cols].values
        dfp_svd = pd.DataFrame(arrp, columns=[f's{si}' for si in range(s_trunc.shape[0])])
        dfp_svd[exclude_cols] = dfp.copy()[exclude_cols].values

        if 'temp_f_id' not in dft_svd.columns:
            print('before svd', 5/0)
        store_csv(dft_svd, self.proc_path / 'train_svd.csv')
        store_csv(dfv_svd, self.proc_path / 'valid_svd.csv')
        store_csv(dfp_svd, self.proc_path / 'pred_svd.csv')

        # Log SVD metrics and schemas
        try:
            total_energy = float(np.sum(s**2)) if s.size else 0.0
            k_energy = float(np.sum(s_trunc**2)) if s_trunc.size else 0.0
            explained_ratio = float(k_energy / total_energy) if total_energy > 0 else 0.0
            _safe_log_metrics({
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
            plt.scatter([k] * 50, np.linspace(np.log(s)[-1], np.log(s)[0], 50), s=1, color='red')
            plt.scatter(np.linspace(0, len(s), 100), [0] * 100, s=1, color='black')
            plt.grid()
            try:
                fig_path = get_data_path('output') / 'metrics' / 'svd_singular_values.png'
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                _safe_log_artifact(fig_path)
            except Exception as e:
                logger.warning(f"Could not save/log singular value plot: {e}")
            finally:
                plt.show()

    def go_natty(self): 
        dsets = ['train', 'valid', 'pred']
        data = [self.open(dset) for dset in dsets]

        del_1h = ['male', 'normal bout', '3 rounds']
        del_1h.extend([col for col in data[0].columns if 'weight' in col])
        del_1h.extend([col for col in data[0].columns if 'has not' in col])
        wc_cols = data[0].loc[:, 'Catch weight':'Heavyweight'].columns 

        for k in range(3): 
            dfk = data[k] 
            dfk['weightclass'] = dfk[wc_cols].to_numpy().argmax(axis=1)
            dfk.drop(columns = del_1h, inplace=True)
            store_csv(dfk, f'{get_data_path("processed")}/{dsets[k]}_natty.csv')


    
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
    }

    TVP = TrainValPred(feature_sets)

    from src.data_processing.scrape_pred import scrape_pred 
    from src.data_processing.clean_pred import clean_pred 

    #scrape_pred() 
    #clean_pred() 
    #TVP.construct_pred(scrape_and_clean=False)

    #FeatureManager({'rock_paper_scissor': feature_sets['rock_paper_scissor']}, overwrite_all=True) 
    #TVP.merge_features(overwrite_feature_sets=False) 
    #TVP.show_correlations(TVP.open_merged_features()) 
    #TVP.construct_pred()
    TVP.split_trainval(last_years = 5, sample_size = 0.15) 
    #TVP.symmetrize(for_svd=False) 
    #TVP.do_svd(plot_sv = True)  

    #data = TVP.get_trainvalpred(which='svd') 
    #dftv = pd.concat([data[0], data[1]], ignore_index=True)
    #TVP.show_correlations(dftv, duplicate=False) 






         
        
