import pandas as pd  

import sys 
import importlib
from loguru import logger

from src.utils.general import (open_csv, store_csv, get_data_path, get_src_path)  
sys.path.append(str(get_src_path('feature_engineering')))

class FeatureManager: 

    def __init__(self, feature_sets=None, overwrite_all=False):  
        """
        Merges chosen feature sets. 
        Mostly helps for constructing final feature set (equivalently, 
        the training and validation set, 'trainval') and dataset to predict ('pred').  
        """
        self.path = get_data_path('interim') / 'chosen_features_merged.csv'
        self.feature_sets = feature_sets or {}
        self.overwrite_all = overwrite_all
        self._merged_features = None

        if feature_sets is not None:
            if overwrite_all: 
                logger.info(f'Opening all chosen feature sets {feature_sets}') 

            # Make sure 
            feature_sets = [fset if fset.endswith('.csv') else fset + '.csv' for fset in feature_sets]

            # Paths of feature engineering modules and feature sets 
            fdir = get_data_path('features')

            dffs = [] 

            for fname, fparams in self.feature_sets.items():  
                # Get path of feature set 
                fset_path = fdir / f'{fname}.csv'

                # Run module if feature set doesn't exist or overwrite 
                if overwrite_all: 
                    logger.warning(f'Feature set {fname} not found. Creating it now.')
                if not fset_path.exists() or overwrite_all:  
                    self.create_feature_set(fname, fparams, upcoming_fights=False)
                dff = open_csv(fset_path) 
                #dff.drop(columns=[col for col in dff.columns if col in ['name f1', 'name f2']], inplace=True) 
                dffs.append(dff) 

            self.fsets = {fset[:-4]: dff for fset, dff in zip(feature_sets,dffs)}
        else:
            logger.info('Opening merged feature set') 
            merged_features = open_csv(self.path) 
            self.merged_features = merged_features 

    def merge_features(self,save=True):  
        dffs = list(self.fsets.values())

        """
        # Remove potential duplicate columns 
        all_features = [] 
        for k in range(len(dffs)): 
            use_cols = [col for col in dffs[k].columns if col not in all_features]
            all_features.extend(use_cols) 
            if len(use_cols) != len(dffs[k].columns): 
                logger.warning(f'{self.fsets.keys()[k]} overlaps with {self.fsets.keys()[:k]}')
            dffs[k] = dffs[k][use_cols]
        print(all_features)  
        """

        chosen_features = pd.concat(dffs, axis=1) 
        chosen_features = chosen_features.loc[:, ~chosen_features.columns.duplicated()]
        
        if save: 
            store_csv(chosen_features, self.path) 
        return chosen_features

    @staticmethod 
    def create_feature_set(fname, fparams, upcoming_fights=False):  
        mdir = get_src_path('feature_engineering')
        func_name = f'get_{fname.replace(".csv","")}'

        fmod_path = f"{func_name}"
        fmod = importlib.import_module(fmod_path)
        func = getattr(fmod, func_name) 
        func(**fparams, process_upcoming_fights = upcoming_fights)

    @property
    def merged_features(self):
        return self._merged_features

    @merged_features.setter
    def merged_features(self, value):
        self._merged_features = value










