import pandas as pd 
from loguru import logger 

from src.utils.general import (get_data_path, open_csv, store_csv, creaopen_file)  
from src.data_processing.cleaned_data import CleanedFights 

class UpcomingFights:
    def __init__(self): 
        clean_path = get_data_path('interim') / 'pred_clean.csv'
        processed_path = get_data_path('processed') / 'pred.csv'
        self.clean_path = clean_path 
        self.processed_path = processed_path 

    def open_clean(self): 
        dfp_clean = CleanedFights(open_csv(self.clean_path)) 
        #dfp_clean.sort_values(by=['tau', 'name f1'], inplace=True)  
        return CleanedFights(dfp_clean) 

    def append_features(self, dfp_feats, drop_name_cols=True): 
        dfp = creaopen_file(self.processed_path) 
        dfp_feats = dfp_feats.sort_values(by=['tau', 'name f1']).reset_index(drop=True)  
        dfp_feats.drop(columns=[col for col in dfp_feats.columns if col in dfp.columns], 
                       inplace=True
        )

        if drop_name_cols: 
            name_cols = ['name f1', 'name f2']


            dfp_feats.drop(columns=[col for col in name_cols if col in dfp_feats.columns],
                           inplace=True
            )

        logger.info(f'Appending {len(dfp_feats)} features to pred...') 
        dfp = pd.concat([dfp, dfp_feats], axis=1)

        store_csv(dfp, self.processed_path)     


    def open_processed(self, add_names=False):
        names = self.open_clean()[['name f1', 'name f2']] 
        dfp = open_csv(self.processed_path) 

        for i in [1,2]: 
            name_fi = f'name f{i}'
            if name_fi not in dfp.columns: 
                dfp[name_fi] = names[name_fi] 

        return CleanedFights(dfp) 



            

