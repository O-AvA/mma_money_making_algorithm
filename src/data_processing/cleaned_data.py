"""
Opens the cleaned ufcstats.com data set and provides utilities for feature engineering.

""" 

import pandas as pd 
from loguru import logger 
from numpy import nan

from src.utils.general import open_csv, get_data_path, store_csv

class CleanedFights(pd.DataFrame):
    """
    Functions on the cleaned dataset that are mainly
    used for feature engineering but can be done on any DataFrame 
    with fighter stats.

    TO DO: 
    - Make the pipeline with subclasses or something, most are 
      parent functions. You can't 
    """

    def __init__(self,dfc=pd.DataFrame(), *args, **kwargs):
        """
        Initalizes cleaned ufc.com stats or custom DataFrame.
        """
        path = get_data_path('interim') / 'clean_ufcstats-com_data.csv' 

        if dfc.empty:
            try:
                dfc = open_csv(path) 
            except FileNotFoundError:
                logger.error(f"Cleaned data not found.")
                raise

        # This order will be the basis for all feature sets 
        sort_cols = ['tau'] if 'tau' in dfc.columns else [] 
        sort_cols = sort_cols + ['name f1'] if 'name f1' in dfc.columns else sort_cols
        dfc = dfc.sort_values(by=sort_cols) if len(sort_cols) > 0 else dfc  
        dfc.reset_index(drop=True, inplace=True)  
        super().__init__(dfc, *args, **kwargs)

    def duplicate(self, 
                  custom_cols = [], 
                  base_cols = ['tau', 'name f1', 'name f2', 'result'],
                  add_labels = True): 
        """
        Duplicates cleaned data for given custom_cols + base_cols
        and swaps fighters 1 and 2.
        Also adds new column upp_low for future unduplication. 
        Note, only custom_cols and base_cols columns and upp_low get returned. 

        Args: 
            list(str) custom_cols: 
                Column names you want to duplicate from CleanedFights.
                Please provide non-shared data with at least one fighter 
                label 1 or 2 / f1 or f2  
            list(str) base_cols: 
                Column names that automatically get duplicated.
                A bit useless in hindsight. 
            bool add_labels
                Default True, Adds columns that help with unduplicating and sorting back 
                to original order. 
        """
        dfc = self.copy()  
        base_cols = base_cols.copy()

        # Add a fight id column, will be used later for unduplicating
        dfc['temp_f_id'] = dfc.index
        base_cols.append('temp_f_id') 

        # Avoid duplicate columns 
        custom_cols = [ccol for ccol in custom_cols if ccol not in base_cols]

        # Make sure that DataFrame doesn't contain columns that aren't in base_cols 
        # Yeah this is why its useless but I no longer know what depends on it 
        base_cols = [col for col in base_cols if col in dfc]

        cols = base_cols + custom_cols

        # Do result separately
        do_results = False 
        if 'result' in cols: 
            cols.remove('result')
            do_results = True 

        for col in cols: 
            if col not in dfc.columns:
                raise Exception(f'CleanedData DataFrame does not contain column {col}')

        # Subdividing in f1 cols, f2 cols and shared cols and
        # making sure all columns for both fighters are included.
        # and columns for both fighters are aligned. 
        cols1 = [col for col in cols if col[-1] == '1'] 
        cols2 = [col for col in cols if col[-1] == '2'] 
        scols = [col for col in cols if col not in cols1+cols2] 
   
        cols1 += [col2.replace('f2','f1') for col2 in cols2 if col2.replace('f2','f1') not in cols1]
        cols2 = [col1.replace('f1','f2') for col1 in cols1]

        # Duplicating
        d_f = {}
        for scol in scols: 
            d_f[scol] = pd.concat([dfc[scol], dfc[scol]]).reset_index(drop=True)  
        for i in range(len(cols1)):
            c1 = cols1[i]
            c2 = cols2[i]
            d_f[c1] = pd.concat([dfc[c1], dfc[c2]]).reset_index(drop=True)
            d_f[c2] = pd.concat([dfc[c2], dfc[c1]]).reset_index(drop=True)

        dff = pd.DataFrame(d_f) 

        dff['upp_low'] = pd.Series([0]*len(dfc) + [1]*len(dfc))
        
        if do_results:  
            dff['result'] = pd.concat([dfc['result'], 6-dfc['result']]).reset_index(drop=True)
            dff['result'] = dff['result'].replace(7,-1) 
        if not add_labels: 
            dff.drop(columns=['upp_low', 'temp_f_id'], inplace=True)

        # NOTE: Returning as CleanedFights automatically sorts by tau and name f1! 

        return CleanedFights(dff) 


    def get_record(self, 
                  cols, 
                  include_tau=True, 
                  ignore_current=True,
                  intervals=None
        ): 
        """
        For each fight, gets past fight data for fighter1 for 
        chosen columns from clean_ufcstats-com_data.csv in a list. 

        Args: 
            DataFrame self 
            cols list(str) 
                stats for which you want past data
            bool include_tau
                Default True, you may want it for time splicing
            bool ignore_current
                Default True, ignores current bout. 
            list(list(int)) intervals:  
                time intervals in weeks if you wanna slice the lists 
        Returns: 
            DataFrame: 
                CleanedData with columns in cols overwritten with a list 
                of column values for previous fights for each fight for 
                name f1.
                All other columns are kept as is. 

        Example: 

        df = {'name f1': [Jopie, Jopie, Jopie, Henkie], 
              'stat': [100,30,20,10] 
        } 

        Then this function returns for each stat in cols: 

        df = {'name f1': [Jopie, Jopie, Jopie, Henkie] 
              'stat': [[], [100], [100, 30], []] 
        }
        """
        logger.info("Aggregating fighter's previous fights stats...")

        dff = self.copy()  
        cols = cols+['tau'] if include_tau and 'tau' not in cols else cols

        dff_g = dff.groupby('name f1') 

        if intervals is not None: 
            # Months ago since carreer start for each bout  
            dff['wa'] = (dff_g['tau'] - dff_g['tau'].transform('min'))  
            cols.append('wa') 

        if ignore_current: 
            dff[cols] = dff_g[cols].transform(
                lambda x: [[]] + [list(x.iloc[:i]) for i in range(1,len(x))]) 
        else: 
            dff[cols] = dff_g[cols].transform(
                lambda x: [list(x.iloc[:i+1]) for i in range(len(x))])
        logger.info('Previous fight stats overwritten current fight stat') 

        if intervals is not None:
            for iv in intervals:
                time_sliced_stats = dff[cols].apply(lambda row: _time_slice_stats(row, cols, iv), axis=1)
                nkeys = [f"{iv[0]}-{iv[1]}wa{col}" for col in cols]
                dff[nkeys] = pd.DataFrame([x for x in time_sliced_stats], index=dff.index)
                dff.drop([nkeys], axis=1, inplace=True)   
                logger.info(f'Time sliced previous fight stats on intervals {str(intervals)} (weeks), dropped unsliced fight stats lists.')


        logger.info("Aggregating fighter's previous fights stats complete.")
        return dff  

    def unduplicate(self):
        # For upp_low = 0, sort by ['tau', 'name f1'] is equivalent to 
        # sort by 'temp_f_id'

        dff = self.copy() 
        dff.drop(columns=[col for col in dff.columns if col[-2:] == 'f2'], inplace=True)  

        f1_cols = [col for col in dff.columns if 'f1' in col] 
        f2_cols = [col.replace('f1','f2') for col in f1_cols] 

        df_upp = dff[dff['upp_low'] == 0].copy() 
        df_upp.sort_values(by='temp_f_id', inplace=True)  
        df_upp.reset_index(drop=True, inplace=True) 

        df_low = dff[dff['upp_low'] == 1].copy() 
        df_low = df_low[f1_cols + ['temp_f_id']] 
        df_low.sort_values(by='temp_f_id', inplace=True)
        df_low.reset_index(drop=True, inplace=True)

        if len(df_upp) != len(df_low):
            logger.info(f'{len(df_upp)}, {len(df_low)}') 

        df_low = df_low.rename(columns = dict(zip(f1_cols, f2_cols)))

        #df_low = df_low.drop(columns='result') if 'result' in df_low.columns else df_low 

        dff = pd.concat([df_upp, df_low], axis=1)  
        dff = dff.drop(columns=['upp_low', 'temp_f_id'])
        #dff = dff.sort_values(by=['tau','name f1'])
        dff.reset_index(drop=True, inplace=True)

        return CleanedFights(dff) 



    def _time_slice_stats(bout, cols, interval): 
        """ 
        For a row in CleanedData, access lists in bout[cols] 
        and truncates fight stats for fights outside of interval 
        """
        t1, t2 = interval
        arr = np.array([bout[col] for col in cols], dtype=float)
        wa_idx = cols.index('wa')
        mask = (arr[wa_idx] > t1) & (arr[wa_idx] <= t2)
        if not np.any(mask):
            return [[] for _ in cols]

        sliced = arr[:, mask]

        return [sliced[i].tolist() for i in range(len(cols))]

            













        


