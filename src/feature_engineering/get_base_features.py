"""
Name, age, reach weightclass etc.

TO DO: 
    Find a cleaner way to process upcoming fights, bit messy like this. 
"""

import pandas as pd 
from numpy import nan 
from loguru import logger 

from src.data_processing.cleaned_data import CleanedFights 
from src.data_processing.upcoming_fights import UpcomingFights 
from src.utils.general import (open_csv, store_csv, get_data_path, stat_names) 

def get_base_features(
        process_upcoming_fights=False
    ): 
    """
    Extracts basic features from cleaned ufcstats.com dataset and
    makes them ready for training and/or to be used for further 
    feature engineering. 
    """
    logger.info('Getting base features...') 

    # Open cleaned ufcstats.com data   
    dfo = CleanedFights()
    if pd.isna(dfo['female']).any(): 
        print(dfo[pd.isna(dfo['female'])][['name f1','tau','WEIGHTCLASS']])
        print('in data_processing...', 5/0) 
    dfb = pd.DataFrame() 
    base_feat_path = get_data_path('features') / 'base_features.csv' 

    # Remove unnecessary columns 
    snames = stat_names(rounds=True,avgs=True,stds=False,fighters=True)
    raw_stats_cols = open_csv(get_data_path('raw') / 'ufc_fight_stats.csv').columns.tolist()
    raw_stats_cols = [col for col in raw_stats_cols if col in dfo.columns]   
    dfo.drop(columns=snames + raw_stats_cols, inplace=True)

    # One-hot encoded columns 
    one_hot_cols = [
            'title bout', '5 rounds', 
            *[f'round {ri} fought' for ri in range(1,6)] 
    ] 
    # Continuous feature columns  
    cont_cols = [f'r{ri} time' for ri in range(1,6)] 
    # Ccolumns we can take straight from CleanedFights
    wc_cols = [col for col in dfo.columns if 'weight' in col] 
    fcols = ['age f1', 'height f1', 'reach f1','tau', 'name f1','normal bout', 
             '3 rounds','female','male'] + wc_cols 

    # Duplicate data and swap fighter 1 and 2 in dfo
    dfo = dfo.duplicate(custom_cols = one_hot_cols + cont_cols + fcols)
    fcols.extend(['title bout', '5 rounds']) 
            
    ###########################################################################
    if process_upcoming_fights:
        # Use already-cleaned prediction data (clean_pred.py) for demographics/flags.
        # Only map history-dependent features from base_features.csv.
        dfp = UpcomingFights().open_clean()
        # Work on per-fighter rows to attach f1-style features, then unduplicate back to bouts
        dfp = CleanedFights(dfp).duplicate(base_cols=dfp.columns.tolist())

        # Load last-known historical aggregates per fighter from base_features.csv
        dfb_hist = open_csv(base_feat_path)
        if dfb_hist.empty:
            logger.warning('base_features.csv is empty; treating all upcoming fighters as debutants for history features.')
            dfb_hist = pd.DataFrame(columns=['name f1', 'tau'])
        dfb_hist = CleanedFights(dfb_hist).duplicate(base_cols=dfb_hist.columns.tolist())
        # Keep most recent row per fighter (latest tau)
        dfb_hist_last = dfb_hist.sort_values(by=['name f1', 'tau']).groupby('name f1', as_index=False).tail(1)

        # Identify unknown/debuting fighters
        known_names = set(dfb_hist_last['name f1'])
        unknown_mask = ~dfp['name f1'].isin(known_names)
        n_unknown = int(unknown_mask.sum())
        if n_unknown > 0:
            logger.warning(f'Encountered {n_unknown} debuting/unknown fighters for base history; filling neutral defaults.')

        # Build list of columns to map from history
        hist_has_cols = [f'has {col} f1' for col in one_hot_cols]
        hist_has_not_cols = [f'has not {col} f1' for col in one_hot_cols]
        hist_tot_onehot_cols = [f'tot {col} f1' for col in one_hot_cols]
        hist_tot_time_cols = [f'tot {col} f1' for col in cont_cols]
        hist_extra_cols = ['weeks active f1', 'last fight wa f1']

        cols_to_map = hist_has_cols + hist_has_not_cols + hist_tot_onehot_cols + hist_tot_time_cols + hist_extra_cols

        # Initialize missing columns to avoid KeyErrors on merge/map when base_features schema changes
        for c in cols_to_map:
            if c not in dfb_hist_last.columns:
                dfb_hist_last[c] = pd.NA

        # Map historical aggregates onto upcoming per-fighter rows
        name_to_vals = {c: dict(zip(dfb_hist_last['name f1'], dfb_hist_last[c])) for c in cols_to_map}
        for c in cols_to_map:
            dfp[c] = dfp['name f1'].map(name_to_vals[c])

        # Default-fill for debutants/unknowns
        for c in hist_has_not_cols:
            dfp[c] = dfp[c].fillna(1)
        for c in hist_has_cols + hist_tot_onehot_cols + hist_tot_time_cols:
            dfp[c] = dfp[c].fillna(0)
        for c in hist_extra_cols:
            dfp[c] = dfp[c].fillna(0)

        # Return to bouts and append features
        dfp = CleanedFights(dfp).unduplicate()
        UpcomingFights().append_features(dfp, drop_name_cols=False) 
        return 0  

    #########################################################################
    # Handle feature-ready columns. 
    dfb[fcols] = dfo[fcols] 

    # For future duplication. have to do handle this differently later. 
    dfb[['upp_low','temp_f_id']] = dfo[['upp_low','temp_f_id']]
   
    # Process non-ready featuers
    # Has fought title bout, has fought round ri etc.. 
    
    # Get previous fight data for each fight. 
    dfo = dfo.get_record(one_hot_cols + cont_cols)   

    for col in one_hot_cols:
        dfb[f'has {col} f1'] = dfo[col].apply(
            lambda x: any(xi == 1 for xi in x)
        ).astype(int) 

        dfb[f'has not {col} f1'] = dfo[col].apply(
            lambda x: all(xi == 0 for xi in x)
        ).astype(int) 

        # I should do this with transform 
        dfb[f'tot {col} f1'] = dfo[col].apply(
            lambda x: sum([xi == 1 for xi in x])
        )

    for col in cont_cols: 
        dfb[f'tot {col} f1'] = dfo[col].apply(
                lambda x: sum(x)
        )


    # Time since carreer start (weeks) 
    dfb.sort_values(by=['tau','name f1'], inplace=True) 
    min_tau = dfb.groupby('name f1')['tau'].min()
    dfb['weeks active f1'] = dfb['tau'] - dfb['name f1'].map(min_tau)


    # Take last value from list 
    dfo['previous tau'] = dfo['tau'].apply(lambda x: x[-1] if len(x) > 0 else nan) 
    dfo['previous tau'] = dfo['previous tau'].fillna(dfb['tau']) 
    #wa_map = dict(zip(dfo['name f1'], dfo['previous tau'])) 
    dfb['last fight wa f1'] = dfb['tau'] - dfo['previous tau'] 

    # Unduplicate
    dfb = CleanedFights(dfb).unduplicate() 

    store_csv(dfb, base_feat_path) 

if __name__ == '__main__': 
    get_base_features()





















