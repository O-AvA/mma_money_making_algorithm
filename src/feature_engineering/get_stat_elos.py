"""
Calculates elos for striking and Control Time 
"""

import pandas as pd
import numpy as np
from loguru import logger 

from src.utils.general import (open_csv, store_csv, Dict, 
                               get_data_path, stat_names,
                               expectancy, compute_elo) 
from src.data_processing.cleaned_data import CleanedFights 
from src.data_processing.upcoming_fights import UpcomingFights 

def get_stat_elos(per_round=True,
                exact_score = True,
                process_upcoming_fights=False, 
                which_K = 'log',
                always_update=False):
    """
    Elo score for the relative amount of strikes thrown for 
    each fighter. 

    Args: 
        per_round bool:
            True: Calculate an elo for each round separately
            False: Calculate an elo for stats averaged over rounds fought 
        exact_score bool: 
            True: score_f1 = stat_f1/(stat_f1 + stat_f2) 
            False: score_f1 = 1 if stat_f1 > stat_f2 
        process_upcoming_fights: 
            True: Calculate elos for the prediction data set. 
    """
    fname = 'stat_elos_per_round.csv' if per_round else 'stat_elos_round_averages.csv' 
    stat_elos_path = get_data_path('features') / fname 

    dfo = CleanedFights()

    rounds = True if per_round else False 
    avgs = True if not per_round else False 
    stat_labels = stat_names(which=['','l'],
                        rounds=rounds, 
                        extras=True, 
                        avgs=avgs,
                        stds = False,
                        fighters=True
    )
    # Ensure no SA/KD slip through if config changes elsewhere

    if process_upcoming_fights: 
        # Forward params so upcoming can handle unknown fighters robustly
        upcoming_stat_elos(stat_elos_path, stat_labels, exact_score=exact_score, always_update=always_update)
        return 0 

    # Construct DataFrame with scores 
    # Get f1, f2 stats (stat_names() already aligns them) 
    stat_labels1 = [stat_label for stat_label in stat_labels if stat_label[-2:] == 'f1']
    stat_labels2 = [stat_label1.replace('f1','f2') for stat_label1 in stat_labels1] 
    df1 = dfo[stat_labels1] 
    df2 = dfo[stat_labels2]

    if exact_score: 
        scores1 = df1.values / (df1.values + df2.values) 
        dfs = pd.DataFrame(scores1, 
                           index = df1.index, 
                           columns = stat_labels1
        )
        if always_update: 
            scores2 = df2.values / (df1.values + df2.values) 
            dfs[df2.columns] = scores2 
            dfs = dfs.fillna(0) 
        else: 
            dfs[df2.columns] = 1-dfs
    else:
        vals1 = df1.values.copy()
        vals2 = df2.values.copy() 

        dfs = pd.DataFrame(np.nan,
                           index = df1.index, 
                           columns = stat_labels1
        )

        dfs[vals1 < vals2] = 0  
        dfs[(vals1 != 0) & (vals2 != 0) & (vals1 == vals2)] = 0.5 
        dfs[vals1 > vals2] = 1 
        #dfs[(vals1 == 0) & (vals2 == 0)] = np.nan 

        dfs[df2.columns] = 1 - dfs 
        dfs = dfs.fillna(0) if always_update else dfs 

    del df1,df2 

    # Get the necessary parameters (K)   
    dfep = open_csv(get_data_path('features') / 'elo_params.csv') 

    # Current elos 
    d_R = Dict()

    # Combined:
    dfo = pd.concat([dfo[['name f1', 'name f2','tau']],
                     dfep[[col for col in dfep.columns if 'K' in col]],
                     dfs
                     ], 
                    axis = 1
    )
    

    # Get feature names 
    feature_names = [] 
    for fi in [1,2]: 
        for flabel in [f'Rf{fi}', f'Ef{fi}', f'R_REMOVE_f{fi}']:
            for stat_label1 in stat_labels1: 
                feature_names.append(stat_label1.replace('f1', flabel)) 

    # Features 
    d_e = {}

    logger.info(f'Starting iterating over fights to process {len(feature_names)} striking elo features...')
    n_feats = len(feature_names) 
    for ib, bout in dfo.iterrows():
        name1 = bout['name f1'] 
        name2 = bout['name f2'] 
        scores1 = bout[stat_labels1]  
        scores2 = bout[stat_labels2]

        if per_round:
            # Build per-round K mappings for f1 and f2
            Kmap1 = {
                'r1': bout.get(f'K{which_K} f1'),
                'r2': bout.get(f'K{which_K}r2 f1'),
                'r3': bout.get(f'K{which_K}r3 f1'),
                'r4': bout.get(f'K{which_K}r4 f1'),
                'r5': bout.get(f'K{which_K}r5 f1'),
            }
            Kmap2 = {
                'r1': bout.get(f'K{which_K} f2'),
                'r2': bout.get(f'K{which_K}r2 f2'),
                'r3': bout.get(f'K{which_K}r3 f2'),
                'r4': bout.get(f'K{which_K}r4 f2'),
                'r5': bout.get(f'K{which_K}r5 f2'),
            }

            def _round_tag(idx: str) -> str:
                # idx examples: 'SSr1f1', 'SSlr3f1'
                for r in ['r1', 'r2', 'r3', 'r4', 'r5']:
                    if r in idx:
                        return r
                return 'r1'

            K1 = scores1.index.to_series().map(lambda x: Kmap1[_round_tag(x)]).to_numpy()
            K2 = scores2.index.to_series().map(lambda x: Kmap2[_round_tag(x)]).to_numpy()
        else: 
            K1 = bout[f'K{which_K} f1'] 
            K2 = bout[f'K{which_K} f2'] 
                                             
        scores1.index = scores1.index.str.replace('f1','') 
        scores2.index = scores2.index.str.replace('f2','') 

        # Get current elos
        R1 = pd.Series(scores1.index.map(
            lambda x: d_R[name1+x] if name1+x in d_R.keys() else 1500)
        )  
        R2 = pd.Series(scores2.index.map(
            lambda x: d_R[name2+x] if name2+x in d_R.keys() else 1500)
        )

        E1 = expectancy(R1, R2) 
        E2 = 1-E1

        nR1 = compute_elo(R1, E1, scores1, K1).values 
        nR2 = compute_elo(R2, E2, scores2, K2).values 
        
        # No rating mask 
        noR1 = scores1.index.map(lambda x: name1+x not in d_R.keys()).values
        noR2 = scores2.index.map(lambda x: name2+x not in d_R.keys()).values 

        # No strikes mask 
        noS1 = scores1.isna().values 
        noS2 = scores2.isna().values  

        # Note that, if exact_score = True and S1 = S2 = 0, then scores1 = scores2 = nan  
        # and thus nR1 = nan. This will then be forwarded even though fighter may already
        # have a rating. Thus we need to replace nR1, nR2 with previously known scores if  
        # fighter has rating.
        nR1[~(noR1) & noS1] = scores1.index[~(noR1) & noS1].map(lambda x: d_R[name1 + x])
        nR2[~(noR2) & noS2] = scores2.index[~(noR2) & noS2].map(lambda x: d_R[name2 + x])

        # We also may wanna replace initial ratings and corresponding expectancies 
        # with the average later when standardizing the dataset.
        R1[(R1==1500) & noR1] = np.nan
        R2[(R2==1500) & noR2] = np.nan
        E1[(E1==0.5) & noR1] = np.nan 
        E2[(E2==0.5) & noR2] = np.nan 

        # We only wanna update current ratings if nR1, nR2 is not nan 
        nonan_mask1 = ~(noR1 & noS1)
        nonan_mask2 = ~(noR2 & noS2)
        nR1_nonan = nR1[nonan_mask1] 
        nR2_nonan = nR2[nonan_mask2] 
        scores1_nonan = scores1[nonan_mask1] 
        scores2_nonan = scores2[nonan_mask2]

        for k in range(len(nR1_nonan)): 
            d_R[name1 + scores1_nonan.index[k]] = nR1_nonan[k] 
        for k in range(len(nR2_nonan)): 
            d_R[name2 + scores2_nonan.index[k]] = nR2_nonan[k] 

        # Append features 
        features = pd.concat([R1, E1, pd.Series(nR1), 
                              R2, E2, pd.Series(nR2)]).reset_index(drop=True) 
        d_e[ib] = features 

    d_e = np.array(list(d_e.values()))

    dfe = pd.DataFrame(columns=feature_names, data=d_e)
    dfe[['name f1', 'name f2', 'tau']] = dfo[['name f1', 'name f2', 'tau']]
    store_csv(dfe, stat_elos_path)  

def upcoming_stat_elos(stat_elos_path, stat_labels, exact_score=True, always_update=False):
    # Get columns for matching and sortin from pred 
    dfp = UpcomingFights().open_clean()
    dfp = CleanedFights(dfp).duplicate(base_cols =['name f1', 'name f2', 'tau'])
    dfp = dfp.sort_values(by=['name f1']).reset_index(drop=True)

    # Open main feature set and get final elos  
    dfe = open_csv(stat_elos_path)
    final_elos = [col for col in dfe.columns if '_REMOVE_' in col and col[-2:] != 'f2']

    # Guard: nothing to merge if there are no final elo columns
    if len(final_elos) == 0:
        logger.warning('No final elo columns found in feature file for upcoming fights. Skipping.')
        return

    dfe = CleanedFights(dfe).duplicate(custom_cols = final_elos, 
                                         base_cols = ['name f1', 'tau']
    )
    # Keep the most recent per fighter
    dfe = dfe.sort_values(by=['name f1','tau']).groupby('name f1', as_index=False).tail(1)

    # Left-join to keep all upcoming fighters, including debutants
    dfp = dfp.merge(dfe[['name f1'] + final_elos], on='name f1', how='left')

    # Report unknown fighters (no historical elo found)
    known_names = set(dfe['name f1'])
    unknown_mask = ~dfp['name f1'].isin(known_names)
    n_unknown = int(unknown_mask.sum())
    if n_unknown > 0:
        logger.warning(f'Encountered {n_unknown} unknown debuting fighters in upcoming set. '
                       f'Using neutral rating=1500 for expectancy; '
                       f'{"also filling R features" if always_update else "leaving R features as NaN"}.')

    # Map final elos (R_REMOVE_) to live R feature names (Rf1)
    dfp_elo_cols = [elo.replace('_REMOVE_','') for elo in final_elos]
    rename_map = {src: tgt for src, tgt in zip(final_elos, dfp_elo_cols)}
    dfp.rename(columns=rename_map, inplace=True)

    # If configured, fill missing R with neutral 1500; else keep NaN in R
    base_rating = 1500.0
    if always_update:
        dfp[dfp_elo_cols] = dfp[dfp_elo_cols].fillna(base_rating)

    # Unduplicate back to fights format (will produce Rf1 and Rf2 columns)
    dfp = CleanedFights(dfp).unduplicate() 
    
    # Append expectancies using neutral fill-ins for any missing R
    dfp_exp_cols1 = [elo.replace('R','E') for elo in dfp_elo_cols] 
    dfp_exp_cols2 = [elo.replace('f1','f2') for elo in dfp_exp_cols1]
    dfp_elo_cols2 = [elo.replace('f1','f2') for elo in dfp_elo_cols] 

    # Compute E with temporary neutral substitutions; do not overwrite R unless always_update=True
    r1_tmp = dfp[dfp_elo_cols].copy().fillna(base_rating).values
    r2_tmp = dfp[dfp_elo_cols2].copy().fillna(base_rating).values
    dfp[dfp_exp_cols1] = expectancy(r1_tmp, r2_tmp)
    dfp[dfp_exp_cols2] = 1 - dfp[dfp_exp_cols1]

    UpcomingFights().append_features(dfp) 



if __name__ == '__main__': 
    get_stat_elos(per_round=False,
                  exact_score=True)  
    


    from src.feature_engineering.get_elo_params import get_elo_params
    
    wl_elos_path = get_data_path('features') / 'stat_elos.csv'
    dfe = open_csv(wl_elos_path) 
    #dfe.drop(columns=[col for col in dfe.columns if 'REM' in col],inplace=True)
    print(dfe[[col for col in dfe.columns if 'REM' in col]]) 

    dfe['result'] = CleanedFights()['result']
    dfe['tau'] = CleanedFights()['tau']
    dfe = dfe[dfe['result'] >= 0] 
    dfe['result'] = dfe['result'].map({0:0, 1: 0, 2: 0, 
                                       3: 1, 
                                       4: 2, 5: 2, 6: 2
                                       }) 

    dfe = CleanedFights(dfe).duplicate(base_cols=dfe.columns.tolist())     

    dfe.drop(columns=dfe.columns[dfe.columns.isin(['tau','upp_low','temp_f_id'])], inplace=True)
    correlations = dfe.corr()['result'].abs().sort_values(ascending=False)
    for col, corr in correlations.items():
        print(f"{col}: {corr}")


    


    # Bux fixing 
    nR1_nan = nR1[nan_mask1] 
    nR2_nan = nR2[nan_mask2] 
    if len(nR1_nan) > 0 and pd.isna(nR1_nan).any():
        raise Exception('wtf broer, sommige zijn nan')  
    if len(nR2_nan) > 0 and pd.isna(nR2_nan).any():
        raise Exception('wtf broer, sommige zijn nan')














