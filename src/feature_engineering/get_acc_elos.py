import pandas as pd 
import numpy as np 
from loguru import logger

from src.utils.general import * 
from src.data_processing.cleaned_data import CleanedFights 
from src.data_processing.upcoming_fights import UpcomingFights




def get_acc_elos(per_round=False, 
                 which_K = 'log', 
                 process_upcoming_fights=False,
                 always_update=False):  
    # Output file mirrors stat_elos naming
    fname = 'acc_elos_per_round.csv' if per_round else 'acc_elos_round_averages.csv'
    acc_elos_path = get_data_path('features') / fname

    dfo = CleanedFights()
    dfo = dfo.reset_index(drop=True)

    # Get necessary stat labels. Use round labels or averages depending on per_round
    use_avgs = not per_round
    A1_labels = stat_names(which=['A'], rounds=per_round, extras=False, avgs=use_avgs, fighters=[1]) 
    D1_labels = stat_names(which=['D'], rounds=per_round, extras=False, avgs=use_avgs, fighters=[1]) 
    A2_labels = stat_names(which=['A'], rounds=per_round, extras=False, avgs=use_avgs, fighters=[2]) 
    D2_labels = stat_names(which=['D'], rounds=per_round, extras=False, avgs=use_avgs, fighters=[2]) 

    if process_upcoming_fights: 
        # Forward always_update so unknown fighters can be handled consistently
        upcoming_acc_elos(acc_elos_path, A1_labels + D1_labels, always_update=always_update)
        return 0
         

    f1_labels = A1_labels + D1_labels
    f2_labels = D2_labels + A2_labels

    # Get actual accuracies. 'score' is in this case just the accuracy.
    f1_scores = dfo[f1_labels]
    f2_scores = dfo[f2_labels] 

    # Get K parameters and always include K columns for per-row selection
    dfep = open_csv(get_data_path('features') / 'elo_params.csv')
    dfep = dfep.reset_index(drop=True)
    df_k = dfep[[col for col in dfep.columns if 'K' in col]]

    # Current elos 
    d_R = {} 

    # New elos 
    d_e = {} 

    # Aggregate all necessary values 
    parts = [dfo[['name f1', 'name f2', 'tau']], f1_scores, f2_scores, df_k]
    dfa = pd.concat(parts, axis=1)

    for ib, (_, bout) in enumerate(dfa.iterrows()):
        name1 = bout['name f1'] 
        name2 = bout['name f2'] 

        f1_scores_b = bout[f1_labels]
        f2_scores_b = bout[f2_labels]

        if per_round:
            # Build per-round K arrays matching labels
            def _round_tag(idx: str) -> str:
                for r in ['r1','r2','r3','r4','r5']:
                    if r in idx:
                        return r
                return 'r1'
            # Map label -> K using dfep row
            def _k_for_round(rtag: str, f: int):
                if rtag == 'r1':
                    return bout[f'K{which_K} f{f}']
                else:
                    return bout[f'K{which_K}{rtag} f{f}']
            K1_b = pd.Series([_k_for_round(_round_tag(lbl), 1) for lbl in f1_labels])
            K2_b = pd.Series([_k_for_round(_round_tag(lbl), 2) for lbl in f2_labels])
        else:
            # scalar K for averages; expand to vector length
            K1_b = pd.Series([bout[f'K{which_K} f1']]*len(f1_labels))
            K2_b = pd.Series([bout[f'K{which_K} f2']]*len(f2_labels))

        # Labels for current rating 
        f1_Rlabels = f1_scores.columns.str.replace('f1',name1)
        f2_Rlabels = f2_scores.columns.str.replace('f2',name2)

        # Get current rating 
        R1 = f1_Rlabels.map(lambda x: d_R[x] if x in d_R.keys() else 1500) 
        R2 = f2_Rlabels.map(lambda x: d_R[x] if x in d_R.keys() else 1500)

        # Get expectancy
        E1 = expectancy(R1, R2) 
        E2 = 1-E1 

        # Compute final elos
        nR1 = compute_elo(R1, E1, f1_scores_b, K1_b) 
        nR2 = compute_elo(R2, E2, f2_scores_b, K2_b) 


        # We want to keep elos of fighters without elos (no attempted strikes and/or round not fought) 
        # equal to nan to exlcude all the 1500s from standardizing before doing SVD. 
        # So we will update out current elo reference d_R without nans 
        # and the actual features d_e with nans. 
        update_mask1 = f1_Rlabels.map(lambda x: x not in d_R.keys()).values 
        update_mask2 = f2_Rlabels.map(lambda x: x not in d_R.keys()).values 
        R1 = pd.Series(R1) 
        R2 = pd.Series(R2) 
        E1 = pd.Series(E1) 
        E2 = pd.Series(E2) 

        R1[update_mask1] = np.nan 
        R2[update_mask2] = np.nan  
        E1[update_mask1] = np.nan 
        E2[update_mask2] = np.nan  


        # if f1_scors_b = nan, then nR1 is nan. But it may be so that R1 != nan. 
        # So, for those cases nR1 should be equal to R1. 
        falsely_nan1 = ~update_mask1 & pd.isna(f1_scores_b).values 
        falsely_nan2 = ~update_mask2 & pd.isna(f2_scores_b).values 

        nR1 = nR1.values
        nR2 = nR2.values

        nR1[falsely_nan1] = R1.values[falsely_nan1] 
        nR2[falsely_nan2] = R2.values[falsely_nan2]

        # Update current ratings
        for k, (lab1, lab2) in enumerate(zip(f1_Rlabels, f2_Rlabels)): 
            if not pd.isna(nR1[k]): 
                d_R[lab1] = nR1[k]
            if not pd.isna(nR2[k]): 
                d_R[lab2] = nR2[k] 


        """
        # Get stats for which (still) no elo has been updated
        # So, make nan if R1 = nan and f1_scores_b = 0
        # But shouldn't it already be? Yes...   
        f1_scores_b.reset_index(drop=True,inplace=True)

        dfupd1 = pd.DataFrame(data = np.array([R1.values, nR1.values, f1_scores_b.values]).T, 
                              columns = ['R1', 'nR1', 'score1']) 
        dfupd2 = pd.DataFrame(data = np.array([R2.values, nR2.values, f2_scores_b.values]).T,
                              columns = ['R2', 'nR2', 'score2']) 
        fupdate_mask1 = dfupd1.apply(lambda x: not (pd.isna(x['R1']) and x['score1']==0),
                                     axis = 1).values
        fupdate_mask2 = dfupd2.apply(lambda x: not (pd.isna(x['R2']) and x['score2']==0),
                                     axis = 1).values

        nR1 = pd.Series(nR1)
        nR2 = pd.Series(nR2) 
        nR1[fupdate_mask1] = np.nan
        nR2[fupdate_mask2] = np.nan 
        """

        # Record features for this bout
        d_e[ib] = pd.concat([R1, E1, pd.Series(nR1),
                             R2, E2, pd.Series(nR2)]).reset_index(drop=True)
        # optional: progress logging could be added here
   


    # Stack per-bout feature vectors into a 2D array (n_bouts x n_features)
    rows = []
    for ib in sorted(d_e.keys()):
        vec = d_e[ib]
        if isinstance(vec, pd.Series):
            rows.append(vec.to_numpy())
        else:
            rows.append(np.asarray(vec))
    arr_e = np.vstack(rows)

    feat_names = [] 
    feat_names.extend([l.replace('f1','Rf1') for l in A1_labels+D1_labels]) 
    feat_names.extend([l.replace('f1','Ef1') for l in A1_labels+D1_labels]) 
    feat_names.extend([l.replace('f1','_REMOVE_Rf1') for l in A1_labels+D1_labels]) 
    feat_names.extend([l.replace('f2','Rf2') for l in D2_labels+A2_labels]) 
    feat_names.extend([l.replace('f2','Ef2') for l in D2_labels+A2_labels]) 
    feat_names.extend([l.replace('f2','_REMOVE_Rf2') for l in D2_labels+A2_labels]) 

    dfe = pd.DataFrame(columns=feat_names, data=arr_e)
    dfe[['name f1', 'name f2', 'tau']] = dfo[['name f1', 'name f2', 'tau']] 
    store_csv(dfe, acc_elos_path)     

def upcoming_acc_elos(acc_elos_path, AD_labels1, always_update=False):
    Rlabels1 = [l.replace('f1', 'Rf1') for l in AD_labels1] 
    Rlabels2 = [l.replace('f1', 'Rf2') for l in AD_labels1] 
    nRlabels1 = [l.replace('f1', '_REMOVE_Rf1') for l in AD_labels1] 
    Elabels1 = [l.replace('f1', 'Ef1') for l in AD_labels1] 
    Elabels2 = [l.replace('f1', 'Ef2') for l in AD_labels1] 

    dfp = UpcomingFights().open_clean()
    # Ensure duplicated structure keyed on name f1 and tau
    dfp = CleanedFights(dfp).duplicate(base_cols=['name f1', 'tau'])
    dfp = dfp.sort_values(by=['name f1']).reset_index(drop=True)

    dfe = open_csv(acc_elos_path) 
    # Extract final elos for f1 and keep last by tau per fighter
    dfe = CleanedFights(dfe).duplicate(custom_cols=nRlabels1, base_cols=['name f1', 'tau'])
    dfe = dfe.sort_values(by=['name f1', 'tau']).groupby('name f1', as_index=False).tail(1)
    dfe = dfe[['name f1'] + nRlabels1].sort_values(by=['name f1']).reset_index(drop=True)

    # Left-join to keep debuting/unknown fighters
    dfp = dfp.merge(dfe, on='name f1', how='left')

    # Warn on unknowns
    known = set(dfe['name f1'])
    unknown_mask = ~dfp['name f1'].isin(known)
    n_unknown = int(unknown_mask.sum())
    if n_unknown > 0:
        logger.warning(f'Encountered {n_unknown} unknown/upcoming debutants. '
                       f'Using neutral rating=1500 for expectancy; '
                       f'{"also filling R features" if always_update else "leaving R features as NaN"}.')

    # Map final remove-columns to active Rf1 columns
    dfp[Rlabels1] = dfp[nRlabels1].values

    # Optionally fill missing R with neutral base
    base_rating = 1500.0
    if always_update:
        dfp[Rlabels1] = dfp[Rlabels1].fillna(base_rating)

    # Unduplicate to get opponent view (Rf2)
    dfp = CleanedFights(dfp).unduplicate()

    # Compute expectancies with neutral substitution for any missing ratings
    r1_tmp = dfp[Rlabels1].copy().fillna(base_rating).values
    r2_tmp = dfp[Rlabels2].copy().fillna(base_rating).values
    dfp[Elabels1] = expectancy(r1_tmp, r2_tmp) 
    dfp[Elabels2] = 1 - dfp[Elabels1].values 

    UpcomingFights().append_features(dfp) 

if __name__ == '__main__': 
    get_acc_elos(per_round=False) 
    


    from src.feature_engineering.get_elo_params import get_elo_params
    
    wl_elos_path = get_data_path('features') / 'acc_elos.csv'
    dfe = open_csv(wl_elos_path) 
    dfe.drop(columns=[col for col in dfe.columns if 'REM' in col],inplace=True)

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


    



            
            





    




    

        








        


        # To d_e add the np.nans, to d_R not!
























