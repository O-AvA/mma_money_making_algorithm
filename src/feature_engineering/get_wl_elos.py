"""
Calculates elos based on win type.
"""
import pandas as pd
from numpy import isnan,nan
from loguru import logger 

from src.utils.general import (open_csv, store_csv, Dict, 
                               get_data_path,
                               expectancy, compute_elo) 
from src.data_processing.cleaned_data import CleanedFights 
from src.data_processing.upcoming_fights import UpcomingFights 

def get_wl_elos(win_score = [1,1,1,0.5,0,0,0],
                which_K = 'log', 
                process_upcoming_fights=False): 
    # Win score: [KO win, Sub win, Dec win, Draw, Dec loss, ...]  

    wl_elos_path = get_data_path('features') / 'wl_elos.csv'

    if process_upcoming_fights: 
        upcoming_elos(wl_elos_path) 

    dfo = CleanedFights()

    # Drop unnecessary columns
    dfo = dfo[['name f1', 'name f2', 'result','tau']]

    # Get the necessary parameters (K)   
    dfep = open_csv(get_data_path('features') / 'elo_params.csv') 

    # Current elos 
    d_R = Dict()
    # Features 
    d_e1 = Dict()
    d_e2 = Dict()
    
    for ib, bout in dfo.iterrows(): 
        name1 = bout['name f1'] 
        name2 = bout['name f2']
        result = bout['result'] 
        K1 = dfep.loc[ib, f'K{which_K} f1'] 
        K2 = dfep.loc[ib, f'K{which_K} f2'] 
        
        # Get current rating 
        R1 = d_R[name1] if name1 in d_R.keys() else 1500
        R2 = d_R[name2] if name2 in d_R.keys() else 1500

        E1 = expectancy(R1, R2)
        E2 = 1-E1

        if result > -1:
            S1 = win_score[result]
            S2 = win_score[6-result]
            nR1 = compute_elo(R1, E1, S1, K1) 
            nR2 = compute_elo(R2, E2, S2, K2) 
        else:
            nR1 = R1 
            nR2 = R2

        # We keep first fights separate so that we can exclude them 
        # from SVD. 
        R1 = nan if name1 not in d_R.keys() else R1
        R2 = nan if name2 not in d_R.keys() else R2
        E1 = nan if name1 not in d_R.keys() and E1==0.5 else E1
        E2 = nan if name2 not in d_R.keys() and E2==0.5 else E2
         
        d_e1.append_stat('name f1', name1) 
        d_e1.append_stat('E f1', E1) 
        d_e1.append_stat('R f1', R1)  
        d_e1.append_stat('R _REMOVE_ f1', nR1) 

        d_e2.append_stat('name f2', name2) 
        d_e2.append_stat('E f2', E2) 
        d_e2.append_stat('R f2', R2) 
        d_e2.append_stat('R _REMOVE_ f2', nR2)

        d_R[name1] = nR1 
        d_R[name2] = nR2

    dfe = pd.concat([pd.DataFrame(d_e1), 
                    pd.DataFrame(d_e2)
                    ],
                   axis = 1) 
    dfe[['name f1', 'name f2', 'tau', 'result']] = dfo[['name f1', 'name f2', 'tau','result']]
    store_csv(dfe, wl_elos_path) 

def upcoming_elos(wl_path):
    # Handles elos for fights to predict. 
    # Takes final elos calculated above and 
    # computes expectancy

    dfp = UpcomingFights().open_processed()
    dfp = dfp.duplicate(base_cols=['tau', 'name f1']) 

    dfe = open_csv(wl_path) 

    dfe = CleanedFights(dfe).duplicate(base_cols = dfe.columns.tolist())  
    dfe = dfe[dfe['name f1'].isin(dfp['name f1'])]
    dfe = dfe.groupby('name f1').tail(1) 

    # little bit inefficient but hey fuck you 
    dfp['R f1'] = dfp['name f1'].map(dict(zip(dfe['name f1'], 
                                      dfe['R _REMOVE_ f1']
                                    ) 
                                ) 
                            )

    
    dfp = dfp.unduplicate()
    dfp['E f1'] = expectancy(dfp['R f1'], dfp['R f2']) 
    dfp['E f2'] = 1-dfp['E f1'] 

    UpcomingFights().append_features(dfp)

if __name__ == '__main__':
    from src.feature_engineering.get_elo_params import get_elo_params
    
    d_params = {} 
    d_params['K0']  = 40
    d_params['Kmin'] = 5
    d_params['wf'] = 1
    d_params['ww'] = 1
    d_params['alpha'] = 1
    d_params['gamma'] = 0.03

    get_elo_params(d_params) 

    get_wl_elos(which_K='log')
    wl_elos_path = get_data_path('features') / 'wl_elos.csv'
    dfe = open_csv(wl_elos_path) 
    dfe.drop(columns=['name f1', 'name f2'] + [col for col in dfe.columns if 'REM' in col],
             inplace=True)  

    #dfe['result'] = CleanedFights()['result']
    dfe['tau'] = CleanedFights()['tau']
    dfe = dfe[dfe['result'] >= 0] 

    dfe['DE f1'] = dfe['E f1'] - dfe['E f2']
    dfe['DE f2'] = dfe['E f2'] - dfe['E f1']
    dfe['DR f1'] = dfe['R f1'] - dfe['R f2']
    dfe['DR f2'] = dfe['R f2'] - dfe['R f1']

    print('result' in dfe.columns) 
    dfe = CleanedFights(dfe).duplicate(base_cols=dfe.columns.tolist())     
    print('result' in dfe.columns) 
    dfe['result'] = dfe['result'].map({0:0, 1: 0, 2: 0, 
                                       3: 2, 
                                       4: 1, 5:1, 6: 1
                                       }) 
    dfe = dfe[dfe['result'] != 2]
    dfe.drop(columns=dfe.columns[dfe.columns.isin(['tau','upp_low','temp_f_id'])], inplace=True)
    correlations = dfe.corr()['result'].abs().sort_values(ascending=False)
    for col, corr in correlations.items():
        print(f"{col}: {corr}")


    



            
            





    




    
