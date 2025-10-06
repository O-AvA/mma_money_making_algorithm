import pandas as pd 
from numpy import sqrt, log, exp 
from loguru import logger

from src.utils.general import (get_data_path, open_csv, store_csv)
from src.data_processing.cleaned_data import CleanedFights
from src.data_processing.upcoming_fights import UpcomingFights


def get_elo_params(d_params, process_upcoming_fights=False): 
    """
    Creates parameters necessary for Elo calculation. These can be 
    vectorized so can be done separetely.

    Args: 
        d_params dict: 
            Dictionary with parameters: 
                K0: Bare K 
                Kmin: Minimal rating update 
                gamma: DEFUNCT uncertainty to give inactive fighters 
                wf: DEFUNCT weight for number of fights [0,1] 
                ww: DEFUNCT weight for number of weeks active [0,1]
                alpha: DEFUNCT 
    """
    K0 = d_params['K0'] 
    Kmin = d_params['Kmin'] 
    wf = d_params['wf'] 
    ww = d_params['ww'] 
    alpha = d_params['alpha'] 
    gamma = d_params['gamma'] 
   
    # Open base features. Contains necessary features.   
    base_feat_path = get_data_path('features') / 'base_features.csv' 
    dfb = open_csv(base_feat_path)
    needed_cols = ['tot round 1 fought f1', 
                   'weeks active f1', 
                   'last fight wa f1'
    ]

    needed_cols.extend([f'tot round {ri} fought f1' for ri in range(2,6)]) 

    base_cols = ['name f1', 'tau'] 
    dfb = CleanedFights(dfb).duplicate(custom_cols = needed_cols, base_cols=base_cols) 

    rounds = range(1,6) #if per_round else [1] 

    if process_upcoming_fights:
        dfp = UpcomingFights().open_processed()
        logger.info(f'len dfp {len(dfp)}') 
        dfp = dfp.duplicate(needed_cols,base_cols) 
        # Filter on names 
        dfb = dfb[dfb['name f1'].isin(dfp['name f1'])]
        # Only get last fights 
        dfb = dfb.groupby('name f1').tail(1)
        if len(dfb) > len(dfp): 
            logger.error('Huh') 
            raise 

    
    # Note that 'tot round 1 fought' = Number of fights fought 
    # 'weeks active' is just carreer length in weeks
    lendfb = len(dfb) 
    new_cols = [] 
    for ri in rounds: 

        rlabel = f'r{ri}' if ri > 1 else ""

        dfb[f'xp_score{rlabel} f1'] = wf*sqrt(dfb[f'tot round {ri} fought f1']) + ww*log(1 + dfb['weeks active f1'])

        # Last fight in weeks ago 
        Kcust_base = 10 + K0 / (1 + alpha*dfb[f'xp_score{rlabel} f1']) * (1 + gamma*dfb['last fight wa f1'])
        Klog_base = 10 + 40 / (1 + exp(0.3* dfb[f'tot round {ri} fought f1'] - 4))
        dfb[f'Klog{rlabel} f1'] = Klog_base.apply(lambda x: max(x, Kmin))
        dfb[f'Kcust{rlabel} f1'] = Kcust_base.apply(lambda x: max(x, Kmin))

        new_cols.extend([f'Klog{rlabel} f1', f'Kcust{rlabel} f1', f'xp_score{rlabel} f1']) 
    if len(dfb) != lendfb: 
        logger.error('Ok hier dus') 
        raise 
   
    if process_upcoming_fights:
        # Prepare neutral defaults for debutants/unknowns
        names_upcoming = set(dfp['name f1'])
        names_known = set(dfb['name f1'])
        unknown_names = names_upcoming - names_known
        if len(unknown_names) > 0:
            logger.warning(f'Encountered {len(unknown_names)} upcoming fighters without base history. '
                           f'Filling xp_score=0 and K with neutral defaults.')

        # Compute neutral defaults once (0 fights, 0 weeks active, last fight wa = 0)
        xp_default = 0.0
        Klog_default = max(10 + 40 / (1 + exp(0.3*0 - 4)), Kmin)
        Kcust_default = max(10 + K0 / (1 + alpha*xp_default) * (1 + gamma*0), Kmin)

        dfp.drop(columns=needed_cols, inplace=True)
        for col in new_cols:  
            param_map = dict(zip(dfb['name f1'], dfb[col])) 
            # Map known values and fill unknowns with neutral defaults
            if 'xp_score' in col:
                fill_val = xp_default
            elif 'Klog' in col:
                fill_val = Klog_default
            elif 'Kcust' in col:
                fill_val = Kcust_default
            else:
                fill_val = None  # should not occur
            dfp[col] = dfp['name f1'].map(param_map).fillna(fill_val)

        dfp = dfp.unduplicate() 
        UpcomingFights().append_features(dfp) 
    else:
        dfb = CleanedFights(dfb).unduplicate()
        keep_cols = new_cols + [nc.replace('f1', 'f2') for nc in new_cols] + ['name f1', 'tau']
        dfb = dfb[keep_cols] 

        params_path = get_data_path('features') / 'elo_params.csv' 
        store_csv(dfb, params_path) 
        
def set_elo_params(K0 = 40, 
                   Kmin = 5, 
                   wf = 1,
                   ww = 0.75,
                   alpha = 0.8,
                   gamma = 0.05
                   ): 
    d_params = {} 
    d_params['K0']  = K0 
    d_params['Kmin'] = Kmin
    d_params['wf'] = wf
    d_params['ww'] = ww
    d_params['alpha'] = alpha
    d_params['gamma'] = gamma
    return d_params

if __name__=='__main__': 


    get_elo_params(set_elo_params())


