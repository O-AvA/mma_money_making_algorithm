import itertools
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, log_loss, accuracy_score
from sklearn.model_selection import ParameterGrid 
from loguru import logger 
    
from src.utils.general import *
from src.data_processing.trainvalpred import TrainValPred


def do_cv(folds_seed=42):
    depths = [4] 
    rates = [0.02] 
    gammas = [1] 
    min_child_weights = [20]
    subsamples = [0.8] 
    colsamples = [0.8]  
    estimators = [400]
    n_repeats = 1
    num_class = 7
    n_folds = 5 


    processed_path = get_data_path('processed') 
    dfv = open_csv(processed_path / 'valid_stan_symm.csv')    
    
    dft, folds = TrainValPred().get_folds(n_repeats = n_repeats, 
                                        n_folds = n_folds, 
                                        first_seed=folds_seed) 

    dfv = dfv[dft.columns]


    # alle parameter grids
    param_grid = {
        "max_depth": depths,
        "learning_rate": rates,
        "n_estimators": estimators,
        "gamma": gammas,
        "min_child_weight": min_child_weights,
        "subsample": subsamples,
        "colsample_bytree": colsamples
    }


    Xt, yt = dft[[col for col in dft.columns if col != 'result']], dft['result']
    Xv, yv = dfv[[col for col in dfv.columns if col != 'result']], dfv['result']
    yv2 = yv.map({0:0, 1:0, 2:0, 3:1, 4:2, 5:2, 6:2})

    logger.info(f'Optimizing parameters for {len(dft.columns)} features using {n_repeats} repeat of a {n_folds}-fold CV')
    logger.info(f'Starting grid search...') 


    for params in ParameterGrid(param_grid):
        # Skip if parameters are already checked 
        d = params.copy()
        d['seed'] = folds_seed
        d['n_repeats'] = n_repeats
        opt_path = get_data_path('output') / 'optimization.csv'
        df_opt = creaopen_file(opt_path) 
        if not df_opt.empty:  
            skip = (df_opt[list(d)] == pd.Series(d)).all(axis=1).any() 
            skip = False 
            if skip:
                skip 
            
                continue 

        f1_scores, log_losses, accs7, accs2 = [], [], [], []

        # jouw custom folds + repeats
        for j in range(n_repeats):

            for train_idx, val_idx in folds[j]:
                model = XGBClassifier(
                    objective="multi:softprob",
                    num_class=num_class,
                    early_stopping_rounds=25,
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                    verbosity=0, 
                    **params
                )

                Xtt, Xtv = Xt.iloc[train_idx], Xt.iloc[val_idx]
                ytt, ytv = yt.iloc[train_idx], yt.iloc[val_idx]

                model.fit(
                    Xtt, ytt,
                    eval_set=[(Xtv, ytv)],
                    verbose=False
                )

                y_proba = model.predict_proba(Xtv)
                y_pred = np.argmax(y_proba, axis=1)

                f1_scores.append(f1_score(ytv, y_pred, average="macro"))
                log_losses.append(log_loss(ytv, y_proba))

                yv_proba = model.predict_proba(Xv)
                yv_pred = np.argmax(yv_proba, axis=1)
                accs7.append(accuracy_score(yv, yv_pred))

                win = yv_proba[:, :3].sum(axis=1)
                draw = yv_proba[:, 3]
                lose = yv_proba[:, 4:].sum(axis=1)
                y_grouped_proba = np.vstack([win, draw, lose]).T
                x = np.random.randint(len(y_grouped_proba)) 
                print(y_grouped_proba[x]) 

                yv_pred2 = np.argmax(y_grouped_proba, axis=1)

                accs2.append(accuracy_score(yv2, yv_pred2)) 

        # gemiddelde + std berekenen
        # To this:
        f1_scores_clean = np.nan_to_num(f1_scores, nan=0.0)
        f1 = round(np.mean(f1_scores_clean), 3)
        ll = round(np.mean(log_losses), 3)
        f1std = round(np.std(f1_scores), 3)
        llstd = round(np.std(log_losses), 3)
        acc = round(np.mean(accs7), 3)
        accstd = round(np.std(accs7), 3)
        acc2 = round(np.mean(accs2), 3)
        acc2std = round(np.std(accs2), 3)

        # netjes printen
        d = {k: [d[k]] for k in d.keys()} 
        d['logloss'] = [ll]
        d['f1 score'] = [f1] 
        d['acc7'] = [acc] 
        d['acc2'] = [acc2]
    
        df_opt = pd.concat([df_opt, pd.DataFrame(d)], ignore_index=True)
        store_csv(df_opt, opt_path) 
        print(params) 
        print(f"logloss: {ll} +/- {llstd} |||| f1-score: {f1} +/- {f1std} |||| dfv acc7: {acc} +/- {accstd} |||| dfv acc2: {acc2} +/- {acc2std}")
        print("")


if __name__ == '__main__': 
    do_cv() 
