from loguru import logger
import pandas as pd
import numpy as np

from src.model_selection.trainvalpred import TrainValPred
from src.utils.general import get_data_path, open_csv


class CVDataloader:

    def __init__(self, CVmain):
        self.CVmain = CVmain

    def _load_train(self):

        # Training data
        dft, folds = TrainValPred().get_folds(
            self.CVmain.suffix,
            n_repeats=self.CVmain.cv_params['n_repeats'],
            n_folds=self.CVmain.cv_params['n_folds'],
            first_seed=self.CVmain.cv_params['fold_seed']
        )

        Xt, yt = dft.drop(columns=['result']), dft['result']

        self.CVmain.Xt = Xt
        self.CVmain.yt = yt
        self.CVmain.folds = folds

        logger.info(f'Number of features: {len(Xt.columns)}')

    def _load_valid(self):
        Xt = self.CVmain.Xt.copy()
        valid_params = self.CVmain.valid_params
        vv_size = valid_params['vv_size']

        dfv = TrainValPred().open('valid', self.CVmain.suffix)

        # Align to training columns, keep a copy
        Xv, yv = dfv[Xt.columns], dfv['result']
        Xvt, yvt = Xv.copy(), yv.copy()
        Xvv, yvv = None, None

        if vv_size > 0:
            vv_seed = valid_params['vv_seed']
            vv_random_split = valid_params['vv_random_split']

            vnames_path = get_data_path('interim') / 'valid_names.csv'
            df_names = open_csv(vnames_path)
            f_ids_v = df_names['temp_f_id']

            # Attach f_id for splitting but don't persist it
            Xv = Xv.copy()
            Xv['temp_f_id'] = f_ids_v.values

            if vv_random_split:
                rng = np.random.default_rng(seed=vv_seed)
                unique_ids = f_ids_v.unique()
                n_sample = max(1, int(vv_size * len(unique_ids)))
                f_ids_vv = rng.choice(unique_ids, size=n_sample, replace=False)
            else:
                # Most recent fights have largest temp_f_id; take the top vv_size fraction
                cutoff = (1 - vv_size) * f_ids_v.max()
                f_ids_vv = f_ids_v[f_ids_v > cutoff].unique()

            Xvv = Xv[Xv['temp_f_id'].isin(f_ids_vv)]
            yvv = yv.iloc[Xvv.index]

            Xvt = Xv[~Xv['temp_f_id'].isin(f_ids_vv)]
            yvt = yv.iloc[Xvt.index]

            # Drop helper column and reset indices for downstream code
            Xvv = Xvv.drop(columns=['temp_f_id']).reset_index(drop=True)
            Xvt = Xvt.drop(columns=['temp_f_id']).reset_index(drop=True)
            yvv = yvv.reset_index(drop=True)
            yvt = yvt.reset_index(drop=True)

        self.CVmain.Xvt = Xvt
        self.CVmain.yvt = yvt
        self.CVmain.Xvv = Xvv
        self.CVmain.yvv = yvv

    def _load_pred(self):
        dfp = TrainValPred().open('pred', self.CVmain.suffix)
        self.CVmain.Xp = dfp

     





