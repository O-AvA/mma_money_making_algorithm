

class CVParams(): 
    """
    Utilities for setting parameters for the Cross validation functions

    """

    def __init__(self, CVmain): 
        self.CVmain = CVmain 


    def set_valid_params(
            self, 
            params_dict=None,
            vv_size =  0, 
            vv_random_split = False, 
            vv_seed = 42
    ): 
        """
        In stead of minimizing on the training data, you can also 
        minimize on a subset on the validation data. Since in the 
        current set-up the validation data contains the most recent 
        portion of fights, it represents a more relevant part of the 
        data and it may be worthwhile minimizing on this in stead. 

        Args: 
            vv_size float [0,1): 
                The portion of the validation data you wanna split off
                and use for validation 'valid_valid'. The hyperparamters
                will be optimized on the remainder 'valid_train'.
                If kept to zero, it will just minimize on training data.  
            vv_random_split bool: 
                You can either split by recency (default, False) or 
                randomly. 
            vv_seed: 
                When `vv_random_split = True`, choose your seed.
            params_dict dict: 
                Set all at once 
        """
        self.CVmain.valid_params = {
                'vv_size': vv_size, 
                'vv_random_split': vv_random_split, 
                'vv_seed': vv_seed
        } if params_dict is None else params_dict  
   
    def set_cv_params(
            self, 
            params_dict=None,
            n_repeats= 1,
            n_folds = 5, 
            fold_seed = 42 
    ): 
       """
       Set parameters for the cross validation. 

       Args: 
           n_repeats int: 
                The number of times you wanna repeat the cross validation
                (for n_repeats different folds)
           n_folds int: 
                The number of folds. So the model gets evaluated
                n_repeats * n_folds times. 
           fold_seed int:  
                Random seed for folding the training data. 
            params_dict dict: 
                Set all values yourself 
       """
       self.CVmain.cv_params = {
           'fold_seed': fold_seed, 
           'n_folds' : n_folds, 
           'n_repeats': n_repeats,
       } if params_dict is None else params_dict 

    def set_hyper_params(self,params_dict=None): 
        xgb_params = { 
            "max_depth": 5,
            "learning_rate": (0.02, 0.025),
            "n_estimators": (500,600),
            "min_child_weight": (0, 40),
            "gamma": (0, 2.5),
            "subsample": (0.7, 0.85),
            "colsample_bytree": (0.95, 1.0),
            # Optional regularization
            "reg_alpha": 0.0,
            "reg_lambda": 1.0
        } if params_dict is None else params_dict 
        self.CVmain.hyper_params = xgb_params

    def set_feature_params(
            self, 
            params_dict = None, 
            select_by = 'frequency', 
            feature_range = (12, 15) 
    ):
        """
        At stage (3) of the pipeline, you can also 
        vary over the best features. 
        select_by str: 
            Either vary over a range of top features by `frequency`
            or by `index` 
        feature_range (list, tuple) or int:
            Depending on your choice for `select_features_by`, 
            set a range to vary over, fix it.  
        """
        self.CVmain.feature_params = { 
            'select_by': select_by, 
            'feature_range': feature_range
        } if params_dict is None else params_dict 


    def set_stability_check_params(
            self, 
            params_dict = None, 
            top_n = 5,
            n_repeats = 3
    ):
        """
        Params for xgb random_state seed stability check. Perform 
        n_repeats for the top_n best hyper params. 
        """
        self.CVmain.stability_check_params = { 
            'top_n': top_n,
            'n_repeats': n_repeats
        } if params_dict is None else params_dict 


    def change_cv_param(self, key, new_val):
        """
        Change a single param without having to reset all 
        """

        if not hasattr(self.CVmain, 'cv_params'): 
            self.CVparams.set_cv_params() 
        cv_params = self.CVmain.cv_params.copy() 
        cv_params[key] = new_val 
        self.CVmain.cv_params = cv_params


