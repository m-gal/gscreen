# ------------------------------------------------------------------------------
# ----------------------------- X G B O O S T ----------------------------------
# ------------------------------------------------------------------------------
#%% Import models's libraries
import xgboost as xgb
from skopt.space import Real, Categorical, Integer

#%% Toggles to go through
random_state = 42

#%% Define parameters and model
def model():
    model_params = {
        # "tree_method": "gpu_hist",
        # "gpu_id": 0,
        "booster": "gbtree",
        "objective": "reg:squarederror",
        # "objective": "reg:pseudohubererror",
        # "objective": "reg:gamma",
        "n_estimators": 2500,
        "subsample": 0.6,
        # "colsample_bytree": 1,
        # "max_depth": 6,
        # "min_child_weight": 1,
        # "learning_rate": 0.3,
        # "reg_lambda": 1,
        # "gamma": 0,
        "n_jobs": -1,
        "random_state": random_state,
    }
    # early_stopping_params = {
    #     "early_stopping_rounds": 20,
    #     "eval_metric": "mae",
    #     # "eval_metric": "mape",
    #     # "eval_metric": "rmse",
    #     "eval_set": [(X_val, y_val)],
    # }
    model = xgb.XGBRegressor(**model_params)
    model_name = type(model).__name__

    return model_name, model, model_params


#%% --------------------------- GridSearchCV -----------------------------------
def param_search():
    # Parameters' distributions tune in case RANDOMIZED grid search
    ## Dictionary with parameters names (str) as keys and distributions
    ## or lists of parameters to try.
    ## If a list is given, it is sampled uniformly.
    param_dist = {
        "n_estimators": [x for x in range(2250, 3001, 250)],
        # ^ subsample: default=1. Lower ratios avoid over-fitting
        "subsample": [x / 10 for x in range(5, 11, 1)],
        # ^ "colsample_bytree: default=1. Lower ratios avoid over-fitting.
        "colsample_bytree": [x / 10 for x in range(6, 11, 1)],
        # ^ max_depth: default=6. Lower ratios avoid over-fitting.
        "max_depth": [x for x in range(6, 51, 4)],
        # ^ min_child_weight: default=1. Larger values avoid over-fitting.
        "min_child_weight": [1] + [x for x in range(2, 11, 2)],
        # ^ Eta (lr): default=0.3. Lower values avoid over-fitting.
        "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
        # ^ Lambda: default=1. Larger values avoid over-fitting.
        "reg_lambda": [1],  # + [x for x in range(2, 11, 1)],
        # ^ Gamma: default=0. Larger values avoid over-fitting.
        "gamma": [0.0],  # + [x/10 for x in range(5, 60, 5)]
    }
    # Parameters what we wish to tune in case SIMPLE grid search
    ## Dictionary with parameters names (str) as keys
    ## and lists of parameter settings to try as values
    param_grid = {
        "n_estimators": [2400, 2500, 2600],
        # ^ subsample: default=1. Lower ratios avoid over-fitting
        "subsample": [0.6, 0.8],
        # ^ "colsample_bytree: default=1. Lower ratios avoid over-fitting.
        "colsample_bytree": [0.6, 0.8],
        # ^ max_depth: default=6. Lower ratios avoid over-fitting.
        "max_depth": [6, 12, 24],
        # ^ min_child_weight: default=1. Larger values avoid over-fitting.
        "min_child_weight": [1] + [x for x in range(2, 11, 2)],
        # ^ Eta (lr): default=0.3. Lower values avoid over-fitting.
        "learning_rate": [0.01, 0.1, 0.3],
        # ^ Lambda: default=1. Larger values avoid over-fitting.
        "reg_lambda": [0.5, 1, 2],
        # ^ Gamma: default=0. Larger values avoid over-fitting.
        "gamma": [0, 1, 2, 5],
    }
    # Their core idea of Bayesian Optimization is simple:
    # when a region of the space turns out to be good, it should be explored more.
    # Real: Continuous hyperparameter space.
    # Integer: Discrete hyperparameter space.
    # Categorical: Categorical hyperparameter space.
    bayes_space = {
        # "n_estimators": Integer(2000, 3000),
        "subsample": Real(0.6, 1.0),
        "colsample_bytree": Real(0.7, 1.0),
        "max_depth": Integer(3, 20),
        "min_child_weight": Integer(1, 20),
        "learning_rate": Real(0.01, 0.4),
        "reg_lambda": Real(0.5, 5),
        "gamma": Real(0.0, 5),
    }

    return param_dist, param_grid, bayes_space
