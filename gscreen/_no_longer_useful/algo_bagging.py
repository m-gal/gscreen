# ------------------------------------------------------------------------------
# -------------------  sklearn.ensemble.BaggingRegressor  ----------------------
# ------------------------------------------------------------------------------
#%% Import models's libraries
from sklearn import tree  # Canonical Decision tree & Extremely randomized tree
from sklearn import ensemble  # RF, Gradient Boosting, AdaBoost
from skopt.space import Real, Categorical, Integer

import xgboost as xgb

#%% Toggles to go through
random_state = 42

#%% base_estimator = tree.ExtraTreeRegressor
base_xt_reg = tree.ExtraTreeRegressor(
    criterion="mse",  # {"mse", "friedman_mse", "mae"} default="mse"
    splitter="random",  # {"random", "best"} default="random"
    max_depth=None,  # int, default=None
    min_samples_split=2,  # int or float, default=2
    min_samples_leaf=1,  # int or float, default=1
    min_weight_fraction_leaf=0.0,  # float, default=0.0
    max_features=None,  # int, float or {“auto”, “sqrt”, “log2”}, default=None
    random_state=random_state,
)

#%% base_estimator = tree.DecisionTreeRegressor
base_dt_reg = tree.DecisionTreeRegressor(
    criterion="mse",  # {"mse", "friedman_mse", ""mae"} default="mse"
    splitter="best",  # {"random", "best"} default="best"
    max_depth=None,  # int, default=None
    min_samples_split=2,  # int or float, default=2
    min_samples_leaf=1,  # int or float, default=1
    min_weight_fraction_leaf=0.0,  # float, default=0.0
    max_features=None,  # int, float or {“auto”, “sqrt”, “log2”}, default=None
    random_state=random_state,
)

#%% base_estimator = ensemble.RandomForestRegressor
base_rf_reg = ensemble.RandomForestRegressor(
    criterion="mse",  # {"mse", "mae"} default="mse"
    n_estimators=1000,
    max_depth=None,  # int, default=None
    min_samples_split=2,  # int or float, default=2
    min_samples_leaf=1,  # int or float, default=1
    min_weight_fraction_leaf=0.0,  # float, default=0.0
    max_features="auto",
    max_samples=None,
    bootstrap=True,
    n_jobs=-1,
    random_state=random_state,
)

#%% base_estimator = xgb.XGBRegressor
base_xgb_reg = xgb.XGBRegressor(
    booster="gbtree",
    objective="reg:squarederror",
    n_estimators=2500,
    subsample=1,
    n_jobs=-1,
    random_state=random_state,
)

#%% Bagging model
def model(base_estimator=base_xt_reg):
    model_params = {
        "base_estimator": base_estimator,
        "n_estimators": 80,
        "max_samples": 1.0,
        "max_features": 1.0,
        "bootstrap": True,
        "bootstrap_features": False,
        "oob_score": False,
        "n_jobs": -1,
        "random_state": random_state,
        "verbose": 3,
    }
    model = ensemble.BaggingRegressor(**model_params)
    model_name = type(model).__name__
    return model_name, model, model_params


#%% ----------------------- Parameters Searh CV --------------------------------
def param_search():
    # Parameters what we wish to tune in case SIMPLE grid search
    ## Dictionary with parameters names (str) as keys
    ## and lists of parameter settings to try as values
    param_grid = {
        # The number of base estimators in the ensemble.
        # If base_estimator=None, then the base estimator is a DecisionTreeRegressor.
        "n_estimators": [10, 50, 100],
        # ^ subsample: default=1. Lower ratios avoid over-fitting
        "max_samples": [0.6, 0.8, 1],
        # ^ colsample default=1. Lower ratios avoid over-fitting.
        "max_features": [0.8, 0.9, 1],
        # ^ Whether samples are drawn with replacement.
        # ^ if False, sampling without replacement is performed. default=True
        "bootstrap": [True, False],
        # ^ Whether features are drawn with replacement. default=False
        "bootstrap_features": [True, False],
    }
    # Parameters' distributions tune in case RANDOMIZED grid search
    ## Dictionary with parameters names (str) as keys and distributions
    ## or lists of parameters to try.
    ## If a list is given, it is sampled uniformly.
    param_dist = {
        "n_estimators": [x for x in range(10, 101, 10)],
        "max_samples": [x / 100 for x in range(6, 101, 1)],
        "max_features": [x / 100 for x in range(7, 101, 1)],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
    }
    # Their core idea of Bayesian Optimization is simple:
    # when a region of the space turns out to be good, it should be explored more.
    # Real: Continuous hyperparameter space.
    # Integer: Discrete hyperparameter space.
    # Categorical: Categorical hyperparameter space.
    bayes_space = {
        "n_estimators": Integer(50, 100),
        "max_samples": Real(0.5, 1.0),
        "max_features": Real(0.7, 1.0),
        "bootstrap": Categorical([True, False]),
        "bootstrap_features": Categorical([True, False]),
    }

    return param_dist, param_grid, bayes_space
