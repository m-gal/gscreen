# ------------------------------------------------------------------------------
# ------------------- S T A C K E D   R E G R E S S O R ------------------------
# ------------------------------------------------------------------------------
#%% Import models's libraries
from sklearn import tree
from sklearn import ensemble  # RF, Gradient Boosting, AdaBoost

#%% Toggles to go through
random_state = 42

#%% Define parameters and model
def model(model_name="stacked"):
    model_params = None
    model = ensemble.StackingRegressor(
        estimators=[
            (
                "rf",
                ensemble.RandomForestRegressor(
                    random_state=random_state,
                    n_estimators=100,  # default=100
                    criterion="mse",  # {"mse", "mae"}, default="mse"
                    max_depth=None,  # default=None
                    min_samples_split=2,  # default=2
                    min_samples_leaf=1,  # default=1
                ),
            ),
            # (
            #     "dt",
            #     tree.DecisionTreeRegressor(
            #         random_state=random_state,
            #         criterion="mse",  # {"mse", "mae"}, default="mse"
            #         max_depth=None,  # default=None
            #         min_samples_split=2,  # default=2
            #         min_samples_leaf=1,  # default=1
            #     ),
            # ),
            # (
            #     "etr",
            #     tree.ExtraTreeRegressor(
            #         random_state=random_state,
            #         criterion="mse",  # {"mse", "mae"}, default="mse"
            #         splitter="random",  # {"random", "best"}, default="random"
            #         max_depth=None,  # default=None
            #         min_samples_split=2,  # default=2
            #         min_samples_leaf=1,  # default=1
            #     ),
            # ),
            (
                "etrs",
                ensemble.ExtraTreesRegressor(
                    random_state=random_state,
                    n_estimators=100,  # default=100
                    criterion="mse",  # {"mse", "mae"}, default="mse"
                    max_depth=None,  # default=None
                    min_samples_split=2,  # default=2
                    min_samples_leaf=1,  # default=1
                    max_features="auto",  # {"auto", "sqrt", "log2"}, default=”auto”
                    max_samples=None,  # int or float, default=None
                    bootstrap=False,  # default=False
                ),
            ),
        ],
        # The final estimator. default regressor is a None
        final_estimator=None,
        # CV for final_estimator. None, to use the default 5-fold cross validation
        cv=None,
        n_jobs=-1,  # -1 means using all processors
        passthrough=True,
        verbose=3,
    )
    model_name = type(model).__name__
    return model_name, model, model_params


#%% --------------------------- GridSearchCV -----------------------------------
def param_search():
    # Parameters' distributions tune in case RANDOMIZED grid search
    ## Dictionary with parameters names (str) as keys and distributions
    ## or lists of parameters to try.
    ## If a list is given, it is sampled uniformly.
    param_dist = {
        # The number of trees in the forest. default=100
        "rf__n_estimators": [x for x in range(100, 1001, 100)],
        # Max number of levels in tree. default=None
        "rf__max_depth": [None] + [x for x in range(6, 19, 4)],
        # Min number of samples required to split a node. default=2.
        "rf__min_samples_split": [x for x in range(2, 21, 2)],
        # Min number of data points allowed in a leaf node. default=1.
        "rf__min_samples_leaf": [x for x in range(1, 11, 1)],
    }

    # Parameters what we wish to tune in case SIMPLE grid search
    ## Dictionary with parameters names (str) as keys
    ## and lists of parameter settings to try as values
    param_grid = {
        "n_estimators": [2500],
        # ^ subsample: default=1. Lower ratios avoid over-fitting
        "subsample": [0.6, 0.8],
        # ^ "colsample_bytree: default=1. Lower ratios avoid over-fitting.
        "colsample_bytree": [0.6, 0.8],
        # ^ max_depth: default=6. Lower ratios avoid over-fitting.
        "max_depth": [12, 24],
        # ^ min_child_weight: default=1. Larger values avoid over-fitting.
        "min_child_weight": [1] + [x for x in range(2, 11, 2)],
        # ^ Eta (lr): default=0.3. Lower values avoid over-fitting.
        "learning_rate": [0.001, 0.01, 0.1],
        # ^ Lambda: default=1. Larger values avoid over-fitting.
        "reg_lambda": [0.5, 1, 2],
        # ^ Gamma: default=0. Larger values avoid over-fitting.
        "gamma": [0, 1, 2, 5],
    }

    return param_dist, param_grid
