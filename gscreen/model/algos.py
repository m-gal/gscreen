import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import tree  # Canonical Decision tree & Extremely randomized tree
from sklearn import ensemble  # RF, Gradient Boosting, AdaBoost
from skopt.space import Real, Categorical, Integer

rnd_state = 42

# ------------------------------------------------------------------------------
# ------------------  Base estimators which will be used  ----------------------
# ------------------------------------------------------------------------------


def get_xtr():
    """An extremely randomized tree regressor.

    * criterion: {“mse”, “friedman_mse”, “mae”}, default=”mse”
        The function to measure the quality of a split.
    * splitter: {“random”, “best”}, default=”random”
        The strategy used to choose the split at each node.
    * max_depth: int, default=None
        The maximum depth of the tree.
        If None, then nodes are expanded until all leaves are pure
        or until all leaves contain less than min_samples_split samples.
    * min_samples_split: int or float, default=2
        The minimum number of samples required to split an internal node.
    * min_samples_leaf: int or float, default=1
        The minimum number of samples required to be at a leaf node.
    * min_weight_fraction_leaf: float, default=0.0
        The minimum weighted fraction of the sum total of weights
        (of all the input samples) required to be at a leaf node.
        Samples have equal weight when sample_weight is not provided.
    * max_features: int, float, {“auto”, “sqrt”, “log2”} or None, default=”auto”
        The number of features to consider when looking for the best split.
    """
    return tree.ExtraTreeRegressor(
        criterion="mse",
        splitter="random",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=rnd_state,
    )


def get_dtr():
    """A decision tree regressor.

    * criterion: {“mse”, “friedman_mse”, “mae”, “poisson”}, default=”mse”
        The function to measure the quality of a split.
    * splitter: {“best”, “random”}, default=”best”
        The strategy used to choose the split at each node.
    * max_depth: int, default=None
        The maximum depth of the tree.
        If None, then nodes are expanded until all leaves are pure or until
        all leaves contain less than min_samples_split samples.
    * min_samples_split: int or float, default=2
        The minimum number of samples required to split an internal node:
    * min_samples_leaf: int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at least
        min_samples_leaf training samples in each of the left and right branches.
    * min_weight_fraction_leaf: float, default=0.0
        The minimum weighted fraction of the sum total of weights
        (of all the input samples) required to be at a leaf node.
        Samples have equal weight when sample_weight is not provided.
    * max_features: int, float or {“auto”, “sqrt”, “log2”}, default=None
        The number of features to consider when looking for the best split.
    """
    return tree.DecisionTreeRegressor(
        criterion="mse",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=rnd_state,
    )


def get_rfr():
    """A random forest regressor.

    * n_estimators: int, default=100
        The number of trees in the forest.
    * criterion: {“mse”, “mae”}, default=”mse”
        The function to measure the quality of a split.
    * max_depth: int, default=None
        The maximum depth of the tree.
        If None, then nodes are expanded until all leaves are pure or until
        all leaves contain less than min_samples_split samples.
    * min_samples_split: int or float, default=2
        The minimum number of samples required to split an internal node:
    * min_samples_leaf: int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at least
        min_samples_leaf training samples in each of the left and right branches.
    * min_weight_fraction_leaf: float, default=0.0
        The minimum weighted fraction of the sum total of weights
        (of all the input samples) required to be at a leaf node.
        Samples have equal weight when sample_weight is not provided.
    * max_features: {“auto”, “sqrt”, “log2”}, int or float, default=”auto”
        The number of features to consider when looking for the best split.
    * max_samplesint or float, default=None
        If bootstrap is True, the number of samples to draw from X to train each
        base tree.
    * bootstrapbool, default=True
        Whether bootstrap samples are used when building trees.
    """
    return ensemble.RandomForestRegressor(
        n_estimators=100,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_samples=None,
        bootstrap=True,
        n_jobs=-1,
        random_state=rnd_state,
    )


def get_xgbr():
    """XGBoost for Scikit-learn

    * booster: default= gbtree
        Which booster to use. Can be gbtree, gblinear or dart;
        gbtree and dart use tree based models while gblinear uses linear functions.
    * objective: default=reg:squarederror
        Specify the learning task and the corresponding learning objective.
        The regression objective options are below:
            - reg:squarederror: regression with squared loss.
            - reg:squaredlogerror: regression with squared log loss.
            - reg:logistic: logistic regression.
            - reg:pseudohubererror: regression with Pseudo Huber loss.
            - reg:gamma: gamma regression with log-link.
            - reg:tweedie: Tweedie regression with log-link.
    * n_estimators: int, default=100
        The number of trees in the forest.
    """
    return xgb.XGBRegressor(
        booster="gbtree",
        objective="reg:squarederror",
        n_estimators=1000,
        subsample=1,
        n_jobs=-1,
        random_state=rnd_state,
    )


# ------------------------------------------------------------------------------
# -------------------  sklearn.ensemble.BaggingRegressor  ----------------------
# ------------------------------------------------------------------------------


def get_bagging_search_params():
    """ Grids of hyper-parameters for searching optima of its """

    """
        Parameters' distributions tune in case RANDOMIZED grid search
        Dictionary with parameters names (str) as keys and distributions
        or lists of parameters to try. If a list is given, it is sampled uniformly.
    """
    params_dist = {
        "n_estimators": [x for x in range(10, 101, 10)],
        "max_samples": [x / 100 for x in range(60, 101, 1)],
        "max_features": [x / 100 for x in range(70, 101, 1)],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
    }

    """
        Parameters what we wish to tune in case SIMPLE grid search.
        Dictionary with parameters names (str) as keys and
        lists of parameter settings to try as values.
    """
    params_grid = {
        "n_estimators": [50, 100],
        "max_samples": [0.6, 0.8, 1],
        "max_features": [0.8, 0.9, 1],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
    }

    """
        Their core idea of Bayesian Optimization is simple:
        when a region of the space turns out to be good, it should be explored more.
        Real: Continuous hyperparameter space.
        Integer: Discrete hyperparameter space.
        Categorical: Categorical hyperparameter space.
    """
    bayes_space = {
        "n_estimators": Integer(50, 100),
        "max_samples": Real(0.5, 1.0),
        "max_features": Real(0.7, 1.0),
        "bootstrap": Categorical([True, False]),
        "bootstrap_features": Categorical([True, False]),
    }

    return params_dist, params_grid, bayes_space


def get_bagging_model(
    base_estimator=get_xtr(),
    n_estimators=80,
    n_jobs=-1,
    verbose=1,
):
    """
    Parameters which we will use in final model training on DEVELOPMENT set.
    Dict with parameters names (str) as keys and parameter settings as values.

        * base_estimator: object, default=None.
            The base estimator to fit on random subsets of the dataset.
            If None, then the base estimator is a DecisionTreeRegressor.
        * n_estimators: int, default=10.
            The number of base estimators in the ensemble.
        * max_samples: int or float, default=1.0.
            The # of samples to draw from X to train each base estimator
            (with replacement by default). Lower ratios avoid over-fitting
        * max_features: int or float, default=1.0.
            Like `max_samples` but refer to features. Lower ratios avoid over-fitting.
        * bootstrap: bool, default=True.
            Whether samples are drawn with replacement. If False, sampling without
            replacement is performed.
        * bootstrap_features: bool, default=False.
            Whether features are drawn with replacement.
        * oob_score: bool, default=False.
            Whether to use out-of-bag samples to estimate the generalization error.
        * n_jobs: int, default=None.
            The number of jobs to run in parallel for both fit and predict.
            None means 1. -1 means using all processors.
    """
    print(f"\nLoad model...")
    model_params = {
        "base_estimator": base_estimator,
        "n_estimators": n_estimators,
        "max_samples": 1.0,
        "max_features": 1.0,
        "bootstrap": True,
        "bootstrap_features": False,
        "oob_score": False,
        "n_jobs": n_jobs,
        "random_state": rnd_state,
        "verbose": verbose,
    }
    model = ensemble.BaggingRegressor(**model_params)
    model_name = type(model).__name__
    return model_name, model, model_params


#! TO DO:
# class myBaggingRegressor(BaseEstimator, RegressorMixin):
#     """
#     An example of BaggingRegressor.

#     Attributes of BaggingRegressor:
#     - base_estimator : object, default=None.
#         The base estimator to fit on random subsets of the dataset.
#         If None, then the base estimator is a DecisionTreeRegressor.
#     - base_estimator_params: the kernel used (string: rbf, poly, lin)
#     - params_grid: the actual kernel function
#     - params_dist : the data on which the LSSVM is trained (call it support vectors)
#     - bayes_space : the targets for the training data
#     - params_fit : coefficents of the support vectors
#     - intercept_ : intercept term
#     """

#     def __init__(
#         self,
#         base_estimator=None,
#         params_grid=baggging_params_grid,
#         params_dist=bagging_params_dist,
#         bayes_space=bagging_bayes_space,
#         params_fit=bagging_params_fit,
#     ):
#         """
#         Called when initializing the BaggingRegressor
#         """
#         self.base_estimator = base_estimator
#         self.params_grid = params_grid
#         self.params_dist = params_dist
#         self.bayes_space = bayes_space
#         self.params_fit = params_fit

#     def fit(self, X, y):
#         """ Fit a model on X features with target y . """
#         return self

#     def predict(self, X):
#         return None
