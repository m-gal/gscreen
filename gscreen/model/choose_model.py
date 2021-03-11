"""
    [06] SIXTH step in project:
        LOAD raw TRAIN & VAL sets from the [./project/data/raw]
            or LOAD processed TRAIN & VAL from the [./project/data/cleaned]
        and LOAD Pipeline_Process for data processing
        and USE Cross-Validation technics on TRAIN set
        and TRAIN number of not-tuned basis models (architectures NN)
        and VALIDATE trained number of models on VALIDATION set
            (or use DEVELOPMENT set with Cross-Validation)
        and SAVE results to the [./project/reports]
            or LOG results with MLflow Tracking
            and SAVE logs in the [.project/models/mlruns]
                or LOG results with Tensorboard
                and SAVE logs in the [.project/tensorboard]
        and CHOOSE (make decision reffered to) best model's type (architecture NN)

    Created on Dec 2020
    @author: mikhail.galkin
"""
"""
    ! EACH TIME when the project is opened newally
    ! you MUST TO START again the Tracking Server locally:
    see ./models/README.md
"""

# ------------------------------------------------------------------------------
# ----------------- C H O O S I N G   M O D E L   T Y P E ----------------------
# ------------------------------------------------------------------------------
#%% Load libraries
import sys
import pandas as pd
import mlflow
import joblib
import time
import winsound

from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

# useful as a simple baseline to compare with other (real) regressors.
from sklearn.dummy import DummyRegressor
from sklearn import linear_model  # Simple, Ridge, Lasso, Elastic, SGD
from sklearn import svm  # Support Vector Machines.
from sklearn import tree  # Canonical Decision tree & Extremely randomized tree
from sklearn import ensemble  # RF, Gradient Boosting, AdaBoost
from sklearn import neighbors  # Models based on k-nearest neighbors.
from sklearn import neural_network  # Multi Layers Perception
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb

#%% Load project's stuff
sys.path.extend(["..", "../..", "../../.."])
import gscreen

# Load custom classes and utils
from gscreen.config import data_raw_dir
from gscreen.config import data_processed_dir
from gscreen.config import pipelines_dir

from gscreen.utils import mlflow_set_exp_id
from gscreen.utils import load_pipeline
from gscreen.utils import load_outliers_mask
from gscreen.utils import set_pd_options
from gscreen.utils import set_matlotlib_params

#%%! Toggles to flow through ---------------------------------------------------
rnd_state = 42
target = "rate"


#%% Print out ------------------------------------------------------------------
def print_versions():
    try:
        skl = Pipeline.__module__[: Pipeline.__module__.index(".")]
    except:
        skl = Pipeline.__module__
    print(f" ")
    print(f"Pandas: {pd.__version__}")
    print(f"MLflow: {mlflow.__version__}")
    print(f"Scikit-learn: {sys.modules[skl].__version__}")
    print(f"XGBoost: {xgb.__version__}")


def print_toggles(
    pipeline_to_load,
    outliers_mask_to_load,
    train_wo_outliers,
    mlflow_tracking,
):
    print(f" ")
    print(f"Data processing pipeline: {pipeline_to_load}")
    print(f"Outliers mask: {outliers_mask_to_load}")
    print(f"Train w\o outliers: {train_wo_outliers}")
    print(f"MLflow Tracking: {mlflow_tracking}")


#%% Load data ------------------------------------------------------------------
def get_data(pipeline_process, outliers_mask, train_wo_outliers=True):
    print(f"\nLoad raw data...")
    df_train = gscreen.utils.load_data(
        dir=data_raw_dir,
        file_to_load="train_20201221.csv",
        drop_duplicated=False,
    )
    df_val = gscreen.utils.load_data(
        dir=data_raw_dir,
        file_to_load="validation_20201221.csv",
        drop_duplicated=False,
    )
    #%% Construct DEV set on which we will choose model via CV
    print(f"Will be train without outlaiers. Prepare data to it...")
    if train_wo_outliers:
        df_dev = df_train[outliers_mask].append(df_val).reset_index(drop=True)
        df_train = df_train[outliers_mask].reset_index(drop=True)
    else:
        df_dev = df_train.append(df_val).reset_index(drop=True)

    #%% Retrive X and y
    # Get info
    print(
        f"""Shape of data:
    df_train={df_train.shape} : {len(df_train) / (len(df_train) + len(df_val))}
    df_val={df_val.shape} : {len(df_val) / (len(df_train) + len(df_val))}
    df_dev={df_dev.shape} : {len(df_dev) / (len(df_train) + len(df_val))}
    """
    )
    #%% Process data
    print(f"Process development data...")
    X_dev = pipeline_process.fit_transform(df_dev.loc[:, df_dev.columns != target])
    y_dev = df_dev[target]
    print(f"Processed X_dev have {X_dev.shape[1]} features and {X_dev.shape[0]} records.")

    # Assing names for df
    df_train.name = "TRAIN"
    df_val.name = "VAL"

    return X_dev, y_dev, df_train, df_val


#%% Define some functions ------------------------------------------------------
def set_custom_scorer_cv(n_splits=5, n_repeats=2):
    """Define custom scorer and Cross-Validation strategy

    Args:
        n_splits (int, optional): Num. of splits in Cross-Validation strategy.
        Defaults to 5.
        n_repeats (int, optional): Num. of repeats for repeated CV.
        Defaults to 1.

    Returns:
        objects: custom scorer, model_selection.RepeatedKFold
    """
    print(f"\nCreate custom scorer...")
    scorer = make_scorer(
        score_func=gscreen.utils.accuracy,
        greater_is_better=True,  # Whether score_func is a score function (default),
        # meaning high is good, or a loss function, meaning low is good.
    )
    #%% Define cross-validation parameters
    print(f"Define Cross-Validation strategy...")
    cv = model_selection.RepeatedKFold(
        n_splits=n_splits,
        # Repeats K-Fold:  n times with different randomization in each repetition.
        n_repeats=n_repeats,
        random_state=rnd_state,
    )
    return scorer, cv


def get_baseline_score(y):
    #%% Calculate baseline model's performance estimate
    print(f"\nCalculate given baseline model score...")
    baseline_score = gscreen.utils.accuracy(
        real_rates=y,
        predicted_rates=y.mean(),
    )
    print(f"Baseline model performance estimate = {baseline_score}")
    return baseline_score


def get_fractioned_data(X, y, fraction):
    if fraction == 1:
        return X, y
    else:
        print(f"\nGet fraction of {fraction} data fo modelig...")
        X_frac, _, y_frac, _ = train_test_split(
            X,
            y,
            train_size=fraction,
            random_state=rnd_state,
            shuffle=True,
        )
        print(f"Records in X_frac={len(y_frac)} : {len(y_frac) / (len(y))}")
        return X_frac, y_frac


#%% List of basic models with default settings for first attempt fitting -------
def get_list_of_basic_models():
    print(f"\nCreate list of basic models will pass through...")
    #! BE CAREFUL ! May take more time then expect or freeze process
    # @ Retun NaN in our case
    basic_models = [
        DummyRegressor(),
        # # ^ ----------------------------------------- Classical linear regressors
        # linear_model.LinearRegression(),
        # linear_model.Ridge(alpha=0.5, random_state=rnd_state),
        # # linear_model.SGDRegressor(random_state=rnd_state),
        # # ^ ---------------------------------- Regressors with variable selection
        # linear_model.Lasso(alpha=0.1, random_state=rnd_state),
        # linear_model.ElasticNet(random_state=rnd_state),
        # # @ linear_model.LassoLars(alpha=0.1, random_state=rnd_state),
        # # ^ ------------------------------------------------- Bayesian regressors
        # # @ linear_model.BayesianRidge(),
        # # @ linear_model.ARDRegression(),
        # # ^ ------------------------------------------- Outlier-robust regressors
        # # @ linear_model.HuberRegressor(),
        # linear_model.RANSACRegressor(random_state=rnd_state),
        # # ^ -----------------------Generalized linear models (GLM) for regression
        # linear_model.TweedieRegressor(power=0, alpha=0.5, link="auto"),
        # # linear_model.PoissonRegressor(),
        # linear_model.GammaRegressor(),
        # # ^ ------------------------------------------------------- Miscellaneous
        # linear_model.PassiveAggressiveRegressor(random_state=rnd_state),
        # # @ KernelRidge(),
        # ## --------------------------------------------- Support Vector Machines
        # # svm.LinearSVR(random_state=rnd_state),
        # #! svm.NuSVR(), #! CAN FREEZE
        # #! svm.SVR(),  #! CAN FREEZE
        # ^ ------------------------------------------------------ Decision Trees
        tree.DecisionTreeRegressor(random_state=rnd_state),
        tree.ExtraTreeRegressor(random_state=rnd_state),
        # ^ ---------------------------------------------------- Ensemble Methods
        # @ ensemble.HistGradientBoostingRegressor(random_state=rnd_state),
        # ensemble.AdaBoostRegressor(n_estimators=50, random_state=rnd_state),
        # ensemble.BaggingRegressor(n_estimators=50, random_state=rnd_state),
        # ensemble.ExtraTreesRegressor(n_estimators=100, random_state=rnd_state),  #! CAN BE LOOONG
        # ensemble.RandomForestRegressor(n_estimators=100, random_state=rnd_state),  #! CAN BE LOOONG
        # ensemble.GradientBoostingRegressor(n_estimators=100, random_state=rnd_state),
        # xgb.XGBRegressor(n_estimators=1000, random_state=rnd_state),
        # ^ --------------------------------------------------- Nearest Neighbors
        # @ neighbors.KNeighborsRegressor(),
        # ^ ----------------------------------------------- Neural network models
        # neural_network.MLPRegressor(hidden_layer_sizes=100, random_state=rnd_state),
    ]

    return basic_models


#%% Model selection ------------------------------------------------------------
def choose_model(X, y, fraction, n_splits, n_repeats, n_jobs, mlflow_tracking):
    print(f"\nStart model selection...")

    # Define dataset for modeling
    X_fit, y_fit = get_fractioned_data(X, y, fraction)

    # Get list of basic models will being estimated
    basic_models = get_list_of_basic_models()

    # # Create dict for modeling results
    # baseline_score = get_baseline_score(y)
    #  = {
    #     "Baseline": {
    #         "cv_score_mean": baseline_score,
    #         "cv_score_std": None,
    #         "time_spent": None,
    #     }
    # }
    basic_results = {}
    # Define num. of CV splits and K-repeats
    scorer, cv = set_custom_scorer_cv(n_splits, n_repeats)

    # Starts MLflow Tracking
    if mlflow_tracking:
        # Setup MLflow tracking server
        exp_id = mlflow_set_exp_id("Model:Choose")

    # Run loop through list of basic models
    for basic_model in basic_models:
        model_name = type(basic_model).__name__
        print(f"Modeling {model_name}...")
        # Fit each basic model via cross-validation
        tic = time.time()
        basic_model_scores = model_selection.cross_val_score(
            X=X_fit,
            y=y_fit,
            estimator=basic_model,
            scoring=scorer,
            cv=cv,
            n_jobs=n_jobs,  # -1 means using all processors
            verbose=0,  # The verbosity level. default=0
        )
        # Calculate time spent
        min, sec = divmod(time.time() - tic, 60)
        time_spent = f"{int(min)}min {int(sec)}sec"
        # Save results to dict
        basic_results.update(
            {
                basic_model: {
                    "cv_score_mean": basic_model_scores.mean(),
                    "cv_score_std": basic_model_scores.std(),
                    "time_spent": time_spent,
                }
            }
        )

        ##* Log models with MLflow logging
        if mlflow_tracking:
            print(f"\tLogging {model_name} results to runs...")
            with mlflow.start_run(experiment_id=exp_id, run_name=model_name):
                mlflow.log_params(
                    {
                        "time_spent": time_spent,
                        "fraction": fraction,
                        "cv_n_splits": n_splits,
                        "cv_n_repeats": n_repeats,
                        "random_state": rnd_state,
                    }
                )
                mlflow.log_metrics(
                    {
                        "cv_score_mean": basic_model_scores.mean(),
                        "cv_score_std": basic_model_scores.std(),
                    }
                )

    # Sort dict by score
    basic_results = dict(
        sorted(
            basic_results.items(),
            key=lambda x: (
                x[1]["cv_score_mean"],
                x[1]["cv_score_std"],
                x[1]["time_spent"],
            ),
        )
    )
    print(" ")
    print("-------------- Models' rating --------------")
    pprint(basic_results, sort_dicts=False)
    # Pick up best model from basic set of
    chosen_model = list(basic_results.keys())[0]

    return basic_results, chosen_model


#%% Train pipeline and get results from chosen model ---------------------------
def eval_chosen_model(chosen_model, df_train, df_val, pipeline):
    """Try to estimating potential model performance
    with training chosen model on TRAIN set
    and evaluation on VALIDATION set

    Args:
        chosen_model (sklearn.oject): Model chosen as a best
        df_train (pandas DF): Train set
        df_val (pandas DF): Validation set

    Returns:
        [figures]: Makes 2 plots real vs. predicted and logs them in MLflow
    """
    model_name = type(chosen_model).__name__
    print(f"\nEstimate potential score for best model: {model_name}...")
    print(f"Fit processing pipeline and transform the train data...")
    X_train = pipeline.fit_transform(df_train.loc[:, df_train.columns != target])
    y_train = df_train[target]
    print(f"Transform the validation data...")
    X_val = pipeline.transform(df_val.loc[:, df_val.columns != target])
    y_val = df_val[target]

    print(f"Train chosen {model_name} on train set...")
    tic = time.time()
    chosen_model.fit(X_train, y_train)
    # Calculate time spent
    min, sec = divmod(time.time() - tic, 60)
    time_spent = f"{int(min)}min {int(sec)}sec"

    def make_plot(X, y, df, time):
        print(f"Make prediction on {df.name} set...")
        y_predicted = chosen_model.predict(X)
        score = gscreen.utils.accuracy(
            real_rates=y,
            predicted_rates=y_predicted,
        )
        print(f"On {df.name} set the {model_name} gives MARE = {score}")
        # Combain prediction to df
        pred_df = pd.concat(
            [
                df["pickup_date"],
                df[target],
                pd.DataFrame({"predict": y_predicted}),
            ],
            axis=1,
        )
        plot = (
            pred_df.groupby([pred_df["pickup_date"].dt.to_period(time)])
            .mean()
            .plot(
                title=f"{model_name} for {df.name} set. MARE = {score}",
                ylim=(0, 5),
            )
        )
        plot
        return plot, score

    # Make plots
    plots = [None, None]
    scores = [None, None]
    df_names = [df_train.name, df_val.name]
    plots[0], scores[0] = make_plot(X_train, y_train, df_train, time="W")
    plots[1], scores[1] = make_plot(X_val, y_val, df_val, time="D")

    eval_results = (model_name, plots, scores, df_names, time_spent)

    return eval_results


#%% Log results for best model -------------------------------------------------
def log_evaluated_results(
    eval_results,
    mlflow_tracking,
    fraction,
    n_splits,
    n_repeats,
):
    """Logs the genereted plot

    Args:
        plots (list of pandas.plot): genereted plots via Pandas
        df_names (list of strings): short names of dataset used
        model_name (string): short name of chosen model
    """
    if mlflow_tracking:
        print(f"Log artifacts for model evaluated...")
        model_name = eval_results[0]
        plots = eval_results[1]
        scores = eval_results[2]
        df_names = eval_results[3]
        time_spent = eval_results[4]

        exp_id = mlflow_set_exp_id("Model:Choose")
        run_name = f"{model_name} : Best"
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
            for i in range(2):
                fig = plots[i].get_figure()
                path = f"./plots/{model_name}_on_{df_names[i]}.png"
                mlflow.log_figure(fig, path)
                mlflow.log_params(
                    {
                        "time_spent": time_spent,
                        "fraction": fraction,
                        "cv_n_splits": n_splits,
                        "cv_n_repeats": n_repeats,
                        "random_state": rnd_state,
                    }
                )
                mlflow.log_metrics(
                    {
                        "score_on_train": scores[0],
                        "score_on_val": scores[1],
                    }
                )


#%% Main function for main.py ==================================================
def main(
    pipeline_to_load="data_process_fitted_on_train.joblib",
    outliers_mask_to_load="outliers_mask_train_thold.joblib",
    train_wo_outliers=True,
    mlflow_tracking=True,
    eval_chosen=True,
    n_jobs=-1,
    fraction=1.0,
    n_splits=5,
    n_repeats=1,
):
    """Performs CHOOSING a best MODEL for further modeling with Cross-Validation
    on DEVELOPMENT dataset.

    Args:
        * pipeline_to_load (str, optional): File with preprocessing pipeline.\
            Defaults to "data_process_fitted_on_train.joblib".
        * outliers_mask_to_load (str, optional): Mask to remove outliers.\
            Defaults to "outliers_mask_train_thold.joblib".
        * train_wo_outliers (bool, optional): Toogle to train withoutoutliers.\
            Defaults to True.
        * mlflow_tracking (bool, optional): Toggle to log model params and metrics.\
            Defaults to True.
        * eval_chosen (bool, optional): Toggle to train chosen model on whole\
            TRAIN set and eval on VAL set logging results to run.\
            Defaults to True.
        * fraction (float, optional): Fraction of DEV set for model training.\
            Useful when DEV set is big.\
            Defaults to 1.0.
        * n_jobs (int, optional): The number of jobs to run in parallel\
            in cross_val_score process. -1 means using all processors.\
            Defaults to -1.
        * n_splits (int, optional): Num. of splits in Cross-Validation strategy.\
            Defaults to 5.
        * n_repeats (int, optional): Num. of repeats for repeated CV.\
            Defaults to 1.
    """
    print(f"-------------- START: Choose model --------------")
    print_versions()
    print_toggles(
        pipeline_to_load,
        outliers_mask_to_load,
        train_wo_outliers,
        mlflow_tracking,
    )
    pipeline_process = load_pipeline(pipelines_dir, pipeline_to_load)
    outliers_mask = load_outliers_mask(data_processed_dir, outliers_mask_to_load)
    X_dev, y_dev, df_train, df_val = get_data(
        pipeline_process,
        outliers_mask,
        train_wo_outliers,
    )
    _, chosen_model = choose_model(
        X_dev,
        y_dev,
        fraction,
        n_splits,
        n_repeats,
        n_jobs,
        mlflow_tracking,
    )
    if eval_chosen:
        eval_results = eval_chosen_model(
            chosen_model,
            df_train,
            df_val,
            pipeline_process,
        )
        log_evaluated_results(
            eval_results,
            mlflow_tracking,
            fraction,
            n_splits,
            n_repeats,
        )
    print(f"!!! DONE: Choose model !!!")
    winsound.Beep(frequency=3000, duration=300)


#%% Workflow ===================================================================
#! Make sure that Tracking Server has been run.
if __name__ == "__main__":
    set_pd_options()
    set_matlotlib_params()
    main()
    gscreen.utils.reset_pd_options()

#%%
