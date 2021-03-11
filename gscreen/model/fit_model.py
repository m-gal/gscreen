"""
    [07] SEVENTH step in project:
        [07.1] LOAD raw Development set from the [./project/data/raw] folder
            or [07.2] LOAD processed Development set from the [./project/data/processed]

        and SPLIT it onto Train and Validation sets
            or USE Cross-Validation technics

        [07.1] and TUNE selected model (architecture NN) with TrainModel_Pipeline
            [
                Clean_Pipeline
                + Transform_Pipeline
                + TuneModel_Pipeline
            ]
            or [07.2] TUNE selected model (architecture NN) with TuneModel_Pipeline

        and VALIDATE models with different sets of hyper-parameters
        and LOG results with MLflow Tracking
        and SAVE logs in the [.project/mlflow/runs] folder
            or LOG results with Tensorboard
            and SAVE logs in the [.project/tensorboard] folder
        and SELECT (make decision reffered to) best set of hyper-parameters
        and TRAIN Final Model on the whole Development dataset
        ans SAVE Final Model as "final_model.*" to the [./project/models] folder
        and SAVE TrainModel_Pipeline to the [./project/pipelines] folder


    Created on Jan 2021
    @author: mikhail.galkin
"""

# ------------------------------------------------------------------------------
# ------ T U N E   H Y P E R - P A R A M S   O N  T R A I N   S E T ------------
# ------------------------------------------------------------------------------
#%% Load libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
import winsound  # to beep when done
import mlflow

from skopt import BayesSearchCV
from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from pprint import pprint

#%% Load project's stuff
sys.path.extend(["..", "../..", "../../.."])
import gscreen

# Load custom classes and utils
from gscreen.config import pipelines_dir
from gscreen.config import data_processed_dir
from gscreen.config import models_dir
from gscreen.plots import plot_residuals_errors
from gscreen.utils import accuracy as mare
from gscreen.utils import calc_metrics
from gscreen.utils import mlflow_set_exp_id
from gscreen.utils import load_pipeline
from gscreen.utils import load_outliers_mask

from gscreen.utils import set_pd_options
from gscreen.utils import set_matlotlib_params

from gscreen.model.choose_model import get_data
from gscreen.model.choose_model import get_fractioned_data
from gscreen.model.choose_model import set_custom_scorer_cv

from gscreen.model.algos import get_bagging_model
from gscreen.model.algos import get_bagging_search_params as get_search_params

#%%! Toggles to flow through ---------------------------------------------------
rnd_state = 42
target = "rate"

#%% Print out ------------------------------------------------------------------
def print_toggles(
    pipeline_to_load,
    outliers_mask_to_load,
    train_wo_outliers,
    log_residuals,
    save_found_params,
    random_grid_search,
    bayesian_search,
    simple_grid_search,
):
    print(f" ")
    print(f"Data processing pipeline: {pipeline_to_load}")
    print(f"Outliers mask: {outliers_mask_to_load}")
    print(f"Train w\o outliers: {train_wo_outliers}")
    print(f"Log model's residuals: {log_residuals}")
    print(f"Save found params: {save_found_params}")
    print(f"Randomized GridSearch: {random_grid_search}")
    print(f"Bayesian Search: {bayesian_search}")
    print(f"Simple GridSearch: {simple_grid_search}")


def print_versions():
    try:
        skl = Pipeline.__module__[: Pipeline.__module__.index(".")]
    except:
        skl = Pipeline.__module__
    print(f" ")
    print(f"Numpy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Matplotlib: {sys.modules[plt.__package__].__version__}")
    print(f"Scikit-learn: {sys.modules[skl].__version__}")


def print_model(model_name, model, model_params):
    print(f" ")
    print(f"model_name: {model_name}")
    pprint(model)
    print(f"model_params:")
    pprint(model_params)


#%% Define some functions ------------------------------------------------------
def get_train_val_processed(df_train, df_val, pipeline):
    print(f"Fit processing pipeline and transform the train data...")
    X_train = pipeline.fit_transform(df_train.loc[:, df_train.columns != target])
    y_train = df_train[target]
    print(f"Transform the validation data...")
    X_val = pipeline.transform(df_val.loc[:, df_val.columns != target])
    y_val = df_val[target]
    return X_train, y_train, X_val, y_val


def log_custom_metrics(model, X_train, y_train, X_val, y_val):
    print(f"Calculate custom metrics and log them...")
    mare_on_train = mare(y_train, model.predict(X_train))
    mare_on_val = mare(y_val, model.predict(X_val))
    mlflow.log_metrics(
        {
            "mare_on_train": mare_on_train,
            "mare_on_val": mare_on_val,
        }
    )


def print_custom_metrics(model, X_train, y_train, X_val, y_val):
    print("TRAIN set:")
    calc_metrics(model, X=X_train, y=y_train)
    print("VALIDATION set:")
    calc_metrics(model, X=X_val, y=y_val)


def log_model_residuals(model, X_train, y_train, X_val, y_val):
    tic = time.time()
    print(f"\nCalculate model's residuals and log them...")
    fig = plot_residuals_errors(model, X_train, y_train, X_val, y_val)
    mlflow.log_figure(fig, "./plots/residuals_errors.png")
    min, sec = divmod(time.time() - tic, 60)
    print(f"Calculating residuals took: {int(min)}min {int(sec)}sec")


#%% ---------------------- RandomizedSearchCV ----------------------------------
def randomized_search_cv(
    X_fit,
    y_fit,
    X_train,
    y_train,
    X_val,
    y_val,
    model,
    params_dist,
    scorer,
    cv,
    n_jobs,
    random_search_params,
    log_residuals,
):
    if random_search_params[0]:
        print(f"\n-------------- Randomized Grid SearchCV started....")
        pprint(f"Parameters' distributions: {params_dist}")
        model_name = type(model).__name__
        # Setup MLflow tracking server
        exp_id = mlflow_set_exp_id("Model:Fit")
        run_name = f"{model_name}-rand"
        ## Enable autologging
        mlflow.sklearn.autolog(log_model_signatures=False)
        print(f"Autologging {model_name} started...")
        # Define RANDOMIZED grid search
        random_search = model_selection.RandomizedSearchCV(
            model,
            param_distributions=params_dist,
            n_iter=random_search_params[1],  # default 10
            scoring=scorer,
            n_jobs=n_jobs,
            cv=cv,
            refit=True,
            return_train_score=True,
            verbose=3,
            random_state=rnd_state,
        )
        ##* Fit model with MLflow logging
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
            tic = time.time()
            model_random_search = random_search.fit(
                X_fit,
                y_fit,
            )
            min, sec = divmod(time.time() - tic, 60)
            ## Disable autologging
            mlflow.sklearn.autolog(disable=True)
            # Log custom metrics and data
            print(f"Randomized grid search took: {int(min)}min {int(sec)}sec")
            print(f"Log custom metrics...")
            log_custom_metrics(model_random_search, X_train, y_train, X_val, y_val)
            if log_residuals:
                log_model_residuals(model_random_search, X_train, y_train, X_val, y_val)

        print(f"Randomized search: Best params are:\n {model_random_search.best_params_}")
        print(f"{model_name.title()}: Random search:")
        print_custom_metrics(model_random_search, X_train, y_train, X_val, y_val)
        winsound.Beep(frequency=2000, duration=300)
        return model, model_random_search.best_estimator_, model_random_search.best_params_
    else:
        print(f"\nSkip a Randomized Grid SearchCV....")
        return model, None, None


#%% ----------------------- Bayesian Optimization ------------------------------
def bayesian_search_cv(
    X_fit,
    y_fit,
    X_train,
    y_train,
    X_val,
    y_val,
    model,
    bayes_space,
    scorer,
    cv,
    n_jobs,
    bayesian_search_params,
    log_residuals,
):
    if bayesian_search_params[0]:
        print(f"\n-------------- Bayesian optimization of hyper-params started....")
        pprint(f"Parameters' space: {bayes_space}")
        model_name = type(model).__name__
        # Setup MLflow tracking server
        exp_id = mlflow_set_exp_id("Model:Fit")
        run_name = f"{model_name}-bayes"
        ## Enable autologging
        mlflow.sklearn.autolog(log_model_signatures=False)
        # Define bayesian search
        bayes_search = BayesSearchCV(
            model,
            search_spaces=bayes_space,
            n_iter=bayesian_search_params[1],  # default 50
            scoring=scorer,
            n_jobs=n_jobs,
            cv=cv,
            refit=True,
            return_train_score=True,
            verbose=3,
            random_state=rnd_state,
        )
        # Callback handler
        def on_step(optim_result):
            """ Print scores after each iteration while performing optimization """
            score = bayes_search.best_score_
            print(f"...current best score: {score}")
            if score <= 2:
                print("Interrupting!")
                return True

        ##* Fit model with MLflow logging
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
            tic = time.time()
            model_bayes_search = bayes_search.fit(
                X_fit,
                y_fit,
                callback=on_step,
            )
            min, sec = divmod(time.time() - tic, 60)
            ## Disable autologging
            mlflow.sklearn.autolog(disable=True)
            # Log custom metrics and data
            print(f"Bayesian search took: {int(min)}min {int(sec)}sec")
            print(f"Log custom metrics...")
            log_custom_metrics(model_bayes_search, X_train, y_train, X_val, y_val)
            if log_residuals:
                log_model_residuals(model_bayes_search, X_train, y_train, X_val, y_val)

        print(f"Bayesian search: Best params are:\n {model_bayes_search.best_params_}")
        print(f"{model_name.title()}: Bayesian search:")
        print_custom_metrics(model_bayes_search, X_train, y_train, X_val, y_val)
        winsound.Beep(frequency=2000, duration=300)
        return model, model_bayes_search.best_estimator_, model_bayes_search.best_params_
    else:
        print(f"\nSkip Bayesian Optimization....")
        return model, None, None


#%% --------------------------- GridSearchCV -----------------------------------
#! Dont forget to fine set up the Grid Search parameters in algo_*.py module
#! and reload that module
def grid_search_cv(
    X_fit,
    y_fit,
    X_train,
    y_train,
    X_val,
    y_val,
    model,
    params_grid,
    scorer,
    cv,
    simple_grid_search,
    n_jobs,
    log_residuals,
):
    if simple_grid_search:
        print(f"\n-------------- Simple Grid SearchCV started....")
        pprint(f"Parameters' grid: {params_grid}")
        model_name = type(model).__name__
        # Setup MLflow tracking server
        exp_id = mlflow_set_exp_id("Model:Fit")
        run_name = f"{model_name}-grid"
        # Enable autologging
        mlflow.sklearn.autolog(log_model_signatures=False)
        # Define SIMPLE grid search
        grid_search = model_selection.GridSearchCV(
            model,
            param_grid=params_grid,
            scoring=scorer,
            n_jobs=n_jobs,
            cv=cv,
            refit=True,
            return_train_score=True,
            verbose=3,
        )
        ##* Fit model with MLflow logging
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
            tic = time.time()
            model_grid_search = grid_search.fit(
                X_fit,
                y_fit,
            )
            min, sec = divmod(time.time() - tic, 60)
            # Disable autologging
            mlflow.sklearn.autolog(disable=True)
            # Log custom metrics and data
            print(f"Simple grid search took: {int(min)}min {int(sec)}sec")
            print(f"Log custom metrics...")
            log_custom_metrics(model_grid_search, X_train, y_train, X_val, y_val)
            if log_residuals:
                log_model_residuals(model_grid_search, X_train, y_train, X_val, y_val)

        print(f"Simple search: Best params are:\n {model_grid_search.best_params_}")
        print(f"{model_name.title()}: Simple search:")
        print_custom_metrics(model_grid_search, X_train, y_train, X_val, y_val)
        winsound.Beep(frequency=2000, duration=300)
        return model, model_grid_search.best_estimator_, model_grid_search.best_params_
    else:
        print(f"\nSkip a Simple Grid SearchCV....")
        return model, None, None


#%% ----------------------- Trivial Training -----------------------------------
def trivial_fit(
    X_fit,
    y_fit,
    X_train,
    y_train,
    X_val,
    y_val,
    model,
    log_residuals,
):
    print(f"\n-------------- Trivial model training w\o any parameters' searching started....")
    model_name = type(model).__name__
    # Setup MLflow tracking server
    exp_id = mlflow_set_exp_id("Model:Fit")
    run_name = f"{model_name}-None"
    ## Enable autologging
    mlflow.sklearn.autolog(log_model_signatures=False)
    print(f"Autologging {model_name} started...")
    ##* Fit model with MLflow logging
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        tic = time.time()
        model.fit(X_fit, y_fit)
        min, sec = divmod(time.time() - tic, 60)
        ## Disable autologging
        mlflow.sklearn.autolog(disable=True)
        # Log custom metrics and data
        print(f"Training took: {int(min)}min {int(sec)}sec")
        print(f"Log custom metrics...")
        log_custom_metrics(model, X_train, y_train, X_val, y_val)
        if log_residuals:
            log_model_residuals(model, X_train, y_train, X_val, y_val)

    print(f"{model_name.title()} model:")
    print_custom_metrics(model, X_train, y_train, X_val, y_val)
    winsound.Beep(frequency=2000, duration=300)
    return model


#%% Save parameters and model --------------------------------------------------
def save_params(model_name, rs_bp, gs_bp, bs_bp, save_found_params):
    cur_time = time.strftime("%Y%m%d-%H%M")
    dir = models_dir / "params"
    file = f"{model_name}_{cur_time}"

    if save_found_params:
        print(f"\nSave parameters found...")
        if rs_bp is not None:
            joblib.dump(rs_bp, dir / f"params_rand_srch_{file}.joblib")

        if bs_bp is not None:
            joblib.dump(bs_bp, dir / f"params_bayes_srch_{file}.joblib")

        if gs_bp is not None:
            joblib.dump(gs_bp, dir / f"params_grid_srch_{file}.joblib")
    else:
        print(f"No one parameters' sets found was saved...")


#%% Main function for main.py ==================================================
def main(
    pipeline_to_load="data_process_fitted_on_train.joblib",
    outliers_mask_to_load="outliers_mask_train_thold.joblib",
    train_wo_outliers=True,
    log_residuals=True,
    save_found_params=True,
    random_grid_search=True,
    bayesian_search=True,
    simple_grid_search=True,
    n_rand_sets_of_params=20,
    n_bayes_sets_of_params=20,
    n_jobs=-1,
    fraction=1.0,
    n_splits=5,
    n_repeats=1,
):
    """Performs FITTING a chosen MODEL on DEVELOPMENT set with searching an
    optimal hyper-parameters with Randomized, Baeysian or Simple grid search.

    Args:
        * pipeline_to_load (str, optional): File with preprocessing pipeline.\
            Defaults to "data_process_fitted_on_train.joblib".
        * outliers_mask_to_load (str, optional): Mask to remove outliers.\
            Defaults to "outliers_mask_train_thold.joblib".
        * train_wo_outliers (bool, optional): Toogle to train withoutoutliers.\
            Defaults to True.
        * log_residuals (bool, optional): Toogle to calculate and log model's rediduals.\
            Defaults to True.
        * save_found_params (bool, optional): Toggle to save parameters found.\
            Defaults to True.
        * random_grid_search (bool, optional): Toggle to perform Randomized grid search.\
            Defaults to True.
        * bayesian_search (bool, optional): Toggle to perform Bayesian Optimization.\
            Defaults to True.
        * simple_grid_search (bool, optional): Toggle to perform classical Grid grid search.\
            Defaults to True.
        * n_rand_sets_of_params (int, optional): Number of parameter settings that are sampled.\
            Defaults to 20.
        * n_bayes_sets_of_params (int, optional): Number of parameter settings that are sampled.\
            Defaults to 20.
        * n_jobs (int, optional): The number of jobs to run in parallel for params'\
            searching process. -1 means using all processors. Defaults to -1.
        * fraction (float, optional): Fraction of DEV set for model training.\
            Useful when DEV is big.\
            Defaults to 1.0.
        * n_splits (int, optional): Num. of splits in Cross-Validation strategy.\
            Defaults to 5.
        * n_repeats (int, optional): Num. of repeats for repeated CV.\
            Defaults to 1.
    """
    print(f"-------------- START: Fit model. Hyper-params searching --------------")
    print_versions()
    print_toggles(
        pipeline_to_load,
        outliers_mask_to_load,
        train_wo_outliers,
        log_residuals,
        save_found_params,
        random_grid_search,
        bayesian_search,
        simple_grid_search,
    )
    model_name, model, model_params = get_bagging_model(n_jobs=n_jobs)
    print_model(model_name, model, model_params)
    params_dist, params_grid, bayes_space = get_search_params()
    pipeline_process = load_pipeline(pipelines_dir, pipeline_to_load)
    outliers_mask = load_outliers_mask(data_processed_dir, outliers_mask_to_load)
    X_dev, y_dev, df_train, df_val = get_data(
        pipeline_process,
        outliers_mask,
        train_wo_outliers,
    )
    X_fit, y_fit = get_fractioned_data(X_dev, y_dev, fraction)
    X_train, y_train, X_val, y_val = get_train_val_processed(
        df_train,
        df_val,
        pipeline_process,
    )
    scorer, cv = set_custom_scorer_cv(n_splits, n_repeats)
    model, _, rs_bp = randomized_search_cv(
        X_fit,
        y_fit,
        X_train,
        y_train,
        X_val,
        y_val,
        model,
        params_dist,
        scorer,
        cv,
        n_jobs=n_jobs,
        random_search_params=(random_grid_search, n_rand_sets_of_params),
        log_residuals=log_residuals,
    )
    model, _, bs_bp = bayesian_search_cv(
        X_fit,
        y_fit,
        X_train,
        y_train,
        X_val,
        y_val,
        model,
        bayes_space,
        scorer,
        cv,
        n_jobs=n_jobs,
        bayesian_search_params=(bayesian_search, n_bayes_sets_of_params),
        log_residuals=log_residuals,
    )
    model, _, gs_bp = grid_search_cv(
        X_fit,
        y_fit,
        X_train,
        y_train,
        X_val,
        y_val,
        model,
        params_grid,
        scorer,
        cv,
        simple_grid_search,
        n_jobs=n_jobs,
        log_residuals=log_residuals,
    )
    if not random_grid_search and not simple_grid_search and not bayesian_search:
        model = trivial_fit(
            X_fit,
            y_fit,
            X_train,
            y_train,
            X_val,
            y_val,
            model,
            log_residuals=log_residuals,
        )
    save_params(model_name, rs_bp, gs_bp, bs_bp, save_found_params)
    print(f"!!! DONE: Fit model !!!")
    winsound.Beep(frequency=3000, duration=300)


#%% Workflow ===================================================================
if __name__ == "__main__":  #! Make sure that Tracking Server has been run.
    set_pd_options()
    set_matlotlib_params()
    main()
    gscreen.utils.reset_pd_options()
