"""
    [08] EIGHTH step in project:
        LOAD raw TRAIN & VAL sets from the [./project/data/raw]
            or LOAD processed TRAIN & VAL from the [./project/data/cleaned]
        and CONCAT TRAIN and VALIDATION set to one DEVELOPMENT set

        and CREATE piped_model_@modelname:
            [
                pipeline_process
                + Estimator (model)
            ]

        and PROCESS DEVELOPMENT set using the pipeline_process
        and TRAIN model with selected best set of hyper-parameters on whole DEV
        and LOG results with MLflow Tracking
        and SAVE logs in the [.project/models/mlruns]
            or LOG results with Tensorboard
            and SAVE logs in the [.project/tensorboard]
        and SAVE results to the [./project/reports]
        and SAVE Trained Piped Model as "piped_model_@modelname.*"
            to the [./project/models/piped]

    Created on Jan 2021
    @author: mikhail.galkin
"""
"""
    ! EACH TIME when the project is opened newally
    ! you MUST TO START again the Tracking Server locally:
    see ./models/README.md
"""

# ------------------------------------------------------------------------------
# ------ F I N A L   T R A I N I N G   O N   D E V E L O P M E N T   S E T -----
# ------------------------------------------------------------------------------
#%% Load libraries
import sys
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib
import time
import winsound  # to beep when done
import mlflow

# from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from pprint import pprint

#%% Load project's stuff
sys.path.extend(["..", "../..", "../../.."])
import gscreen

# Load custom classes and utils
from gscreen.config import data_raw_dir
from gscreen.config import data_processed_dir
from gscreen.config import models_dir
from gscreen.config import pipelines_dir
from gscreen.plots import plot_residuals_errors
from gscreen.utils import accuracy as mare
from gscreen.utils import mlflow_set_exp_id
from gscreen.utils import mlflow_get_run_data
from gscreen.utils import mlflow_del_default_experiment
from gscreen.utils import load_pipeline
from gscreen.utils import load_outliers_mask
from gscreen.utils import make_example_df
from gscreen.model.fit_model import print_model
from gscreen.model.algos import get_bagging_model


#%%! Toggles to go through train process ---------------------------------------
rnd_state = 42
target = "rate"

#%% Print out ------------------------------------------------------------------
def print_toggles(
    pipeline_to_load,
    outliers_mask_to_load,
    train_wo_outliers,
    mlflow_tracking,
    log_residuals,
    save_mlmodel_separatly,
):
    print(f" ")
    print(f"Data processing pipeline: {pipeline_to_load}")
    print(f"Outliers mask: {outliers_mask_to_load}")
    print(f"Train w\o outliers: {train_wo_outliers}")
    print(f"Track modelwith MLflow: {mlflow_tracking}")
    print(f"Log model's residuals: {log_residuals}")
    print(f"Save trained model separatly: {save_mlmodel_separatly}")


# Print out libs versions
def print_versions():
    try:
        skl = Pipeline.__module__[: Pipeline.__module__.index(".")]
    except:
        skl = Pipeline.__module__
    print(f" ")
    print(f"MLflow: {mlflow.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Matplotlib: {sys.modules[plt.__package__].__version__}")
    print(f"Scikit-learn: {sys.modules[skl].__version__}")


#%% Load data ------------------------------------------------------------------
def get_data(outliers_mask, train_wo_outliers):
    """
    Load TRAIN & VALIDATION data
    and create DEVELOPMENT data set for final model training
    """
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
    #%% Final model: Prep development set
    # Construct DEV set
    print(f"Will be train without outlaiers. Prepare data to it...")
    if train_wo_outliers:
        df_dev = df_train[outliers_mask].append(df_val).reset_index(drop=True)
    else:
        df_dev = df_train.append(df_val).reset_index(drop=True)

    # Retrive X and y from dev
    X_train, y_train = df_train.loc[:, df_train.columns != target], df_train[target]
    X_val, y_val = df_val.loc[:, df_val.columns != target], df_val[target]
    X_dev, y_dev = df_dev.loc[:, df_dev.columns != target], df_dev[target]
    #%% Print out
    print(f"X_train: {X_train.shape[1]} variables & {X_train.shape[0]} records.")
    print(f"X_val: {X_val.shape[1]} variables & {X_val.shape[0]} records.")
    print(f"X_dev: {X_dev.shape[1]} variables & {X_dev.shape[0]} records.")

    return X_dev, y_dev, X_train, y_train, X_val, y_val


#%% Define some functions ------------------------------------------------------
def fit_pipeline_process(pipeline_process, X, save_pipeline_as):
    print(f"\nFit data processing pipeline on DEV...")
    pipeline_process.fit(X)
    print(f"Save data processing pipeline...")
    with open(pipelines_dir / save_pipeline_as, "wb") as f:
        joblib.dump(pipeline_process, f)
    print(f"Data processing pipeline saved into {pipelines_dir}.")
    return pipeline_process

def build_pipeline(pipeline_process, model):
    print(f"\nConstruct final pipeline to train...")
    pipeline = Pipeline(
        steps=[
            ("process", pipeline_process),
            ("model", model),
        ]
    )
    return pipeline


#%% Save model separatly -------------------------------------------------------
def save_mlmodel_aside(model, run_id):
    print(f"\nSave aside a trained model at MLflow's format...")
    if run_id is not None:
        t = mlflow_get_run_data(run_id)[4]
        time_point = f"{t[0]}{t[1]:02d}{t[2]:02d}-{t[3]:02d}{t[4]:02d}"
        folder = f"{time_point}_{run_id}"
    else:
        time_point = time.strftime("%Y%m%d-%H%M")
        model_name = type(model).__name__
        folder = f"{time_point}_{model_name}"

    path = models_dir / "mlmodels" / folder
    input_example, _ = make_example_df(num_rows=2)
    mlflow.sklearn.save_model(
        model,
        path,
        conda_env=None,
        mlflow_model=None,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        input_example=input_example,
    )
    print(f"MLflow model saved aside.You can find it in: {path} folder...")
    return folder


#%% Final model training -------------------------------------------------------
def train_model(
    model,
    X_dev,
    y_dev,
    X_train,
    y_train,
    X_val,
    y_val,
    mlflow_tracking=True,
    log_residuals=True,
    save_mlmodel_separatly=True,
):
    print(f"\nTrain final model on Development set...")

    tic = time.time()
    model_name = type(model).__name__
    if mlflow_tracking:
        # Setup MLflow tracking server
        exp_id = mlflow_set_exp_id("Model:Train")
        run_name = f"{model_name}"
        ## Enable autologging
        mlflow.sklearn.autolog()
        ##* Fit model with MLflow logging
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"Active run_id: {run_id} ...\n")
            model = model.fit(X_dev, y_dev)
            toc = time.time()
            ## Disable autologging
            mlflow.sklearn.autolog(disable=True)

            ##* Log custom metrics
            mare_on_dev = mare(y_dev, model.predict(X_dev))
            mare_on_train = mare(y_train, model.predict(X_train))
            mare_on_val = mare(y_val, model.predict(X_val))
            print(f"\nMARE on DEV: {mare_on_dev}")
            print(f"MARE on TRAIN: {mare_on_train}")
            print(f"MARE on VAL: {mare_on_val}")
            mlflow.log_metrics(
                {
                    "mare_on_dev": mare_on_dev,
                    "mare_on_train": mare_on_train,
                    "mare_on_val": mare_on_val,
                }
            )
            ##* Log custom plots
            if log_residuals:
                print(f"\nCalculate and log model's residuals...")
                fig = plot_residuals_errors(model, X_train, y_train, X_val, y_val)
                mlflow.log_figure(fig, "./plots/residuals_errors.png")
    else:
        ##* Fit trivial
        model = model.fit(X_dev, y_dev)
        toc = time.time()
        exp_id, run_id = None, None

    ## Evaluate time spent
    min, sec = divmod(toc - tic, 60)
    print(f"Model training took: {int(min)}min {int(sec)}sec\n")

    ## Save trained pipeline
    if save_mlmodel_separatly:
        folder = save_mlmodel_aside(model, run_id)
    else:
        print(f"No one model was NOT saved separatly...")
        folder = None

    print(f"\nExperiment ID: {exp_id}")
    print(f"Run ID: {run_id}")
    print(f"Folder: {folder}")

    return exp_id, run_id, folder


#%% Main function for main.py ==================================================
def main(
    pipeline_to_load="data_process_fitted_on_train.joblib",
    save_pipeline_as="data_process_fitted_on_dev.joblib",
    outliers_mask_to_load="outliers_mask_train_thold.joblib",
    train_wo_outliers=True,
    mlflow_tracking=True,
    log_residuals=True,
    save_mlmodel_separatly=True,
    n_estimators=80,
    n_jobs=-1,
):
    """Performs a final MODEL TRAINING on DEVELOPMENT set
    with a set of hyper-params found out on previous step.

    Args:
        * pipeline_to_load (str, optional): File with preprocessing pipeline.\
            Defaults to "data_process_fitted_on_train.joblib".
        * save_pipeline_as (str, optional): File name to save pipeline\
            fitted on development dataset.\
                Defaults to "data_process_fitted_on_train.joblib".
        * outliers_mask_to_load (str, optional): Mask to remove outliers.\
            Defaults to "outliers_mask_train_thold.joblib".
        * train_wo_outliers (bool, optional): Toogle to train withoutoutliers.\
            Defaults to True.
        * mlflow_tracking (bool, optional): Toggle to log model training process.\
            Defaults to True.
        * log_residuals (bool, optional): Toggle to estimate and log model's residuals.\
            Defaults to True.
        * save_mlmodel_separatly (bool, optional): Toggle to save model in MLflow\
            format into particular folder. Defaults to True.
        * n_estimators (int, optional): Number ov estimators in case Enssemble.\
            Defaults to 80.
    """
    print(f"-------------- START: Train model --------------")
    print_versions()
    print_toggles(
        pipeline_to_load,
        outliers_mask_to_load,
        train_wo_outliers,
        mlflow_tracking,
        log_residuals,
        save_mlmodel_separatly,
    )
    model_name, model, model_params = get_bagging_model(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    print_model(model_name, model, model_params)
    outliers_mask = load_outliers_mask(data_processed_dir, outliers_mask_to_load)
    X_dev, y_dev, X_train, y_train, X_val, y_val = get_data(
        outliers_mask,
        train_wo_outliers,
    )
    pipeline_process = load_pipeline(pipelines_dir, pipeline_to_load)
    pipeline_process = fit_pipeline_process(pipeline_process, X_dev, save_pipeline_as)
    pipeline = build_pipeline(pipeline_process, model)
    exp_id, run_id, folder = train_model(
        pipeline,
        X_dev,
        y_dev,
        X_train,
        y_train,
        X_val,
        y_val,
        mlflow_tracking,
        log_residuals,
        save_mlmodel_separatly,
    )
    if mlflow_tracking:
        mlflow_del_default_experiment()
    print(f"!!! DONE: Train model !!!")
    winsound.Beep(frequency=3000, duration=300)
    return exp_id, run_id, folder


#%% Workflow ===================================================================
if __name__ == "__main__":
    main()

#%%
