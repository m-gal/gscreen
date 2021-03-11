"""
    Helps to get MLflow's runs results from MLflow Registry
    for deeper analysis can be done on python environment
    and make one off prediction with MLflow Staged (not deployed) model
    * MLflow Trackig server must be runed!

    Created on Jan 2021
    @author: mikhail.galkin
"""
#%% Import libs
import sys
import mlflow
import pandas as pd
from pprint import pprint
from mlflow.tracking import MlflowClient

#%% Load project's stuff
sys.path.extend(["..", "../..", "../../.."])
import gscreen
from gscreen.config import models_dir
from gscreen.utils import make_example_df


#%% Set the tracking server URI ------------------------------------------------
def set_tracking_uri():
    # This does not affect the currently active run (if one exists),
    # but takes effect for successive runs.
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Check
    tracking_uri = mlflow.get_tracking_uri()
    print(f"\nCurrent tracking uri: {tracking_uri}")


#%%# Select the run of the experiment ------------------------------------------
def find_best_model(experiment_id, view_metric):  # TODO doesnot work
    df_runs = mlflow.search_runs(experiment_ids=experiment_id)
    print(f"\nNumber of all runs done: {len(df_runs)}")
    # Sort
    df_runs.sort_values([view_metric], ascending=True, inplace=True)
    print(df_runs[["run_id", view_metric, "tags.estimator_name", "end_time"]].head())
    # Get the best one
    # The best model has a specific runid that can be used to execute the deployment.
    runid_best = df_runs["run_id"].head(1).values[0]
    best_metrics = df_runs[[view_metric]].head(1).values[0]
    print(f"The best run_id: {runid_best} with best metrics = {best_metrics}")
    return runid_best


#%% Fetching an MLflow Model from the Model Registry ---------------------------
def get_specific_model_version(model_name, model_version=1):
    """
    Fetch a specific model version:
    To fetch a specific model version,
    just supply that version number as part of the model URI.
    """
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    return model


def get_latest_model_version(model_name, stage="Production"):
    """
    Fetch the latest model version in a specific stage:
    To fetch a model version by stage, simply provide the model stage
    as part of the model URI, and it will fetch the most recent version
    of the model in that stage.
    """
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
    return model


def listing_all_models():
    """
    Listing and Searching MLflow Models:
    You can fetch a list of all registered models in the registry
    with a simple method.
    """
    client = MlflowClient()
    for rm in client.list_registered_models():
        pprint(dict(rm), indent=4)


def listing_models_by_name(model_name):
    """
    With hundreds of models, it can be cumbersome to peruse the results returned
    from this call. A more efficient approach would be to search for a specific
    model name and list its version details using search_model_versions() method
    and provide a filter string such as "name='model_name'"
    """
    for mv in MlflowClient().search_model_versions(f"name='{model_name}'"):
        pprint(dict(mv), indent=4)


#%% Main -----------------------------------------------------------------------
def main(
    models_dir,
    experiment_id=5,
    model_version=1,
    model_name="gscreen_model",
    stage="Production",
    view_metric="metrics.mare_on_dev",
):
    set_tracking_uri()
    find_best_model(experiment_id, view_metric)
    # model = get_specific_model_version(model_name, model_version=1)
    # listing_all_models()
    # listing_models_by_name(model_name)
    ## Define data frame containing two hand-crafted inputs
    example_df, real_rate = make_example_df(num_rows=2)
    example_df
    model = get_latest_model_version(model_name, stage="Production")
    prediction = model.predict(example_df)
    # Print out
    print(f"Real Rates is {real_rate}")
    print(f"Prediction of Rate is: {list(prediction)}")
    print(f"Mean Absolute Ratio Error: {gscreen.utils.accuracy(real_rate, prediction)}")


#%%# Predict on a Pandas DataFrame ---------------------------------------------
if __name__ == "__main__":
    main()

#%%
