""" Contains the functions used across the project.

    Created on Dec 07 2020
    @author: mikhail.galkin
"""

#%% Import needed python libraryies and project config info
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow

from pathlib import Path
from matplotlib import rcParams
from mlflow.tracking import MlflowClient

# from sklearn.metrics import max_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
from IPython.display import display


# ------------------------------------------------------------------------------
# ----------------------------- P A R A M E T E R S ----------------------------
# ------------------------------------------------------------------------------
#%% Set up: Pandas options
def set_pd_options():
    """ Set parameters for PANDAS to InteractiveWindow"""

    display_settings = {
        "max_columns": 40,
        "max_rows": 100,
        "width": 500,
        "max_info_columns": 500,
        "expand_frame_repr": True,  # Wrap to multiple pages
        # "float_format": lambda x: "%.5f" % x,
        "precision": 5,
        "show_dimensions": True,
    }
    print("Pandas options established are:")
    for op, value in display_settings.items():
        pd.set_option(f"display.{op}", value)
        option = pd.get_option(f"display.{op}")
        print(f"\tdisplay.{op}: {option}")


#%% Set up: Reset Pandas options
def reset_pd_options():
    """ Set parameters for PANDAS to InteractiveWindow """

    pd.reset_option("all")
    print("Pandas all options re-established.")


#%% Set up: Matplotlib params
def set_matlotlib_params():
    """Set parameters for MATPLOTLIB to InteractiveWindow"""

    rcParams["figure.figsize"] = 10, 5
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["xtick.labelsize"] = 10
    rcParams["ytick.labelsize"] = 11


# ------------------------------------------------------------------------------
# ----------------------------- U T I L I T S ----------------------------------
# ------------------------------------------------------------------------------
def accuracy(real_rates, predicted_rates):
    """Project's accuracy value estimator"""
    return np.average(abs(real_rates / predicted_rates - 1.0)) * 100.0


def drop_duplicats(df):
    """Drop fully duplicated rows"""
    n = df.duplicated().sum()  # Number of duplicated rows
    df.drop_duplicates(keep="first", inplace=True)
    df = df.reset_index(drop=True)
    print(f"Droped {n} duplicated rows")
    return df


def load_data(dir, file_to_load="train.csv", drop_duplicated=False):
    """Load data set"""
    df = pd.read_csv(
        dir / file_to_load,
        header=0,
        parse_dates=["pickup_date"],
        date_parser=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S"),
        nrows=None,
    )
    print(f"{file_to_load}: Loaded {len(df)} rows X {len(df.columns)} cols")

    if drop_duplicated:
        drop_duplicats(df)

    return df


def get_na_cols(df):
    """Get: Info about NA's in df"""
    print(f"\n#NA = {df.isna().sum().sum()}\n%NA = {df.isna().sum().sum()/df.size*100}")

    # View NA's through variables
    df_na = pd.concat(
        [df.isna().sum(), df.isna().sum() / len(df) * 100, df.notna().sum()],
        axis=1,
        keys=["# NA", "% NA", "# ~NA"],
    ).sort_values("% NA")
    display(df_na)


def load_pipeline(dir, pipeline_to_load="data_process_fitted_on_train.joblib"):
    print(f"\nLoad data processing pipeline...")
    with open(dir / pipeline_to_load, "rb") as f:
        pipeline = joblib.load(f)
    return pipeline


def load_outliers_mask(dir, outliers_mask_to_load):
    print(f"Load outliers mask...")
    outliers_mask = joblib.load(dir / outliers_mask_to_load)
    print(f"Outliers mask have: {sum(outliers_mask==0)} outliers.")
    return outliers_mask


def get_model_paths(exp_id, run_id, models_dir, folder):
    model_paths = []
    model_names = []
    # Get path to model in mlruns folder
    if (exp_id and run_id) is not None:
        model_file = f"{exp_id}/{run_id}/artifacts/model/model.pkl"
        model_paths.insert(0, models_dir/"mlruns"/model_file)
        model_names.insert(0, f"{exp_id}_{run_id}")
    else:
        model_paths.insert(0, None)
        model_names.insert(0, None)
    # Get path to model in mlmodels folder
    if folder is not None:
        model_paths.insert(1,  models_dir/"mlmodels"/folder/ "model.pkl")
        model_names.insert(1, f"{run_id}")
    else:
        model_paths.insert(1, None)
        model_names.insert(1, None)
    # Print out paths
    print(f" ")
    for model_path in model_paths:
        print(model_path)
    return model_paths, model_names


def load_model(model_path):
    print(f"\nLoad early trained and saved model...")
    print(f"{model_path}")
    with open(model_path, "rb") as f:
        model = joblib.load(f)
        return model


def calc_metrics(model, X, y, log_target=False):
    """Calculates result metrics"""

    from sklearn.metrics import max_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    y_true = y
    y_pred = model.predict(X)
    if log_target:
        y_pred = np.exp(y_pred)

    me = max_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mare = accuracy(y_true, y_pred)

    print(f"\tMax Error: {me}")
    print(f"\tMean Absolute Error: {mae}")
    print(f"\tRoot Mean Squared Error: {rmse}")
    print(f"\tMean Absolute Ratio Error: {mare}")


def make_example_df(num_rows=1):
    """Create example df for checking model prediction"""
    example = {}
    real_rate = []
    if num_rows == 1:
        # Take the 25 row from train set: rate=4.55063
        example["valid_miles"] = [87.90]
        example["transport_type"] = ["VAN"]
        example["weight"] = [10032]
        example["pickup_date"] = ["2018-01-02 07:00:00"]
        example["origin_kma"] = ["IL_CHI"]
        example["destination_kma"] = ["IL_JOL"]
        real_rate = 4.55063
    if num_rows == 2:
        # Take the 296701 row from train set: rate=10.64801
        # Take the 4989 row from validation set: rate=4.58994
        example["valid_miles"] = [32.87, 577.34998]
        example["transport_type"] = ["VAN", "REEFER"]
        example["weight"] = [20000, 36085]
        example["pickup_date"] = [
            "2020-10-27 14:00:00",
            "2020-11-13 05:00:00",
        ]
        example["origin_kma"] = ["CA_ONT", "TN_KNO"]
        example["destination_kma"] = ["CA_ONT", "PA_HAR"]
        real_rate = [10.64801, 4.58994]

    # Pipeline fitted on data frame, due that convert to df
    example_df = pd.DataFrame.from_dict(example, orient="columns")
    return example_df, real_rate


# ------------------------------------------------------------------------------
# ----------------------------- M L   F L O W ----------------------------------
# ------------------------------------------------------------------------------
def mlflow_set_server_local(experiment):
    print(f"\n!!! MAKE SURE THAT TRACKING SERVER HAS BEEN RUN !!!\n")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    exp_info = MlflowClient().get_experiment_by_name(experiment)

    print(f"Current tracking uri: {mlflow.get_tracking_uri()}")
    print(f"Current registry uri: {mlflow.get_registry_uri()}")

    if exp_info:
        exp_id = exp_info.experiment_id
    else:
        exp_id = MlflowClient().create_experiment(experiment)

    return exp_id


def mlflow_set_exp_id(experiment: str):
    print(f"Setup MLflow tracking server...")
    exp_id = mlflow_set_server_local(experiment)
    print(f"\nExperiment ID: {exp_id}")
    return exp_id


def mlflow_get_run_data(run_id):
    import time

    """Fetch params, metrics, tags, and artifacts in the specified run
    for MLflow Tracking
    """
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    start_time = client.get_run(run_id).info.start_time
    start_time = time.localtime(start_time / 1000.0)
    run_data = (data.params, data.metrics, tags, artifacts, start_time)
    return run_data


def mlflow_del_default_experiment():
    print(f"\nDelete 'Default' experiment from MLflow loggs...")
    try:
        default_id = mlflow.get_experiment_by_name("Default").experiment_id
        default_loc = mlflow.get_experiment_by_name("Default").artifact_location
        mlflow.delete_experiment(default_id)
        print(f"'Default' experiment located:'{default_loc}' was deleted.\n")
    except Exception as e:
        print(f"'Default' experiment doesnt exist: {e}\n")


# def run_mlflow_server_local():  #! Does not finished
#     # TODO
#     import subprocess

#     command = f"mlflow server \
#     --backend-store-uri sqlite:///mltracking/mlruns.db \
#     --default-artifact-root file://{tracking_dir}/mlruns"
#     print(command)

#     proc = subprocess.Popen(
#         [
#             "cd",
#             project_dir,
#             "&",
#             "C:/Users/User/anaconda3/Scripts/activate",
#             "&",
#             "conda activate py38_ml",
#             "&",
#             command,
#         ],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         shell=True,  # specify the to create a new shell
#     )
#     out, err = proc.communicate()
#     print(out.decode("latin-1"))

#     subprocess.call(
#         ["cd", project_dir, "&", "C:/Users/User/anaconda3/Scripts/activate"], shell=True
#     )


def mlflow_set_registry_local(model_name, tracking_dir, registry_db="mlregistry.db"):
    ## Setup the registry server URI.
    # $ By deafult "sqlite:///mlregistry.db" it create <mlregistry.db>
    # $ and [mlruns] folder in parent irectory
    registry_uri = f"sqlite:///{tracking_dir}\{registry_db}"
    mlflow.tracking.set_tracking_uri(registry_uri)

    # Set given experiment as active experiment.
    # If experiment does not exist, create an experiment with provided name.
    exp_id = mlflow.set_experiment(experiment_name=model_name)

    # The URIs should be different
    # assert mlflow.get_tracking_uri() != mlflow.get_registry_uri()

    def print_mlflow_manual():
        #%% Print out
        print(
            f"""
        TO RUN MLflow Tracking Server locally:
        Open in Integrated Terminal the root project's folder gscreen
        and start MLflow Tracking Server (command is one line):
        >mlflow server --backend-store-uri sqlite:///models/mlregistry.db \\
        --default-artifact-root file://t:/DataProjects/_pets/gscreen/models/mlruns
        """
        )
        print(f"Current tracking uri: {mlflow.get_tracking_uri()}")
        print(f"Current registry uri: {mlflow.get_registry_uri()}")

    print_mlflow_manual()
    return exp_id


def mlflow_set_tracking_local(model_name, tracking_dir):
    ## Setup the tracking server URI.
    ## Should be like "file:///...." and MUST BE NAMED AS 'mlruns'
    tracking_uri = f"file:///{tracking_dir}\mlruns"
    mlflow.tracking.set_tracking_uri(tracking_uri)

    # Set given experiment as active experiment.
    # If experiment does not exist, create an experiment with provided name.
    exp_id = mlflow.set_experiment(experiment_name=model_name)

    return exp_id
