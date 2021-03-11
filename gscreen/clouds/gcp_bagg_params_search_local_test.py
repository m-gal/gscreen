"""

"""
#%%
model_name = "bagging"
random_state = 42
bayesian_search = False

print(f"model_name: {model_name}")
print(f"bayesian_search: {bayesian_search}")

#%%
import argparse
import subprocess

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from pathlib import Path

from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn import tree
from sklearn import ensemble

try:
    skl = Pipeline.__module__[: Pipeline.__module__.index(".")]
except:
    skl = Pipeline.__module__

print(f"Numpy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {sys.modules[plt.__package__].__version__}")
print(f"Scikit-learn: {sys.modules[skl].__version__}")

#%% Define project's paths
data_processed_dir = "t:/DataProjects/_pets/gscreen/data/processed/"
models_dir = "t:/DataProjects/_pets/gscreen/models"

STORAGE_BUCKET = "petprojects\\gscreen\\data"

DATA_PATH_X_fit = "X_fit.joblib.compressed"
DATA_PATH_y_fit = "y_fit.joblib"
DATA_PATH_X_train = "X_train.joblib.compressed"
DATA_PATH_y_train = "y_train.joblib"
DATA_PATH_X_val = "X_val.joblib.compressed"
DATA_PATH_y_val = "y_val.joblib"

PROJECT_ID = f"{model_name}_gcp_local_test"

#%%
# To pass these hyperparameters to the application (and to the pipeline),
# we have to define a list of arguments with the argparse library, like this
parser = argparse.ArgumentParser()
parser.add_argument(
    "--storage-path",
    type=str,
    default="gs://" + STORAGE_BUCKET,
    # required=True,
    help="GCP Storage path where to store training artifacts (string, required)",
)
parser.add_argument(
    "--n-estimators",
    type=int,
    default=10,
    help="The number of base estimators in the ensemble." "(integer, default 10)",
)
parser.add_argument(
    "--max-samples",
    type=float,
    default=1.0,
    help="The number of samples to draw from X"
    "to train each base estimator"
    "(float, default 1.0)",
)
parser.add_argument(
    "--max-features",
    type=float,
    default=1.0,
    help="The number of features to draw from X"
    "to train each base estimator"
    "(float, default 1.0)",
)
parser.add_argument(
    "--bootstrap",
    type=bool,
    default=True,
    help="Whether samples are drawn with replacement."
    "If False, sampling without replacement is performed."
    "(bool, default True)",
)
parser.add_argument(
    "--bootstrap-features",
    type=bool,
    default=False,
    help="Whether features are drawn with replacement." "(bool, default False)",
)
parser.add_argument(
    "--n-jobs",
    type=int,
    default=1,
    help="Number of parallel jobs to run (int, default 1)",
)

# Parse arguments
args, unknown = parser.parse_known_args()
print(f"{args}")

#%% Download dataset
# ^ X_fit & y_fit --------------------------------------------------------------
subprocess.run(
    [
        "gsutil",
        "cp",
        # # Storage path
        os.path.join("gs://", STORAGE_BUCKET, DATA_PATH_X_fit),
        # Local path
        os.path.join(data_processed_dir, "X_fit.joblib.compressed"),
    ],
    shell=True,
)
subprocess.run(
    [
        "gsutil",
        "cp",
        # # Storage path
        os.path.join("gs://", STORAGE_BUCKET, DATA_PATH_y_fit),
        # Local path
        os.path.join(data_processed_dir, "y_fit.joblib"),
    ],
    shell=True,
)
# ^ X_train & y_train ----------------------------------------------------------
subprocess.run(
    [
        "gsutil",
        "cp",
        # # Storage path
        os.path.join("gs://", STORAGE_BUCKET, DATA_PATH_X_train),
        # Local path
        os.path.join(data_processed_dir, "X_train.joblib.compressed"),
    ],
    shell=True,
)
subprocess.run(
    [
        "gsutil",
        "cp",
        # # Storage path
        os.path.join("gs://", STORAGE_BUCKET, DATA_PATH_y_train),
        # Local path
        os.path.join(data_processed_dir, "y_train.joblib"),
    ],
    shell=True,
)
# ^ X_val & y_val --------------------------------------------------------------
subprocess.run(
    [
        "gsutil",
        "cp",
        # # Storage path
        os.path.join("gs://", STORAGE_BUCKET, DATA_PATH_X_val),
        # Local path
        os.path.join(data_processed_dir, "X_val.joblib.compressed"),
    ],
    shell=True,
)
subprocess.run(
    [
        "gsutil",
        "cp",
        # # Storage path
        os.path.join("gs://", STORAGE_BUCKET, DATA_PATH_y_val),
        # Local path
        os.path.join(data_processed_dir, "y_val.joblib"),
    ],
    shell=True,
)

#%%
with open(data_processed_dir + "X_fit.joblib.compressed", "rb") as f:
    X_fit = joblib.load(f)
with open(data_processed_dir + "y_fit.joblib", "rb") as f:
    y_fit = joblib.load(f)

with open(data_processed_dir + "X_train.joblib.compressed", "rb") as f:
    X_train = joblib.load(f)
with open(data_processed_dir + "y_train.joblib", "rb") as f:
    y_train = joblib.load(f)

with open(data_processed_dir + "X_val.joblib.compressed", "rb") as f:
    X_val = joblib.load(f)
with open(data_processed_dir + "y_val.joblib", "rb") as f:
    y_val = joblib.load(f)

print(
    f"X_fit: {X_fit.shape}, {type(X_fit)}\
    \nX_train: {X_train.shape}, {type(X_train)}\
        \nX_val: {X_val.shape}, {type(X_val)}\n"
)

print(
    f"y_fit: {y_fit.shape}, {type(y_fit)}\
    \ny_train: {y_train.shape}, {type(y_train)}\
    \ny_val: {y_val.shape}, {type(y_val)}\n"
)

#%% Define model parameters for starting tuning
model_params = {
    "base_estimator": tree.ExtraTreeRegressor(
        criterion="mse",  # {"mse", "friedman_mse", ""mae"} default="mse"
        splitter="random",  # {"random", "best"} default="random"
        max_depth=None,  # default=None
        min_samples_split=2,  # default=2
        min_samples_leaf=1,  # default=1
        random_state=random_state,
    ),
    "n_estimators": args.n_estimators,
    "max_samples": args.max_samples,
    "max_features": args.max_features,
    "bootstrap": args.bootstrap,
    "bootstrap_features": args.bootstrap_features,
    "oob_score": False,
    "n_jobs": args.n_jobs,
    "random_state": random_state,
}
model = ensemble.BaggingRegressor(**model_params)

#%%
def accuracy(real_rates, predicted_rates):
    """Project's accuracy value estimator"""
    return np.average(abs(real_rates / predicted_rates - 1.0)) * 100.0


def calc_metrics(model, X, y):
    """Calculates result metrics"""

    from sklearn.metrics import max_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    y_true = y
    y_pred = model.predict(X)

    me = max_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mare = accuracy(y_true, y_pred)

    return me, mae, rmse, mare


#%% Train the model
# Train the model
tic = time.time()
model.fit(X_fit, y_fit)
# Evaluate time spent
min, sec = divmod(time.time() - tic, 60)
print(f"Time taken: {int(min)}min {int(sec)}sec")
print(f"{model}\n")

#%% Calculate a bunch of performance metrics
cur_time = time.strftime("%Y%m%d-%H%M")
me_train, mae_train, rmse_train, mare_train = calc_metrics(model, X=X_train, y=y_train)
me_val, mae_val, rmse_val, mare_val = calc_metrics(model, X=X_val, y=y_val)

results = pd.DataFrame(
    {
        "Max Error": [me_train, me_val],
        "Mean Absolute Error": [mae_train, mae_val],
        "Root Mean Squared Error": [mare_train, mare_val],
    },
    index=["Train", "Validation"],
)
print(f"{results}")
results.to_csv(models_dir + f"results_{cur_time}.csv")

#%%
# Upload model and results Dataframe to Storage
subprocess.run(
    [
        "gsutil",
        "cp",
        # Local path of the model
        os.path.join(models_dir, f"model_{cur_time}.joblib"),
        os.path.join(args.storage_path, f"model_{cur_time}joblib"),
    ]
)
subprocess.run(
    [
        "gsutil",
        "cp",
        # Local path of results
        os.path.join(models_dir, f"results_{cur_time}.csv"),
        os.path.join(args.storage_path, f"results_{cur_time}.csv"),
    ]
)

#%%
