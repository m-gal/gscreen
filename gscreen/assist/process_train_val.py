"""
    This module takes raw TRAIN and VALIDATION data sets
    and processes data to the features for training

    Created on Dec 2020
    @author: mikhail.galkin
"""

#%% Load libraries
import sys
import numpy as np
import joblib
import winsound  # to beep when done

#%% Load project's stuff
sys.path.extend(["..", "../..", "../../.."])
import gscreen

# Load custom classes and utils
from gscreen.config import data_raw_dir
from gscreen.config import pipelines_dir
from gscreen.config import data_processed_dir

# Set parameters
from gscreen.utils import set_pd_options
from gscreen.utils import set_matlotlib_params

set_pd_options()
set_matlotlib_params()

#%% For reloading project's utils, classes w\o total reloading workspace
# ! Not required in final module version
# import importlib
# import inspect  # for sub-modules inspection

# importlib.reload(gscreen.utils)
# print(inspect.getsource(gscreen.utils.calc_metrics))

#%% Load data ------------------------------------------------------------------
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

#%% Load early saved process pipeline
pipeline_to_load = "pipeline_process.joblib"
with open(pipelines_dir / pipeline_to_load, "rb") as f:
    pipeline_process = joblib.load(f)

#%% Load outliers mask
outliers_mask_train = joblib.load(data_processed_dir / f"outliers_mask_train.joblib")
print(f"{sum(outliers_mask_train==0)} outliers")

#%% Preparation train, validation, test
# Retrive X and y
target = "rate"
X_train, y_train = df_train.loc[:, df_train.columns != target], df_train[target]
X_val, y_val = df_val.loc[:, df_val.columns != target], df_val[target]
# Get info
print(
    f"""Shape of:
    \t X_train={X_train.shape}: {len(X_train) / (len(X_train) + len(X_val))}%
    \t X_val={X_val.shape}: {len(X_val) / (len(X_train) + len(X_val))}%
    """
)

#%% Process data
# It is nesessary only in RandomizedSearchCV case separatly process data before
# fitting, due the validation set is used inside grig searching.
# In case of simple GridSearchCV we could constuct full pipeline [process, model]
X_train = pipeline_process.transform(X_train)
X_val = pipeline_process.transform(X_val)
print(f"X_train has {X_train.shape[1]} features and {X_train.shape[0]} records.")
print(f"X_val has {X_val.shape[1]} features and {X_val.shape[0]} records.")

# Make set w\o outliers
X_fit, y_fit = X_train[outliers_mask_train, :], y_train[outliers_mask_train]
print(f"X_fit has {X_fit.shape[1]} features and {X_fit.shape[0]} records.")

#%% Save processed data sets
with open(data_processed_dir / f"X_train.joblib.compressed", "wb") as f:
    joblib.dump(X_train, f, compress=3)
print(f"X_train saved")

with open(data_processed_dir / f"X_fit.joblib.compressed", "wb") as f:
    joblib.dump(X_fit, f, compress=3)
print(f"X_fit saved")

with open(data_processed_dir / f"X_val.joblib.compressed", "wb") as f:
    joblib.dump(X_val, f, compress=3)
print(f"X_val saved")

joblib.dump(np.array(y_train), data_processed_dir / f"y_train.joblib")
joblib.dump(np.array(y_fit), data_processed_dir / f"y_fit.joblib")
joblib.dump(np.array(y_val), data_processed_dir / f"y_val.joblib")
print(f"targets saved.")

#%% Check saving accuracy
with open(data_processed_dir / f"X_train.joblib.compressed", "rb") as f:
    X_train_loaded = joblib.load(f)
print(f"Have X_train save accurate? : {(X_train_loaded != X_train).nnz==0}")

with open(data_processed_dir / f"X_fit.joblib.compressed", "rb") as f:
    X_fit_loaded = joblib.load(f)
print(f"Have X_fit save accurate? : {(X_fit_loaded != X_fit).nnz==0}")

with open(data_processed_dir / f"X_val.joblib.compressed", "rb") as f:
    X_val_loaded = joblib.load(f)
print(f"Have X_val save accurate? : {(X_val_loaded != X_val).nnz==0}")

y_train_loaded = joblib.load(data_processed_dir / f"y_train.joblib")
print(f"Have y_train save accurate? : {(y_train_loaded == np.array(y_train)).all()}")

y_fit_loaded = joblib.load(data_processed_dir / f"y_fit.joblib")
print(f"Have y_fit save accurate? : {(y_fit_loaded == np.array(y_fit)).all()}")

y_val_loaded = joblib.load(data_processed_dir / f"y_val.joblib")
print(f"Have y_val save accurate? : {(y_val_loaded == np.array(y_val)).all()}")

#%% Finish
print(f"!!! DONE !!!")
winsound.Beep(frequency=3500, duration=1500)

#%%
