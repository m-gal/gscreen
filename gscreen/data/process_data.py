"""
    [05] FIFTH step in project:
        * process=turn clean data into features for modeling
        LOAD raw TRAIN (& VAL) sets from the [./project/data/raw]
            or LOAD cleaned TRAIN (& VAL) from the [./project/data/cleaned]
        and CREATE Pipeline_Process:
            [
                Pipeline_Clean
                + Pipeline_Transform
            ]
        and FIT Pipeline on TRAIN dataset
        and PROCESS TRAIN data with Pipeline
            (& PROCESS VALIDATION data with Pipeline)
        and SAVE processed TRAIN (& VAL) datasets to the [./project/data/processed]
        and SAVE Pipeline to the [./project/pipelines].

    Created on Dec 2020
    @author: mikhail.galkin
"""

#%% Load libraries
import sys
import pandas as pd
import joblib
import winsound  # to beep when done
from sklearn import preprocessing
from sklearn import impute
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest

# import category_encoders as ce
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.impute import SimpleImputer

#%% Load project's stuff
sys.path.extend(["..", "../..", "../../.."])
import gscreen

# Load custom classes and utils
from gscreen.config import data_raw_dir
from gscreen.config import data_processed_dir
from gscreen.config import pipelines_dir
from gscreen.config import models_dir

from gscreen.data.transformers import TransportTypeTransformer
from gscreen.data.transformers import PickupDateTransformer
from gscreen.data.transformers import InsideStateTransformer

# Set parameters
from gscreen.utils import set_pd_options
from gscreen.utils import set_matlotlib_params

#%%! Toggles to flow through ---------------------------------------------------
rnd_state = 42
# save_pipeline = True
# save_data_processed = True
# get_outliers_mask = True
# save_pipeline_as = "data_process_fitted_on_train.joblib"
# data_save_as = "df_train_processed"

#%% Print out ------------------------------------------------------------------
def print_toggles(
    save_pipeline,
    save_data_processed,
    get_outliers_mask,
):
    print(f"Save data processing pipeline: {save_pipeline}")
    print(f"Save train data processed: {save_data_processed}")
    print(f"Get outliers mask: {get_outliers_mask}")


# Print out versions
def print_versions():
    try:
        skl = Pipeline.__module__[: Pipeline.__module__.index(".")]
    except:
        skl = Pipeline.__module__
    print(f" ")
    print(f"Pandas: {pd.__version__}")
    print(f"Scikit-learn: {sys.modules[skl].__version__}")


#%% Create data processing pipeline --------------------------------------------
def create_pipeline():
    print(f"\nCreate data processing pipeline...")
    #%% Define columns to transform
    cols_num = [
        "valid_miles",
        "weight",
    ]
    cols_transport = [
        "transport_type",
    ]
    cols_date = [
        "pickup_date",
    ]
    cols_kma = [
        "origin_kma",
        "destination_kma",
    ]
    #%% Define individual transformers for a future pipeline
    numerical_process = Pipeline(
        steps=[
            ("impute", impute.SimpleImputer(strategy="median")),
            ("scale", preprocessing.StandardScaler()),
            # ("scale", preprocessing.RobustScaler()), # worse
        ]
    )
    transport_process = Pipeline(
        steps=[
            # ("transport", TransportTypeTransformer()), # slowly
            ("transport", preprocessing.OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pikupdate_process = Pipeline(
        steps=[
            ("pickupdate", PickupDateTransformer()),
        ]
    )
    kma_process = Pipeline(
        steps=[
            ("kma", preprocessing.OneHotEncoder(handle_unknown="ignore")),
            # ("kma", preprocessing.OrdinalEncoder()), # worse
        ]
    )
    instate = Pipeline(
        steps=[
            ("instate", InsideStateTransformer()),
        ]
    )
    #%% Construct Column Transformer pipeline for data processing
    process = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_process,
                cols_num,
            ),
            (
                "transport",
                transport_process,
                cols_transport,
            ),
            (
                "pikupdate",
                pikupdate_process,
                cols_date,
            ),
            (
                "kma",
                kma_process,
                cols_kma,
            ),
        ],
        # By specifying remainder='passthrough',
        # all remaining columns that were not specified in transformers
        # will be automatically passed through
        remainder="passthrough",
    )
    # Define processing pipeline
    pipeline_process = Pipeline(
        steps=[
            ("instate", instate),
            ("process", process),
        ]
    )

    return pipeline_process, (cols_num, cols_transport, cols_date, cols_kma)


#%% Load data ------------------------------------------------------------------
def get_data_as_df(file="train_20201221.csv"):
    print(f"\nLoad raw data from {file}...")
    df = gscreen.utils.load_data(
        dir=data_raw_dir,
        file_to_load=file,
        drop_duplicated=False,
    )
    #%% Retrive X and y
    target = "rate"
    X, y = df.loc[:, df.columns != target], df[target]

    return X, y, df


#%% Pick up features names -----------------------------------------------------
def get_features_names(pipeline, cols):
    print(f"\nGet features names...")
    feature_cols = []
    cols_num = cols[0]
    cols_transport = cols[1]
    # cols_date = cols[2]
    cols_kma = cols[3]

    feature_cols.extend(cols_num)
    feature_cols.extend(
        list(
            pipeline["process"].transformers_[1][1]["transport"].get_feature_names(cols_transport),
        )
    )
    feature_cols.extend(
        [
            "pickup_week",  # 2 5 : 3.5
            "pickup_hour",  # 8 7 : 7.5
            # ^ Feature converted to cyclic type
            "pickup_dayofyear_sin",  # 7 1 : 4
        ]
    )
    feature_cols.extend(
        list(pipeline["process"].transformers_[3][1]["kma"].get_feature_names(cols_kma))
    )
    feature_cols.extend(["instate"])

    return feature_cols


#%% Convert processed data as DF --------------------------------------------------
def convert_processed_to_df(data_processed, feature_cols):
    print(f"\nConvert processed data into Pandas DF...")
    # Convert to df
    df_processed = pd.DataFrame.sparse.from_spmatrix(
        data_processed,
        columns=feature_cols,
    )
    return df_processed


#%% Get the outliers in train set ----------------------------------------------
def get_outliers_treshold(df, treshold_rate=50):
    print(f"\nGet outliers with rate's treshold = {treshold_rate}...")
    treshold_rate = 50
    outliers_mask_train_thold = df["rate"] < treshold_rate
    print(f"It was found out {sum(outliers_mask_train_thold==0)} outliers.")
    joblib.dump(
        outliers_mask_train_thold,
        data_processed_dir / f"outliers_mask_train_thold.joblib",
    )
    print(f"Outliers by treshold saved into {data_processed_dir}.")

    return outliers_mask_train_thold


#%% Analyse outliers in train set ----------------------------------------------
def get_outliers_isolation_forest(
    df,
    n_estimators=100,
    contamination="auto",
    n_jobs=-1,
):
    print(f"\nGet outliers with Isolation Forest...")
    # Identify outliers in the training dataset
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        bootstrap=True,
        n_jobs=n_jobs,
        verbose=1,
        random_state=rnd_state,
    )
    # yhat = iso.fit_predict(df_X_train_processed)
    yhat = iso.fit_predict(df)
    print(f"It was found out {sum(yhat==-1)} outliers.")

    # Get mask for all TRAIN rows that are not outliers
    outliers_mask_train_iforest = yhat != -1
    joblib.dump(
        outliers_mask_train_iforest,
        data_processed_dir / f"outliers_mask_train_iforest.joblib",
    )
    print(f"Outliers by Isolation Forest saved {data_processed_dir}.")

    return outliers_mask_train_iforest


#%% Save processed data --------------------------------------------------------
def save_processed_data(save_data_processed, X_processed, y, save_data_processed_as):
    if save_data_processed:
        df_processed = X_processed.join(y)
        print(f"\nSaving processed train data...")
        # # Save processed data
        print(f"Dumping to .joblib...")
        with open(data_processed_dir / f"{save_data_processed_as}.joblib.compressed", "wb") as f:
            joblib.dump(df_processed, f, compress=3)
        # to CSV
        print(f"Saving as .CSV...")
        df_processed.to_csv(data_processed_dir / f"{save_data_processed_as}.csv")
        print(f"Processed train set saved into {data_processed_dir}.")
    else:
        print(f"Processed train set was NOT saved.")


#%% Save results ---------------------------------------------------------------
def save_processing_pipeline(save_pipeline, save_pipeline_as, pipeline, feature_cols):
    if save_pipeline:
        print(f"\nSave data processing pipeline...")
        #  Save processing pipeline
        with open(pipelines_dir / save_pipeline_as, "wb") as f:
            joblib.dump(pipeline, f)
        # Save feature names
        joblib.dump(feature_cols, data_processed_dir / f"feature_cols.joblib")
        print(f"Data processing pipeline saved into {pipelines_dir}.")
    else:
        print(f"\nData processing pipeline was NOT saved.")


#%% For some reasons you will migth want to check saving accuracy --------------
def check_saved_pipeline(check_pipeline, save_pipeline_as, X, X_processed):
    if check_pipeline:
        print(f"\nCheck how accurate process pipeline was saved...")
        # Check how acurate process pipeline was saved
        ## load saved pipeline
        with open(pipelines_dir / save_pipeline_as, "rb") as f:
            pipeline_to_check = joblib.load(f)
        ## check it
        compare = (pipeline_to_check.transform(X) - X_processed).nnz
        # Print out
        print(f"Have process pipeline save accurate? : {compare==0}\n")


#%% Main function for main.py ==================================================
def main(
    file="train_20201221.csv",
    save_pipeline=True,
    save_pipeline_as="data_process_fitted_on_train.joblib",
    save_data_processed=True,
    save_data_processed_as="df_train_processed",
    check_pipeline=True,
    get_outliers_mask=True,
    treshold_rate=50,
    iforest_n_estimators=100,
    iforest_contamination=0.0001,
    iforest_n_jobs=-1,
):
    """Creates data processing pipeline.

    Args:
        * file (str, optional): Train dataset. Defaults to "train_20201221.csv".
        * save_pipeline (bool, optional): Toggle to save pipeline. Defaults to True.
        * save_pipeline_as (str, optional): File name for pipeline saved. \
            Defaults to "data_process_fitted_on_train.joblib".
        * save_data_processed (bool, optional): Toggle to save data processed.\
            Defaults to True.
        * save_data_processed_as (str, optional): File name for data processed saved.\
            Defaults to "df_train_processed".
        * check_pipeline (bool, optional): Toggle to check how accurate process\
            pipeline was saved. Defaults to True.
        * get_outliers_mask (bool, optional): Toggle to get outliers mask for train set.\
            Defaults to True.
        * treshold_rate (int, optional): Rate's size to cut outliers manualy.\
            Defaults to 50.
        * iforest_n_estimators (int, optional): Num of trees in Isolation Forest\
            to get utliers. Defaults to 100.
        * iforest_contamination (float, optional): Cutting off level of outliers'\
            infection. Defaults to 0.0001.
        * iforest_n_jobs (int, optional): The number of jobs iat IsolationForest\
            to run in parallel. -1 means using all processors. Defaults to -1.
    """
    print(f"-------------- START: Data preprocessing pipeline --------------")
    print_versions()
    print_toggles(save_pipeline, save_data_processed, get_outliers_mask)
    pipeline, cols = create_pipeline()
    X_train, y_train, df_train = get_data_as_df(file)
    X_train_processed = pipeline.fit_transform(X_train)
    feature_cols = get_features_names(pipeline, cols)
    df_X_train_processed = convert_processed_to_df(X_train_processed, feature_cols)
    if get_outliers_mask:
        get_outliers_treshold(df_train, treshold_rate)
        get_outliers_isolation_forest(
            df_X_train_processed,
            n_estimators=iforest_n_estimators,
            contamination=iforest_contamination,
            n_jobs=iforest_n_jobs,
        )
    save_processed_data(
        save_data_processed,
        df_X_train_processed,
        y_train,
        save_data_processed_as,
    )
    save_processing_pipeline(save_pipeline, save_pipeline_as, pipeline, feature_cols)
    check_saved_pipeline(check_pipeline, save_pipeline_as, X_train, X_train_processed)
    print(f"!!! DONE: Data preprocessing pipeline !!!")
    winsound.Beep(frequency=3000, duration=300)


#%% Workflow ===================================================================
if __name__ == "__main__":
    set_pd_options()
    set_matlotlib_params()
    main()
    gscreen.utils.reset_pd_options()

#%%
