"""
    [04] FORTH step in project: *Might be sciped
        LOAD raw TRAIN (& VAL) sets from the [./project/data/raw]
        and CREATE Pipeline_Clean
        and FIT Pipeline on TRAIN dataset
        and CLEAN TRAIN data with Pipeline
        (and CLEAN VALIDATION data with Pipeline)
        and SAVE cleaned TRAIN (& VAL) datasets to the [./project/data/cleaned]
        and SAVE Pipeline as "pipeline_clean.*" to the [./project/pipelines].

        Later you will able view and analize clean data.

    Created on Dec 2020
    @author: mikhail.galkin
"""
#%% Load libraries
import sys
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import Pipeline

#%% Load project's stuff
sys.path.extend(["..", "../..", "../../.."])
import gscreen

# Load project utils and classes
from gscreen.config import data_raw_dir

# Set parameters
from gscreen.utils import set_pd_options
from gscreen.utils import set_matlotlib_params

set_pd_options()
set_matlotlib_params()

#%% Function: Parse pickup date
def parse_pickup_date(df):
    """ Derive Month, Week, Day, Weekday and Hour from pickup_date"""
    df_new = df.copy()
    df_new["pickup_month"] = df_new["pickup_date"].dt.month
    df_new["pickup_week"] = df_new["pickup_date"].dt.week
    df_new["pickup_day"] = df_new["pickup_date"].dt.day
    df_new["pickup_wday"] = df_new["pickup_date"].dt.dayofweek
    df_new["pickup_hour"] = df_new["pickup_date"].dt.hour

    df_new = df_new.drop(["pickup_date"], axis=1)
    # print("Parsing Datetime - done.")
    return df_new


#%% One-hot-encoding categorical
# def ohe_categorical(df):
#     """One-hot-encoding for categorical variables"""
#     global cols_to_ohe

#     df = pd.get_dummies(df, columns=cols_to_ohe)
#     # print("Categorical to OHE - done.")
#     return df

#%% Function: One-hot-encoding categorical
def ohe_categorical(df):
    """One-hot-encoding for categorical variables"""
    global cols_to_ohe
    global encoder

    encoder.fit(df[cols_to_ohe])
    transformed = encoder.transform(df[cols_to_ohe])
    transformed = pd.DataFrame(transformed, columns=encoder.classes_)
    df = pd.concat([df, transformed], axis=1).drop(cols_to_ohe, axis=1)

    # print("Categorical to OHE - done.")
    return df


#%% Function: Labeling categorical
def label_transport(df):
    """Label encoding for transport_type variable"""

    df["transport_type"] = df_train["transport_type"].replace(
        {"FLATBED": 1, "REEFER": 2, "VAN": 3}
    )

    # print("Categorical to OHE - done.")
    return df


#%% Function: Normalize data
# Its critical point that the KNN Imptuer is a distance-based imputation method
# and it requires us to normalize our data
def scale_numeric(df):
    """Normalize numeric variables"""
    global cols_to_scale
    global scaler

    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    # print("Scaling numerical - done.")
    return df


#%% Function: Inpute missing values using KNN algo
def impute_missing(df):
    """Impute numeric missing values using kNN algorithm"""
    global cols_to_impute
    global imputer

    df_impute = df[cols_to_impute]
    df_impute = pd.DataFrame(imputer.fit_transform(df_impute), columns=df_impute.columns)

    df = df[df.columns.difference(cols_to_impute)].join(df_impute)
    # print("Imputing missing - done.")
    return df


#%% Function: Features hashing
def hash_categorical(df):
    """Hashing for kma variables"""
    global cols_to_hash
    global hasher

    df_hash = df[cols_to_hash].copy()
    hashed = hasher.fit_transform(df_hash.values)
    hashed = pd.DataFrame(
        hashed.toarray(), columns=[f"hash_{str(x).zfill(2)}" for x in range(hashed.shape[1])]
    )

    df = df.drop(cols_to_hash, axis=1).join(hashed)
    # print("Hashing categorical - done.")
    return df


#%% Load data ----------------------------------------------------------------------------------
def load_data():
    df_train_origin = gscreen.utils.load_data(
        dir=data_raw_dir,
        file_to_load="train_20201221.csv",
        drop_duplicated=True,
    )
    df_test_origin = gscreen.utils.load_data(
        dir=data_raw_dir,
        file_to_load="validation_20201221.csv",
        drop_duplicated=True,
    )

    # Copy data aims does not hurt original datasets
    df_train = df_train_origin.copy()
    df_test = df_test_origin.copy()

    return df_train, df_test


#%%
if __name__ == "__main__":
    cols_to_ohe = ["transport_type"]
    cols_to_scale = [
        "valid_miles",
        "weight",
        # "pickup_month",
        # "pickup_week",
        # "pickup_day",
        # "pickup_wday",
        # "pickup_hour",
    ]
    cols_to_impute = [
        "valid_miles",
        "weight",
        "pickup_month",
        "pickup_week",
        "pickup_day",
        "pickup_wday",
        "pickup_hour",
    ]
    cols_to_hash = ["origin_kma", "destination_kma"]
    encoder = LabelBinarizer()
    scaler = RobustScaler()
    imputer = KNNImputer(n_neighbors=5, copy=False)
    hasher = FeatureHasher(n_features=70, input_type="string")

    df_train, _ = load_data()
    train_X = parse_pickup_date(df_train)
    train_X = ohe_categorical(train_X)
    train_X = scale_numeric(train_X)
    train_X = impute_missing(train_X)
    train_X = hash_categorical(train_X)
    train_X

#%%
