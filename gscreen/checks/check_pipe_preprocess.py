"""
    [10] TENTH step in project:
    Stub to do one-off prediction using the trained model

    Created on Dec 2020
    @author: mikhail.galkin
"""

#%% Load libraries
import sys
import pandas as pd
import joblib
import winsound

#%% Load project's stuff
sys.path.extend(["..", "../..", "../../.."])
import gscreen
from gscreen.config import pipelines_dir
from gscreen.config import data_processed_dir
from gscreen.utils import make_example_df


#%% Load early trained and saved pipeline
def load_pipeline(pipeline_to_load, feature_cols_to_load):
    # The pipelina
    with open(pipelines_dir / pipeline_to_load, "rb") as f:
        pipeline = joblib.load(f)
    # Load feature names
    feature_cols = joblib.load(data_processed_dir / feature_cols_to_load)
    return pipeline, feature_cols


#%% Define data frame containing one hand-crafted input ------------------------
# example_df, _ = make_example_df(num_rows=2)
# print(example_df)

#%% Push through pipe
def get_processed_df(df, pipeline, feature_cols):
    # Convert to df
    result = pd.DataFrame.sparse.from_spmatrix(
        pipeline.transform(df),
        columns=feature_cols,
    ).T
    print("\nProcessed example DF:")
    print(result)


#%% Main function for main.py ==================================================
def main(
    pipeline_to_load="data_process_fitted_on_train.joblib",
    feature_cols_to_load="feature_cols.joblib",
    num_rows_example=2,
):
    """Checks how correctly the data processing pipeline works.

    Args:
        * pipeline_to_load (str, optional): File with preprocessing pipeline.\
            Defaults to "data_process_fitted_on_train.joblib".
        * feature_cols_to_load (str, optional): Saved features' columns ' names.\
            Defaults to "feature_cols.joblib".
        * num_rows_example (int, optional): Num of rows in example data frame generated.\
            Defaults to 2.
    """
    print(f"-------------- START: Visual check processing pipeline --------------")
    pipeline, feature_cols = load_pipeline(pipeline_to_load, feature_cols_to_load)
    example_df, _ = make_example_df(num_rows_example)
    print("\nExample DF:")
    print(example_df)
    get_processed_df(example_df, pipeline, feature_cols)
    print(f"!!! DONE: Check preprocessing pipeline !!!")
    winsound.Beep(frequency=2000, duration=300)


#%% Workflow ===================================================================
if __name__ == "__main__":
    pd.set_option("display.max_rows", 300)
    main()
    pd.reset_option("display.max_rows")

#%%
