"""
    [10] TENTH step in project:
    Stub to do one-off prediction using the trained model

    Created on Dec 2020
    @author: mikhail.galkin
"""
#%% Load libraries
import sys
import pandas as pd
import winsound

#%% Load project's stuff
sys.path.extend(["..", "../..", "../../.."])
# import gscreen
from gscreen.config import models_dir
from gscreen.utils import get_model_paths
from gscreen.utils import load_model
from gscreen.utils import make_example_df
from gscreen.utils import accuracy

#%% Define some functions ------------------------------------------------------
def predict_example_df(
    model,
    num_rows_1=True,
    num_rows_2=True,
):
    def make_prediction(num_rows):
        example_df, real_rate = make_example_df(num_rows)
        print(example_df)
        #%% Make prediction
        prediction = model.predict(example_df)
        mare = accuracy(real_rate, prediction)
        # Print out
        print(f"\nReal Rate is {real_rate}")
        print(f"Predicted Rate: {list(prediction)}")
        print(f"Mean Absolute Ratio Error: {mare}\n")

    if num_rows_1:
        make_prediction(num_rows=1)
    if num_rows_2:
        make_prediction(num_rows=2)


def check_prediction(
    model_paths,
    num_rows_1=True,
    num_rows_2=True,
):
    for model_path in model_paths:
        if model_path is not None:
            model = load_model(model_path)
            predict_example_df(model, num_rows_1, num_rows_2)
            del(model)
        else:
            print(f"No saved model to check one-off prediction...")


#%% Main function for main.py ==================================================
def main(
    exp_id=None,
    run_id=None,
    folder=None,
    num_rows_1=True,
    num_rows_2=True,
):
    """Checks a one-off prediction using the trained model.

    Args:
        * exp_id (int, optional): Model's MLflow experiment ID. Defaults to None.
        * run_id (str, optional): Model's MLflow run ID. Defaults to None.
        * folder (str, optional): Folder's  name for separatly saved model.\
            Defaults to None.
        * num_rows_1 (bool, optional): Toogle to check prediction on dataframe
            with only 1 rows. Defaults to True.
        * num_rows_2 (bool, optional): Toogle to check prediction on dataframe
            with 2 rows. Defaults to True.
    """
    print(f"-------------- START: Check an one-off prediction --------------")
    model_paths, _ = get_model_paths(exp_id, run_id, models_dir, folder)
    check_prediction(model_paths, num_rows_1, num_rows_2)
    print(f"!!! DONE: Check an one-off prediction !!!")
    winsound.Beep(frequency=2000, duration=300)


#%% Workflow ===================================================================
if __name__ == "__main__":
    main(
        exp_id=1,
        run_id="7a143937d76142949a8fa8d2f06547ea",
        folder="20210310-2148_7a143937d76142949a8fa8d2f06547ea",
    )

#%%
