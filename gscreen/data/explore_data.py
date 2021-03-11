"""
    [02] SECOND step in project:
        LOAD raw data from the [./project/data/raw]
        and GIVE us a first glimpse
        and MAKE Exploratory Data Analysis w\o any changing original data
        and SAVE results of EDA to the [./project/reports].

    Created on Dec 08 2020
    @author: mikhail.galkin
"""
#%% Load libraries
import sys
import pandas as pd
import winsound
import pandas_profiling as pp
import sweetviz

#%% Load project's stuff
sys.path.extend(["..", "../..", "../../.."])
import gscreen

# Load project utils and classes
from gscreen.config import data_raw_dir
from gscreen.config import reports_dir
from gscreen.config import pandas_profiling_dir

# Set parameters
from gscreen.utils import set_pd_options
from gscreen.utils import set_matlotlib_params

#%% Load datasets --------------------------------------------------------------
def load_data():
    df_train = gscreen.utils.load_data(dir=data_raw_dir, file_to_load="train_20201221.csv")
    df_val = gscreen.utils.load_data(dir=data_raw_dir, file_to_load="validation_20201221.csv")

    #%% Get first info about train set
    print(f"Count of duplicated rows = {df_train.duplicated().sum()}")
    print(f"{df_train.dtypes.value_counts()} \n")
    print(f"{df_train.info(verbose=True)} \n")

    # View some random selected records
    n = 6
    print(f"Random {n} rows:")
    df_train.sample(n=n).T

    return df_train, df_val


#%% View data trough Datetime column -------------------------------------------
def view_some_data(df):
    df.groupby(
        [
            df["pickup_date"].dt.to_period("M"),
            df["origin_kma"],
            df["destination_kma"],
        ]
    ).agg(["size", "min", "mean", "max"])


# ------------------------------------------------------------------------------
# Profiling Exploratary Data Analysis
# ------------------------------------------------------------------------------
#%% Profiling data and save report ot HTML
def make_pandas_profiling_report(df):
    print(f"\nPandas profiling report start...")
    # You can choose config between: "config_default." \"config_minimal." \"config_optimal."
    config_file = pandas_profiling_dir / "config_optimal.yaml"

    # Make: Pandas Profile report
    pp_train = pp.ProfileReport(df, config_file=config_file)
    pp_train.to_file(reports_dir / f"PandasProfile_train.html")


# ------------------------------------------------------------------------------
# Sweetviz Exploratary Data Analysis
# ------------------------------------------------------------------------------
#%% Change config for SweetViz report
def make_sweetviz_report(df_train, df_val):
    print(f"\nSweetViz report start...")
    # # Code to view SweetViz configuration from config file
    # sweetviz.config_parser.read("Override.ini")
    # for sect in sweetviz.config_parser.sections():
    #     print("Section:", sect)
    #     for k, v in sweetviz.config_parser.items(sect):
    #         print(" {} = {}".format(k, v))
    #     print()

    sweetviz.config_parser.set(section="Layout", option="show_logo", value="0")
    feature_config = sweetviz.FeatureConfig(skip=None, force_text=None)

    #%% Create and save SweetViz report for train set
    sv_train = sweetviz.analyze(df_train, target_feat="rate", feat_cfg=feature_config)
    sv_train.show_html(reports_dir / f"SweeetViz_train.html")

    #%% Comparing two datasets (Validation vs Training sets)
    sv_compare = sweetviz.compare(
        [df_train, "TRAIN"], [df_val, "VALID"], target_feat="rate", feat_cfg=feature_config
    )
    sv_compare.show_html(reports_dir / f"SweeetViz_compare.html")


#%% Main function for main.py ==================================================
def main(
    perfom_eda=True,
    pandas_profiling=True,
    sweetviz=True,
):
    """Performs a Exploratary Data Analisys with Pnadas Profiling and SweetViz
    packages.

    Args:
        * perfom_eda (bool, optional): Toggle to perform a EDA.\
            Defaults to True.
        * pandas_profiling (bool, optional): Toggle to create Pandas_profiling report.\
            Defaults to True
        * sweetviz (bool, optional): Toggle to create Sweetviz report.\
            Defaults to True
    """
    if perfom_eda:
        print(f"-------------- START: Exploratary Data Analisys --------------")
        df_train, df_val = load_data()
        gscreen.utils.get_na_cols(df_train)
        view_some_data(df_train)
        if pandas_profiling:
            make_pandas_profiling_report(df_train)
        if sweetviz:
            make_sweetviz_report(df_train, df_val)
        print(f"!!! DONE: Exploratory data analysis !!!")
        winsound.Beep(frequency=3000, duration=300)
    else:
        print(f"-------------- SKIP: Exploratary Data Analisys --------------")


#%% Workflow ===================================================================
if __name__ == "__main__":
    set_pd_options()
    set_matlotlib_params()
    main()
    gscreen.utils.reset_pd_options()

#%%
