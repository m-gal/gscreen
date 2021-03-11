"""
config.py
In projectname/projectname/config.py,
we place in special paths and variables that are used across the project.
Then, in our notebooks,
we can easily import these variables and not worry about custom strings littering our code.
    # notebook.ipynb
    |   from projectname.config import data_path
    |   import pandas as pd
    |   df = pd.read_csv(data_path)  # clean!
By using these config.py files,
we get clean code in exchange for an investment of time naming variables logically.

    Created on Dec 07 2020
    @author: mikhail.galkin
"""

#%% Import libraries
from pathlib import Path

#%% Define project's paths
# change 'projectname' onto real project folder's name
project_dir = Path("t:/DataProjects/_pets/gscreen")

data_raw_dir = project_dir / "data/raw"
data_processed_dir = project_dir / "data/processed"
data_cleaned_dir = project_dir / "data/cleaned"

docs_dir = project_dir / "docs"
models_dir = project_dir / "models"
notebooks_dir = project_dir / "notebooks"
pandas_profiling_dir = project_dir / "pandas_profiling"
pipelines_dir = project_dir / "pipelines"
reports_dir = project_dir / "reports"
sql_dir = project_dir / "sql"
tensorboard_dir = project_dir / "tensorboar"
tests = project_dir / "tests"
