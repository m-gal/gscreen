
<img src="https://static.tildacdn.com/lib/unsplash/1e457ee9-a8ee-c92b-33ea-3bb8f607e3b6/photo.jpg" width="1000">

# [Greenscreens.ai](https://greenscreens.ai/) challenge
### This is the end-to-end ML project based on real programming challenge from [GreenScreens.ai](https://greenscreens.ai/)

    Greenscreen - a platform for freight brokers in US.
    The core product of the platform is Rate Engine (RE).
    RE provides the price prediction for loads (order for cargo delivered from origin to destination by truck).
    The price depends on the origin/destination region and many market conditions.
    It is common for the industry to calculate the rate per mile as price divided by miles
    and use as a pricing indicator for the load (for example rate = 2,2 $/mi).
    All US territory was divided into KMA (Key Market Regions).
    These regions grouped by similar market conditions that are inside each market.

### __The Challenge__:
##### Try to develop model predicting the Rate Engine by pushing knowledge about origin and destination KMA into model.
----

### `Project's folders structure:`
_Some folders may be missing due project's particular aim._
```
PROJECTNAME (root folder)
├── data             <- Contains data sources for project.
│   ├── cleaned      <- Cleaned version of raw data.
│   ├── external     <- Data from third party sources.
│   ├── processed    <- The final data and dictionaries for modeling.
│   └── raw          <- The immutable input data.
├── docker           <- Dockerfiles for project.
├── docs             <- Keeps the project`s and models documentations.
├── @PROJECTNAME     <- Source code, scripts for use in this project.
│   ├── _no_longer_useful <- Obviously, but might be helpful to look on previous ideas.
│   ├── api          <- REST API client for model.
│   ├── assist       <- Any types of helping scripts.
│   ├── checks       <- Scripts to check some acts whether them works. Not unittests.
│   ├── clouds       <- Notebooks and modules to run in clouds.
│   ├── data         <- Scripts to get, to explore, to clean and to process data.
│   ├── mlproject    <- Project packaged in MLflow Project format to be reproducible later.
│   ├── model        <- Scripts to choose, fit (tune), train and evaluate models.
│   └── tests        <- Tests for project`s modules, functions, etc.
├── models           <- MLflow MLOps tools, mlregistry.db and trained models reside here.
│   ├── mlmodels     <- Separatly saved models in MLflow Models format to be reused later.
│   ├── mlruns       <- Logged MLflow runs. Keeps models its params, metrics and artifacts.
│   ├── params       <- Separatly saved best models` params found out with fit_model.py
│   ├── predictions  <- The model`s output.
│   └── README.md    <- Read me to maintain MLOps via MLflow.
├── notebooks        <- Jupyter notebooks related to the project.
├── pandas_profiling <- Configs for pandas_profiling EDA reports.
├── pipelines        <- Here are trained pipelines for data processing and models.
├── reports          <- Contains generated analysis as HTML, PDF, LaTeX, Excel and etc.
│   └── figures      <- Graphics and figures to be used in reporting.
├── sql              <- SQL scripts for fetching data.
├── temp             <- Folder to keep project`s temporary stuff.
└── tensorboard      <- Tracked experiments and its results by Tensorboard.
```

### `Workflow through the modules in @PROJECTNAME folder:`
_Some modules and folders may be missing due project's particular aim._
```
@PROJECTNAME
├── ....
├── data
│   ├──  [01] acquire_data.py       :: ACQUIRE DATA.
│   ├──  [02] explore_data.py       :: EXPLORE DATA.
│   ├──  [03] isolate_test_data.py  :: ISOLATE DATA for model TESTing.
│   ├──  [04] preclean_data.py      :: PREliminary DATA CLEANing.
│   ├──  [05] process_data.py       :: PROCESS DATA for modeling. Get features.
│   └──  [..] transformers.py       :: Custom transformers for processing pipeline.
├── model
│   ├──  [..] algos.py              :: Algorithms used for model training.
│   ├──  [06] choose_model.py       :: CHOOSE a best MODEL for further modeling.
│   ├──  [07] fit_model.py          :: FIT (tune) the chosen MODEL on train set.
│   ├──  [08] train_model.py        :: Finally TRAIN the fitted MODEL on development set.
│   └──  [09] value_model.py        :: eVALUATE the trained MODEL on unseen test data.
└── ....
```
