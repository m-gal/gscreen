"""
    [09] NINTH step in project:
        LOAD Trained Model from the [./project/models]
        and LOAD raw isolated TEST set from the [./project/data/raw]
        and LOAD piped_model from the [./project/pipelines]
        and PASS TEST data througt Pipeline
        and EVALUATE Trained Model on unseen TEST data
        and LOG results with MLflow Tracking
        and SAVE logs in the [.project/mlflow_runs]
            or LOG results with Tensorboard
            and SAVE logs in the [.project/tensorboard]
        and SAVE results to the [./project/reports]
        and SAVE predictions to the [./project/models]

    Created on Dec 2020
    @author: mikhail.galkin
"""

#%% Load libraries
import sys
import pandas as pd
import matplotlib.pyplot as plt
import winsound
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow

#%% Load project's stuff
sys.path.extend(["..", "../..", "../../.."])
import gscreen

# Load custom classes and utils
from gscreen.config import data_raw_dir
from gscreen.config import models_dir
from gscreen.config import reports_dir
from gscreen.utils import get_model_paths
from gscreen.utils import load_model
from gscreen.utils import accuracy as mare

# Set parameters
from gscreen.utils import set_pd_options
from gscreen.utils import set_matlotlib_params

#%% Print out ------------------------------------------------------------------
def print_toggles(
    save_predictions,
    save_distplot,
    save_boxplot,
):
    print(f" ")
    print(f"Save predictions: {save_predictions}")
    print(f"Save distributions plot: {save_distplot}")
    print(f"Save boxplot: {save_boxplot}")


#%% Load saved model -----------------------------------------------------------
def load_saved_model(model_paths, model_names):
    # If model reside in MLflow runs
    if model_paths[0] is not None:
        print(f"\nMLflow model will being evaluated...")
        model = load_model(model_paths[0])
        model_name = model_names[0]
        return model, model_name
    # If model was saved separatly
    elif model_paths[1] is not None:
        print(f"\nSeparatly saved model will being evaluated...")
        model = load_model(model_paths[1])
        model_name = model_names[1]
        return model, model_name
    else:
        print(f"\nNo saved model to evaluate...")
        return None, None


#%% Load data ------------------------------------------------------------------
def load_raw_data():
    target = "rate"
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
    df_test = gscreen.utils.load_data(
        dir=data_raw_dir,
        file_to_load="test_20201221.csv",
        drop_duplicated=False,
    )
    # Retrive X and y
    X_train, y_train = df_train.loc[:, df_train.columns != target], df_train[target]
    X_val, y_val = df_val.loc[:, df_val.columns != target], df_val[target]
    X_test = df_test.copy()

    return X_train, y_train, X_val, y_val, X_test


#%% Make prediction Train Val: View results ------------------------------------
def print_train_val_metrics(model, X_train, y_train, X_val, y_val):
    print("TRAIN set:")
    gscreen.utils.calc_metrics(model, X=X_train, y=y_train)
    print("VALIDATION set:")
    gscreen.utils.calc_metrics(model, X=X_val, y=y_val)


#%% Make prediction ------------------------------------------------------------
def predict_train_val_test(model, X_train, X_val, X_test, model_name, save_predictions):
    print(f"\nMake prediction with trained model...")
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    if save_predictions:
        print(f"Save prediction from trained model...")
        dir = models_dir / "predictions"
        pd.DataFrame(pred_train).to_csv(dir / f"predict_eval_train_{model_name}.csv")
        pd.DataFrame(pred_val).to_csv(dir / f"predict_eval_val_{model_name}.csv")
        pd.DataFrame(pred_test).to_csv(dir / f"predict_eval_test_{model_name}.csv")
        print(f"Predictions saved. You can find them in: {dir}")
    return pred_train, pred_val, pred_test


#%% View distribution for predictions ------------------------------------------
def plot_density(
    pred_train,
    pred_val,
    pred_test,
    model_name,
    save_distplot=True,
):
    kwargs = dict(hist=False, kde=True)

    fig = plt.figure(figsize=(15, 5))
    plt.title(f"Density of predicted rate for {model_name}")
    plt.xlabel("Predicted rate")

    sns.distplot(
        pred_train,
        color="blue",
        label="Train",
        kde_kws={"linewidth": 2},
        **kwargs,
    )
    sns.distplot(pred_val, color="lightgreen", label="Val", kde_kws={"linewidth": 2}, **kwargs)
    sns.distplot(
        pred_test,
        color="red",
        label="Test",
        kde_kws={"linewidth": 3},
        **kwargs,
    )

    plt.xlim(0, 20)
    plt.legend()
    plt.show()

    if save_distplot:
        file_name = f"eval_distplot_{model_name}.png"
        fig.savefig(
            reports_dir / "figures" / file_name,
            dpi=100,
        )
        print(f"\nDensity of predicted rate was saved.")

    return fig


#%% Boxblots for predictions ---------------------------------------------------
def plot_boxplot(
    pred_train,
    pred_val,
    pred_test,
    model_name,
    save_boxplot=True,
):
    # Get data and labels
    preds = [pred_train, pred_val, pred_test]
    labels = ["Train", "Val", "Test"]

    # Create 2 stacked sub-plots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 11))

    # box plot w\o outliers
    bplot1 = ax1.boxplot(
        preds,
        vert=False,  # vertical box alignment
        patch_artist=True,  # fill with color
        showfliers=True,
        showmeans=True,
        labels=labels,
    )
    ax1.set_title("Prediction's distributions with outliers & limited by rate=20")
    ax1.set_xlim(0, 20)

    # box plot with limited range
    bplot2 = ax2.boxplot(
        preds,
        vert=False,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        labels=labels,
    )
    ax2.set_title("Prediction's distributions w\o outliers")
    ax2.set_xlim(0, None)

    # box plot with outliers
    bplot3 = ax3.boxplot(
        preds,
        vert=False,
        patch_artist=True,
        showfliers=True,
        showmeans=True,
        labels=labels,
    )
    ax3.set_title("Prediction's distributions with outliers")
    ax3.set_xlim(0, None)

    # fill with colors
    colors = ["blue", "lightgreen", "red"]
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)

    # adding horizontal grid lines
    for ax in [ax1, ax2, ax3]:
        ax.yaxis.grid(True)
        ax.set_ylabel("Dataset")

    plt.show()

    if save_boxplot:
        file_name = f"eval_boxplots_{model_name}.png"
        fig.savefig(
            reports_dir / "figures" / file_name,
            dpi=100,
        )
        print(f"\nBoxplot of predicted rate was saved.")

    return fig


def log_plots(exp_id, run_id, density, boxplot):
    # If model reside in MLflow runs
    if (exp_id and run_id) is not None:
        with mlflow.start_run(experiment_id=exp_id, run_id=run_id):
            print(f"\nCalculate and log model's residuals...")
            mlflow.log_figure(density, "./plots/eval_distplot.png")
            mlflow.log_figure(boxplot, "./plots/eval_boxplots.png")


#%% Main function for main.py ==================================================
def main(
    exp_id=None,
    run_id=None,
    folder=None,
    save_predictions=True,
    save_distplot=True,
    save_boxplot=True,
):
    """Evaluates the trained model.
    Make prediction for raw train, validation and test sets.

    Args:
        * exp_id (int, optional): Model's MLflow experiment ID. Defaults to None.
        * run_id (str, optional): Model's MLflow run ID. Defaults to None.
        * folder (str, optional): Folder's  name for separatly saved model.\
            Defaults to None.
        * save_predictions (bool, optional): Toogle to save prediction.\
            Defaults to True.
        * save_distplot (bool, optional): Toogle to save separatly prediction's\
            density distributions plot. Defaults to True.
        * save_boxplot (bool, optional): Toogle to save separatly prediction's\
            boxplots distributions plot. Defaults to True.
    """
    print(f"-------------- START: Value model --------------")
    print_toggles(
        save_predictions,
        save_distplot,
        save_boxplot,
    )
    model_paths, model_names = get_model_paths(exp_id, run_id, models_dir, folder)
    model, model_name = load_saved_model(model_paths, model_names)
    if model is not None:
        X_train, y_train, X_val, y_val, X_test = load_raw_data()
        pred_train, pred_val, pred_test = predict_train_val_test(
            model, X_train, X_val, X_test, model_name, save_predictions
        )
        print_train_val_metrics(model, X_train, y_train, X_val, y_val)
        density = plot_density(pred_train, pred_val, pred_test, model_name, save_distplot)
        boxplot = plot_boxplot(pred_train, pred_val, pred_test, model_name, save_boxplot)
        log_plots(exp_id, run_id, density, boxplot)
        print(f"!!! DONE: Value model !!!")
    else:
        print(f"!!! DONE: No models evaluated !!!")
    winsound.Beep(frequency=3000, duration=300)


#%% Workflow ===================================================================
if __name__ == "__main__":
    set_pd_options()
    set_matlotlib_params()
    main(
        exp_id=3,
        run_id="1dc483bcea6d483cbbf939dfa35527dd",
        folder="20210310-1728_1dc483bcea6d483cbbf939dfa35527dd",
    )

#%%
