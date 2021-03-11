"""
    This is main script for current project aims to main task of project.
    It COMBAINS the several project's steps into one workflow:
    and APPLY needed:
        Functions
        Classes
        Pipelines
    and SAVE desired outcomes.

    Created on Jan 2021
    @author: mikhail.galkin
"""
#%%
if __name__ == "__main__":
    import sys
    import winsound

    sys.path.extend(["..", "../..", "../../.."])

    from gscreen.data.explore_data import main as explore_data
    from gscreen.data.process_data import main as process_data
    from gscreen.checks.check_pipe_preprocess import main as check_pipeline
    from gscreen.model.choose_model import main as choose_model
    from gscreen.model.fit_model import main as fit_model
    from gscreen.model.train_model import main as train_model
    from gscreen.checks.check_one_off_prediction import main as check_one_off_prediction
    from gscreen.model.value_model import main as value_model

    # None means 1 unless in a joblib.parallel_backend context.
    # -1 means using all processors.
    # * The number of jobs to run in parallel.
    n_jobs=-1
    pipeline_to_load = "data_process_fitted_on_train.joblib"
    outliers_mask_to_load = "outliers_mask_train_thold.joblib"

    explore_data(
        perfom_eda=True,
        pandas_profiling=False,
        sweetviz=False,
    )
    process_data(
        file="train_20201221.csv",
        save_pipeline=True,
        save_pipeline_as=pipeline_to_load,
        save_data_processed=True,  #!
        save_data_processed_as="df_train_processed",
        check_pipeline=True,
        get_outliers_mask=True,
        treshold_rate=50,
        iforest_n_estimators=100,
        iforest_contamination=0.0001,
        iforest_n_jobs=n_jobs,
    )
    check_pipeline(
        pipeline_to_load=pipeline_to_load,
        feature_cols_to_load="feature_cols.joblib",
        num_rows_example=2,
    )
    choose_model(
        pipeline_to_load=pipeline_to_load,
        outliers_mask_to_load=outliers_mask_to_load,
        train_wo_outliers=True,
        mlflow_tracking=True,
        eval_chosen=True,  #!
        n_jobs=n_jobs,
        fraction=0.2,  #!!
        n_splits=2,  #!!
        n_repeats=1,
    )
    fit_model(
        pipeline_to_load=pipeline_to_load,
        outliers_mask_to_load=outliers_mask_to_load,
        train_wo_outliers=True,
        log_residuals=True,  #!
        save_found_params=True,  #!
        random_grid_search=True,  #!
        bayesian_search=True,  #!
        simple_grid_search=False,  #!
        n_rand_sets_of_params=2,
        n_bayes_sets_of_params=2,
        n_jobs=n_jobs,
        fraction=0.2,  #!!
        n_splits=2,  #!!
        n_repeats=1,
    )
    exp_id, run_id, folder = train_model(
        pipeline_to_load=pipeline_to_load,
        save_pipeline_as="data_process_fitted_on_dev.joblib",
        outliers_mask_to_load=outliers_mask_to_load,
        train_wo_outliers=True,
        mlflow_tracking=True,
        log_residuals=True,
        save_mlmodel_separatly=True,
        n_estimators=1,  #!!
        n_jobs=n_jobs,
    )
    check_one_off_prediction(
        exp_id=exp_id,
        run_id=run_id,
        folder=folder,
        num_rows_1=True,
        num_rows_2=True,
    )
    value_model(
        folder=folder,
        exp_id=exp_id,
        run_id=run_id,
        save_predictions=True,  #!
        save_distplot=True,  #!
        save_boxplot=True,  #!
    )
    winsound.Beep(frequency=3500, duration=500)

#%%
