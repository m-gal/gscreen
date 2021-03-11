"""
    Helps to reload project's module and get its inspections
    w\o reloading working space

    Created on Dec 2020
    @author: mikhail.galkin
"""

#%% Import libs
import sys
import inspect
import importlib

sys.path.extend(["..", "../..", "../../.."])
import gscreen

#%% CONFIG: Reload -------------------------------------------------------------
import gscreen.config

importlib.reload(gscreen.config)

print(inspect.getsource(gscreen.config))

#%% UTILS: Reload --------------------------------------------------------------
import gscreen.utils

importlib.reload(gscreen.utils)

print(inspect.getsource(gscreen.utils.calc_metrics))

#%% CLASSES: Reload ------------------------------------------------------------
import gscreen.classes

importlib.reload(gscreen.classes)

#%% PLOTS: Reload --------------------------------------------------------------
import gscreen.plots

importlib.reload(gscreen.plots)

print(inspect.getsource(gscreen.plots.plot_residuals_errors))

#%% ALGO: BaggingRegressor -----------------------------------------------------
import gscreen.model.algos

importlib.reload(gscreen.model.algos)

print(inspect.getsource(gscreen.model.algos.get_bagging_model))
print(inspect.getsource(gscreen.model.algos.get_bagging_search_params))


#%%
