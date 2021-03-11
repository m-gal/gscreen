#%% Load libraries
import sys
import pickle
import xgboost as xgb

#%% Load project's stuff
sys.path.extend(["..", "../..", "../../.."])
import gscreen

# Load project utils and classes
from gscreen.config import models_dir
from gscreen.config import reports_dir
from gscreen.config import data_processed_dir

#%% Load model
model_to_check_name = "model_xgb_final.pkl"
with open(models_dir / model_to_check_name, "rb") as f:
    model_xgb_final = pickle.load(f)

#%% Print out Feature Importanse
fmap = data_processed_dir / "feature_map.txt"
xgb_fi_weight = xgb.plot_importance(
    model_xgb_final,
    title="XGBoost: Feature importance by Weight",
    xlabel="F score",
    fmap=fmap,
    show_values=True,
    importance_type="weight",  #  is the number of times a feature appears in a tree
)

xgb_fi_gain = xgb.plot_importance(
    model_xgb_final,
    title="XGBoost: Feature importance by Gain",
    xlabel="Gain",
    fmap=fmap,
    show_values=False,
    importance_type="gain",  # is the average gain of splits which use the feature
)

xgb_fi_cover = xgb.plot_importance(
    model_xgb_final,
    title="XGBoost: Feature importance by Cover",
    xlabel="Cover",
    fmap=fmap,
    show_values=False,
    importance_type="cover",  #  is the average coverage of splits
    # which use the feature where coverage is defined as the number of samples
    # affected by the split
)
# Save pics
xgb_fi_weight.figure.savefig(reports_dir / "xgb_final_fi_weight.png")
xgb_fi_gain.figure.savefig(reports_dir / "xgb_final_fi_gain.png")
xgb_fi_cover.figure.savefig(reports_dir / "xgb_final_fi_cover.png")

#%% Plot specified tree.
xgb.plot_tree(model_xgb_final, fmap=fmap)

#%% Convert specified tree to graphviz instance and plot it
xgb_tree = xgb.to_graphviz(model_xgb_final, fmap=fmap)
type(xgb_tree)
#%%
import pydotplus

pydot_graph = pydotplus.graph_from_dot_data(xgb_tree)
pydot_graph.set_size('"5,5!"')
# pydot_graph.write_png(reports_dir / 'xgb_final_tree.png')
# pydot_graph.set_size('"5,5!"')
# pydot_graph.write_png('resized_tree.png')

#%%
