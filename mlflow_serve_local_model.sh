#!/usr/bin/env sh

echo "Deploy productioned model from Model Registry..."

#* MLflow manual:
#* https://mlflow.org/docs/latest/model-registry.html#serving-an-mlflow-model-from-model-registry

## Set environment variable for the tracking URL where the Model Registry resides
export MLFLOW_TRACKING_URI=sqlite:///mlflow/mlregistry.db
## Serve the production model from the model registry
mlflow models serve -m models:/gscreen_model/production -p 1234

## Remove empty mlruns created automatically
/bin/rm -f -r mlruns
