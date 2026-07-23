"""
Register step of the pipeline.

Replaces the original Azure ML `register_model.py`. No Workspace or
build-id tagging logic needed: MLflow's Model Registry already versions
every registration and keeps the training run (params, metrics, dataset)
linked to the model version.
"""
import argparse

import mlflow
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser("register")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default="diabetes_regression_model")
    args = parser.parse_args()

    model_uri = f"runs:/{args.run_id}/model"
    result = mlflow.register_model(model_uri, args.model_name)
    print(f"Registered '{args.model_name}' as version {result.version}")

    client = MlflowClient()
    client.set_registered_model_alias(args.model_name, "production", result.version)
    print(f"Alias 'production' -> version {result.version}")


if __name__ == "__main__":
    main()
