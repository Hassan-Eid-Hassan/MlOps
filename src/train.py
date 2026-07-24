"""
Train step of the diabetes regression pipeline.

Simplified from the original MLOpsPython (Azure ML) version: no Workspace,
Run context, or Datastore. MLflow handles experiment tracking and model
packaging (including the input/output signature used later to auto-generate
the serving container).
"""
import argparse
import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame):
    X = df.drop("Y", axis=1)
    y = df["Y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    return {
        "train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test},
    }


def train_model(data, ridge_args):
    reg_model = Ridge(**ridge_args)
    reg_model.fit(data["train"]["X"], data["train"]["y"])
    return reg_model


def get_model_metrics(model, data):
    preds = model.predict(data["test"]["X"])
    mse = mean_squared_error(data["test"]["y"], preds)
    return {"mse": mse}


def write_github_output(**kwargs):
    """Expose values to subsequent GitHub Actions steps, if running in CI."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as fh:
        for key, value in kwargs.items():
            fh.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--data-path", default="data/diabetes.csv")
    parser.add_argument("--params-path", default="parameters.json")
    parser.add_argument("--experiment-name", default="diabetes_regression")
    args = parser.parse_args()

    with open(args.params_path) as f:
        params = json.load(f)
    train_args = params.get("training", {"alpha": 0.5})

    mlflow.set_experiment(args.experiment_name)

    df = pd.read_csv(args.data_path)
    data = split_data(df)

    with mlflow.start_run() as run:
        model = train_model(data, train_args)
        metrics = get_model_metrics(model, data)

        mlflow.log_params(train_args)
        mlflow.log_metrics(metrics)

        signature = mlflow.models.infer_signature(
            data["train"]["X"], model.predict(data["train"]["X"])
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=data["train"]["X"].iloc[:2],
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Metrics: {metrics}")

        write_github_output(run_id=run.info.run_id, mse=metrics["mse"])


if __name__ == "__main__":
    main()
