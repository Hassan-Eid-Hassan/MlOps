"""
Evaluate step of the pipeline.

Replaces the original Azure ML `evaluate_model.py`. Instead of comparing
against a model tagged in an Azure ML workspace, this compares the newly
trained run's metric against whatever MLflow Model Registry version
currently holds the "production" alias. If the new model is better (or
there is no production model yet), it signals downstream steps to promote
and register it.
"""
import argparse
import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def write_github_output(**kwargs):
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as fh:
        for key, value in kwargs.items():
            fh.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser("evaluate")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default="diabetes_regression_model")
    parser.add_argument("--metric", default="mse")
    args = parser.parse_args()

    client = MlflowClient()
    new_run = client.get_run(args.run_id)
    new_metric = new_run.data.metrics[args.metric]

    promote = True
    try:
        prod_version = client.get_model_version_by_alias(args.model_name, "production")
        prod_run = client.get_run(prod_version.run_id)
        prod_metric = prod_run.data.metrics.get(args.metric)
        print(f"Current production {args.metric}: {prod_metric}")
        print(f"New candidate {args.metric}:      {new_metric}")
        promote = prod_metric is None or new_metric < prod_metric
    except MlflowException:
        print("No production model registered yet - this will be the first one.")

    print(f"Promote candidate: {promote}")
    write_github_output(promote=str(promote).lower())


if __name__ == "__main__":
    main()
