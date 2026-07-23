# Diabetes Regression - Simplified MLOps Pipeline

A trimmed-down rebuild of `microsoft/MLOpsPython/diabetes_regression`, with every
Azure ML-specific piece (`Workspace`, `Run.get_context()`, `Datastore`, custom
`score.py` init/run functions) replaced by **MLflow, actually deployed on
Kubernetes**, plus a K8s-deployed serving container. Built for a 5-minute
conference demo slot, not for production - see "Where to go from here" for
what to harden before that.

## Architecture

```
                     ┌─────────────────────── kind cluster ───────────────────────┐
                     │                                                            │
  data-validation    │   ┌────────────────┐        ┌──────────────────────┐       │
  (pytest, no k8s) → │   │  mlflow-server │ ◄───── │ train / evaluate /   │       │
                     │   │  Deployment    │  HTTP  │ register (run on the │       │
                     │   │  (tracking +   │        │ GitHub Actions       │       │
                     │   │  registry UI)  │        │ runner, port-forward)│       │
                     │   └───────┬────────┘        └──────────────────────┘       │
                     │           │ registry: models:/diabetes_regression_model@production
                     │           ▼                                                │
                     │   mlflow models build-docker  →  diabetes-model-server pod │
                     │                                        │                   │
                     └────────────────────────────────────────┼───────────────────┘
                                                              ▼
                                                   Streamlit test app / curl
```

MLflow isn't a local file or an implementation detail here - it's a real pod
with its own Deployment and Service, reachable over HTTP, with a browsable UI
for experiments and the model registry. That's the piece you port-forward to
and show live.

## What changed vs. the original repo

| Original file | Replaced by | Why |
|---|---|---|
| `data_test.py` | `tests/test_data_validation.py` | Same 4 checks (schema, bad schema, missing values, drift), just pointed at a local `data/` folder instead of an AML path helper |
| `train.py` + `train_aml.py` | `src/train.py` | One script. No dataset registration/`Workspace` - reads a CSV and logs params/metrics/model to the MLflow server on k8s |
| `evaluate_model.py` | `src/evaluate.py` | Compares the new run's MSE against whatever model currently holds the `production` alias in the **MLflow Model Registry** |
| `register_model.py` | `src/register.py` | `mlflow.register_model(...)` + `set_registered_model_alias(...)` - MLflow already links model versions to their run, params, metrics, and dataset |
| `score.py`, `scoreA.py`, `scoreB.py` | *(deleted)* | `mlflow models build-docker` generates the serving container and `/invocations` endpoint straight from the model's signature - no hand-written scoring code |
| *(none - new)* | `Dockerfile.mlflow-server` + `k8s/mlflow-server-*.yaml` | Packages `mlflow server` (tracking + registry) as an actual k8s Deployment/Service |
| `parameters.json` | kept, trimmed | Same idea (external config for alpha etc.), minus the empty `evaluation`/`scoring` sections that weren't used |

## Pipeline (`.github/workflows/mlops-pipeline.yml`)

```
data-validation → ml-pipeline:
  1. create a kind cluster
  2. build + deploy the MLflow server to it, wait for it to be healthy
  3. port-forward to it, then train → evaluate → register against it
  4. (only if the new model beats production) build the serving image
     straight from the registered model, load it into kind, deploy it
  5. smoke-test /invocations
```

Everything runs inside one throwaway **kind** (Kubernetes-in-Docker) cluster
per pipeline run - no cloud credentials, no container registry, nothing that
can go down mid-talk. Images are loaded straight into the cluster with
`kind load docker-image`. Swap the "Create local kind cluster" step for
`kubectl config use-context <your-aks-cluster>` and everything else - the
manifests, the MLflow deployment, the build-docker step - is unchanged.

## Running it locally

```bash
pip install -r requirements.txt

# Start MLflow the same way the container does (takes ~30s to fully warm up)
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000 \
  --allowed-hosts "*" --cors-allowed-origins "*"

# In another terminal:
export MLFLOW_TRACKING_URI="http://localhost:5000"
pytest tests/ -v

python src/train.py                       # prints a run_id, and a link to the MLflow UI
python src/evaluate.py --run-id <run_id>  # prints promote=true/false
python src/register.py --run-id <run_id>  # only if promote=true

# Build + run the serving image (needs Docker)
mlflow models build-docker --model-uri "models:/diabetes_regression_model@production" --name diabetes-model-server
docker run -p 8080:8080 diabetes-model-server

# During the talk:
streamlit run app/streamlit_app.py
```

Open `http://localhost:5000` in a browser at any point to show the MLflow UI
- experiments, runs, metrics, and the registered model with its `production`
alias.

## Where to go from here (i.e. what's cut for time)

- **Storage**: the MLflow server uses `emptyDir` + SQLite so the whole thing
  is disposable and demo-safe. For real use, back it with a
  PersistentVolumeClaim and a real DB (Postgres) + object storage (S3/Blob)
  so history survives pod restarts.
- **Cluster**: swap `kind` for your real AKS/EKS/GKE context once you're past
  the demo, and push images to a real registry instead of `kind load`.
- **Security**: the MLflow server runs with `--allowed-hosts "*"` for demo
  convenience. Lock this to real hostnames before exposing it beyond
  localhost/port-forward.
- **Approval gate**: promotion is currently automatic on a better MSE; add a
  manual approval environment in GitHub Actions before the deploy step if you
  want a human in the loop.
- **Rollback**: the deployment just overwrites the image tag; consider
  `kubectl rollout undo` wiring or a canary/blue-green step if this becomes a
  real service.
