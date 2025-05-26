import os
import numpy as np
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
REGISTERED_MODEL_NAME = "random-forest-regressor"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run() as run:
        new_params = {}
        for param in RF_PARAMS:
            new_params[param] = int(params[param])

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        # Evaluate model on validation and test sets
        val_rmse = np.sqrt(mean_squared_error(y_val, rf.predict(X_val)))
        test_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))

        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)

        # Log the model artifact (needed for registration)
        mlflow.sklearn.log_model(rf, artifact_path="model")

        return run.info.run_id, test_rmse


@click.command()
@click.option(
    "--data_path",
    default="../data_preprocessed/",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    # Get top_n runs from HPO experiment
    hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    hpo_runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    # Re-train and evaluate the top models on the test set
    best_rmse = float("inf")
    best_run_id = None

    for run in hpo_runs:
        run_id, test_rmse = train_and_log_model(data_path=data_path, params=run.data.params)
        print(f"Logged run {run_id} with test RMSE: {test_rmse:.4f}")
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_run_id = run_id

    # Register the best model
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
        print(f"✅ Registered best model: {model_uri} as '{REGISTERED_MODEL_NAME}'")


if __name__ == '__main__':
    run_register_model()
