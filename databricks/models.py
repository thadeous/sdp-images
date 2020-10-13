from typing import Union, Tuple, Optional
import click
import time
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus


@click.group()
def main():
    pass


@main.command()
@click.argument('model_name', type=str)
def delete(model_name: str):
    print(f"Deleting the registered model named {model_name}...")
    client = MlflowClient(tracking_uri="databricks")
    for version in client.search_model_versions("name = '%s'" % model_name):
        if version.current_stage != "None":
            print(f"Transition {model_name}:{version.version} from {version.current_stage} to None...")
            client.transition_model_version_stage(model_name, version.version, "None")
    client.delete_registered_model(model_name)


@main.command()
@click.argument('model_name', type=str)
@click.argument('experiment_id', type=int)
@click.argument('metric', type=str)
@click.option('--highest/--lowest', default=True)
@click.option('--databricks_token', '-t', envvar="DATABRICKS_TOKEN")
def stage(model_name: str, experiment_id, metric: str, highest: bool, databricks_token: str):
    client = MlflowClient(tracking_uri="databricks")
    runs = client.search_runs(experiment_ids=experiment_id)
    max_run: Tuple[Optional[str], Union[float, int]] = (None, -float('inf') if highest else float('inf'))
    for run in runs:
        if highest:
            if run.data.metrics[metric] > max_run[1]:
                max_run = (run.info.run_id, run.data.metrics[metric])
        else:
            if run.data.metrics[metric] < max_run[1]:
                max_run = (run.info.run_id, run.data.metrics[metric])
    if max_run[0] is None:
        print(f"No runs found in the provided experiment {experiment_id}")
        return
    print(f"Run {max_run[0]} with {metric} = {max_run[1]} chosen for registration.")
    for version in client.search_model_versions("name = '%s'" % model_name):
        if version.run_id == max_run[0]:
            if version.current_stage != "Staging":
                print(f"Highest performing model is versioned but not in 'Staging'. Promoting..")
                client.transition_model_version_stage(model_name, version.version, "Staging")
                return
    print("Best model not registered. Registering...")
    register(max_run[0], experiment_id, model_name, True)


def register(run_id: str, experiment_id: int, model_name: str, block: bool):
    client = MlflowClient(tracking_uri="databricks")
    if not any([m.name == model_name for m in client.list_registered_models()]):
        client.create_registered_model(model_name)
    source = f"dbfs:/databricks/mlflow-tracking/{experiment_id}/{run_id}/artifacts/model"
    model_details = client.create_model_version(model_name, source, run_id)

    if block:
        def wait_until_ready() -> bool:
            for _ in range(60):
                model_version_details = client.get_model_version(
                    name=model_name,
                    version=model_details.version,
                )
                status = ModelVersionStatus.from_string(model_version_details.status)
                print("Model status: %s" % ModelVersionStatus.to_string(status))
                if status == ModelVersionStatus.READY:
                    return True
                time.sleep(5)
            return False

        if not wait_until_ready():
            print(f"Timeout waiting on registration of model.  Will not stage model.")
            return
    client.transition_model_version_stage(model_name, model_details.version, "Staging")


@main.command()
def list():
    print(MlflowClient(tracking_uri="databricks").list_experiments())


@main.command()
@click.argument("model_name", type=str)
@click.option('--out_path', default='.')
def download(model_name: str, out_path: str):
    import json
    client = MlflowClient(tracking_uri="databricks")
    for rm in client.list_registered_models():
        if rm.name == model_name:
            for version in rm.latest_versions:
                if version.current_stage == "Staging":
                    print(f"Downloading verion {version.version} of {model_name} with run_id {version.run_id}...")
                    client.download_artifacts(version.run_id, "model/data/model.pt", '.')
                    meta = {
                        "version": version.version,
                        "run_id": version.run_id,
                        "name": model_name
                    }
                    with open("model_meta.json", "w") as m:
                        json.dump(meta, m)

if __name__ == "__main__":
    main()
