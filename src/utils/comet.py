import comet_ml
import torch
import tempfile
import ast
import glob
from src.network.utils import parse_model
from src.training.lightning_logic import PretrainingTask


def get_pretrain_task(model_name:str, run_num:int, project_name:str):
    api = comet_ml.api.API(
        api_key="WmA69YL7Rj2AfKqwILBjhJM3k",
    )
    pretrain_exp = api.get("mrart", project_name, f"pretraining-{model_name}-{run_num}")
    model_class = parse_model(model_name)

    output_path=f"/home/cbricout/scratch/{project_name}-{run_num}/{model_class.__name__}"
    pretrain_exp.download_model(
        model_class.__name__,
        output_path=output_path,
    )
    file_path = glob.glob(f"{output_path}/*.ckpt")[0]
    pretrained = PretrainingTask.load_from_checkpoint(
        file_path
    )
    return pretrained

def get_model_from_exp(exp: comet_ml.ExistingExperiment, project_name):
    model_name = exp.get_parameters_summary("model")["valueCurrent"]
    im_shape = ast.literal_eval(exp.get_parameters_summary("im_shape")["valueCurrent"])
    output_class = int(exp.get_parameters_summary("output_class")["valueCurrent"])

    model_class = parse_model(model_name)
    tempdir = tempfile.TemporaryDirectory()
    net = model_class(
        im_shape, run_name=tempdir.name, output_class=output_class, pretrain=True
    )

    if len(exp.get_parameters_summary("run_num")) > 0:
        run_num = exp.get_parameters_summary("run_num")["valueCurrent"]
        exp.download_model(
            model_name,
            output_path=f"/home/cbricout/scratch/{project_name}-{run_num}/{model_name}",
        )
        net.load_state_dict(
            torch.load(
                f"/home/cbricout/scratch/{project_name}-{run_num}/{model_name}/model-data/comet-torch-model.pth"
            )
        )
    else:
        exp.download_model(
            model_name,
            output_path=f"/home/cbricout/scratch/{project_name}/{model_name}",
        )
        net.load_state_dict(
            torch.load(
                f"/home/cbricout/scratch/{project_name}/{model_name}/model-data/comet-torch-model.pth"
            )
        )
    return net


def get_experiment_key(api_key, workspace_name, project_name, run_name):
    api = comet_ml.api.API(
        api_key=api_key,
    )
    exp = api.get_experiment(
        workspace=workspace_name, project_name=project_name, experiment=run_name
    )
    if exp:
        return exp.key
    return None
