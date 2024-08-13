"""Module to store commands for plotting purpose"""

import os
import logging
from io import BytesIO
import requests
import comet_ml
from PIL import Image
from src import config
from src.utils.log import save_array_as_gif


def url_to_image(url: str) -> Image.Image:
    """Get image from url as Pillow Image

    Args:
        url (str): url of the image to retrieve.

    Raises:
        ValueError: raise if image retrieve fail

    Returns:
        Image: Retrieved image
    """
    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        raise ValueError(
            f"Failed to retrieve image from {url}, status code: {response.status_code}"
        )


def pretrain_calibration_gif(
    model_name: str,
    task: str,
    run_num: int,
    project_name: str,
    api_key: str = config.COMET_API_KEY,
):
    """Create and store gif representing the evolution
      of the pretraining calibration plot

    Args:
        model_name (str): name of the model to download
        task (str): task of the model to download
        run_num (int): specific run num to load
        project_name (str): name of the comet project
        api_key (str, optional): comet api key. Defaults to COMET_API_KEY.
    """
    api = comet_ml.api.API(
        api_key=api_key,
    )
    comet_exp_name = f"pretraining-{task}-{model_name}-{run_num}"
    logging.info(f"{comet_exp_name}, {project_name}")
    pretrain_exp: comet_ml.APIExperiment = api.get(
        "mrart", project_name, comet_exp_name
    )
    calibration_img = filter(
        lambda x: "calibration" in x["fileName"],
        pretrain_exp.get_asset_list(asset_type="image"),
    )
    array_of_pil: list[Image.Image] = []
    for asset in sorted(calibration_img, "step"):
        logging.info("Step %i", asset["step"])
        array_of_pil.append(url_to_image(asset["link"]))

    os.makedirs(config.PLOT_DIR, exist_ok=True)
    save_dir = os.path.join(config.PLOT_DIR, comet_exp_name, "calibration.gif")
    save_array_as_gif(array_of_pil, save_dir)
