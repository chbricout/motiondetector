"""Module to store commands for plotting purpose"""

import os
from io import BytesIO
import requests
import comet_ml
from PIL import Image
import tqdm
import cairosvg
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
        if ".svg" in url:
            png = cairosvg.svg2png(response.content)
        else:
            png = response.content
        image = Image.open(BytesIO(png))
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
    api_key: str | None = config.COMET_API_KEY,
):
    """Create and store gif representing the evolution
      of the pretraining calibration plot

    Args:
        model_name (str): name of the model to download
        task (str): task of the model to download
        run_num (int): specific run num to load
        project_name (str): name of the comet project
        api_key (str | None, optional): comet api key. Defaults to COMET_API_KEY.
    """
    api = comet_ml.api.API(
        api_key=api_key,
    )
    comet_exp_name = f"pretraining-{task}-{model_name}-{run_num}"
    pretrain_exp: comet_ml.APIExperiment = api.get(
        "mrart", project_name, comet_exp_name
    )
    calibration_img = filter(
        lambda x: "calibration" in x["fileName"],
        pretrain_exp.get_asset_list(asset_type="image"),
    )

    array_of_pil: list[Image.Image] = []
    for asset in tqdm.tqdm(sorted(calibration_img, key=lambda x: x["step"])):
        array_of_pil.append(url_to_image(asset["link"]))

    save_dir = os.path.join(config.PLOT_DIR, comet_exp_name)
    os.makedirs(save_dir, exist_ok=True)

    save_array_as_gif(array_of_pil, os.path.join(save_dir, "calibration.gif"))
