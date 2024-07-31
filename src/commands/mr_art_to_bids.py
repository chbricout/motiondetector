import os
import logging
import glob
import tqdm
import shutil


def get_ses_id(filename):
    if "standard" in filename:
        return "ses-standard"
    elif "headmotion1" in filename:
        return "ses-headmotion1"
    elif "headmotion2" in filename:
        return "ses-headmotion2"


def launch_convert_mrart_to_bids(input_path: str, output_path: str):
    """Convert MRART dataset to BIDS structure

    Args:
        input (str): Input folder path
        output (str): Output folder path
    """

    assert os.path.exists(input_path)

    if not os.path.exists(output_path):
        logging.info(f"Output folder does not exist. Creating {output_path}")
        os.makedirs(output_path)

    for subject_path in tqdm.tqdm(
        glob.glob(os.path.join(input_path, "sub-*")),
        desc="Processing subjects",
        unit=" subjects",
    ):
        sub_id = os.path.basename(subject_path)
        for session_path in glob.glob(os.path.join(subject_path, "anat", "*.nii.gz")):
            session_path = session_path.replace(".nii.gz", "")
            orig_name = os.path.basename(session_path)
            ses_id = get_ses_id(orig_name)
            fullname = f"{sub_id}_{ses_id}_T1w"
            outses_dir = os.path.join(output_path, sub_id, ses_id, "anat")
            os.makedirs(outses_dir, exist_ok=True)
            outses_path = os.path.join(outses_dir, fullname)

            shutil.copy(f"{session_path}.nii.gz", f"{outses_path}.nii.gz")
            shutil.copy(f"{session_path}.json", f"{outses_path}.json")
