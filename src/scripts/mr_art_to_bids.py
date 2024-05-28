import os
import argparse
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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", type=str, required=True)
    parser.add_argument("-o","--output", type=str, required=True)

    args = parser.parse_args()
    assert os.path.exists(args.input)

    if not os.path.exists(args.output):
        logging.info(f"Output folder does not exist. Creating {args.output}")
        os.makedirs(args.output)
    
    for subject_path in tqdm.tqdm(glob.glob(os.path.join(args.input, "sub-*")), desc="Processing subjects", unit=" subjects"):
        sub_id = os.path.basename(subject_path)
        for session_path in glob.glob(os.path.join(subject_path, "anat", "*.nii.gz")):
            session_path = session_path.replace(".nii.gz", "")
            orig_name = os.path.basename(session_path)
            ses_id = get_ses_id(orig_name)
            fullname = f"{sub_id}_{ses_id}_T1w"
            outses_dir = os.path.join(args.output, sub_id, ses_id, "anat")
            os.makedirs(outses_dir, exist_ok=True)
            outses_path = os.path.join(outses_dir, fullname)

            shutil.copy(f"{session_path}.nii.gz", f"{outses_path}.nii.gz")
            shutil.copy(f"{session_path}.json", f"{outses_path}.json")
