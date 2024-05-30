import shutil
import glob
import os
import tqdm

if __name__=='__main__':
    main_path = "/home/cbricout/projects/def-sbouix/data/AMPSCZ/rawdata/sub-*/ses-*/anat/*_T1w*auxi*.nii*"
    source_dir="/home/cbricout/scratch/AMPSCZ-bids"
    all_T1w = glob.glob(main_path)

    for source in tqdm.tqdm(all_T1w):
        dest = source.replace("/home/cbricout/projects/def-sbouix/data/AMPSCZ/rawdata", source_dir)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(source, dest)
        