from nipype.interfaces.fsl import FLIRT
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=str)
parser.add_argument("--out_dir", type=str)
args = parser.parse_args()


flt = FLIRT(dof=7, cost_func='mutualinfo')
flt.inputs.in_file = args.in_file
flt.inputs.reference = '/home/at70870/fsl/data/linearMNI/MNI152lin_T1_1mm.nii.gz'
flt.inputs.out_file = os.path.join(args.out_dir, os.path.basename(args.in_file))
res = flt.run()