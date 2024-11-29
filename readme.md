# Improving Quality Control of MRI Images Using Synthetic Motion Data

This repo contains the code to reproduce the results presented in our ISBI25 article (submitted)

- [Improving Quality Control of MRI Images Using Synthetic Motion Data](#improving-quality-control-of-mri-images-using-synthetic-motion-data)
  * [Structure](#structure)
  * [Source code](#source-code)
  * [Reproducing](#reproducing)
    + [Datasets](#datasets)
    + [Training](#training)
      - [Pretraining](#pretraining)
      - [Scratch](#scratch)
      - [Transfer](#transfer)
    + [Testing](#testing)
      - [Pretraining](#pretraining-1)
      - [Scratch](#scratch-1)
      - [Transfer](#transfer-1)
    + [SLURM](#slurm)
  * [Questions, Comments and Contributions](#questions--comments-and-contributions)
  * [Cite](#cite)


## Structure

- In `notebooks` : simple code to generate our data for submission.
- In `report` : the final test inference data presented in the article.
- In `slurms` : some simple SLURM scripts to run [clinica's t1 linear pipeline](https://aramislab.paris.inria.fr/clinica/docs/public/dev/Pipelines/T1_Linear/).
- In `src` : the code used to run our experiments
- In `test` : unit test for important, handmade, part of the code.
- `cli.py` : Python file containing the CLI to run our experiments.
- `.env_example` : template to create your own .env file.

## Source code

Inside the `src` folder, you will find :
- `commands` :  each file contain procedure to execute commands from `cli.py`.
- `dataset` : one folder per dataset with associated dataset and datamodule definition. You will also find the base class to define new datasets.
- `motion` : this contain almost unmodified code from ["Quantifying MR head motion in the Rhineland Study–A robust method for population cohorts"](https://github.com/Deep-MI/head-motion-tools/tree/main). It is used to estimate motion.
- `network` : architecture definitions and base classes to define new networks
- `training` : lightning modules containing the logic for each training setting (pretraining, transfer learning and training from scratch). Also contains base class containing common logic that you can extend for your own setting.
- `transforms` : two files containing data pipeline to generate data (synthetic data) and load data for training.
- `utils` :  some common utilities

## Reproducing

### Datasets

You need at least two dataset :
- One with clean volumes for synthetic data generation and pretraining
- One with quality control scores for transfer learning and training from scratch

You can follow our datasets definitions for your own, outside of the code, it just requires a file with a `data` field corresponding to the path between the root of your data directory and the volumes. You also want to have identifiers and labels but those can be configured.

If you define a new dataset, you will certainly need to replace the lightning datamodules used in our `commands` files.

### Training

Before launching any commands, make sure that your quality control score correspond to the `num_classes` variable define in the lightning logic for your task (see `training`), you can even make your own LightningModule for this purpose.

Once everything is in place, you can run :

#### Pretraining
```bash
python cli.py pretrain --max_epochs <num_epochs> --learning_rate <lr> --batch_size <batch_size> 
```

#### Scratch
```bash
python cli.py train --max_epochs <num_epochs> --learning_rate <lr> --batch_size <batch_size> 
```

#### Transfer
```bash
python cli.py transfer --pretrain_path <path to your pretrained model checkpoint> --max_epochs <num_epochs> --learning_rate <lr> --batch_size <batch_size>
```

For every commands, you can use `--help` to get more information and personalize further.

### Testing

Now that you have trained your model, you want to test them against test data. Again, before running any commands, check that the dataset used in the code are yours.

#### Pretraining
```bash
python cli.py test pretrain -d <path to a root directory containing checkpoint(s)>
```

#### Scratch
```bash
python cli.py test scratch -d <path to a root directory containing checkpoint(s)> 
```

#### Transfer
```bash
python cli.py test transfer -d <path to a root directory containing checkpoint(s)>
```

You will find every results in the `report` folder.

### SLURM

If you have access to a SLURM cluster, you can use the `launch` group of commands to automatically launch one SLURM jobs per seed. For more information check the commands define in `cli.py` or use `--help`.

## Questions, Comments and Contributions

If you have questions, comments you can send them to my email address bricout.charles@outlook.com. Please use a tag like "[motiondetector]" in the object of your mail.

We do not accept direct contribution as this repository should always correspond to our ISBI article. I encourage you to make a fork and email me if you need any help !

## Cite

Not submitted yet