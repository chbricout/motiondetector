"""
Module to launch different standard command through slurm jobs
"""

import re
import sys
from collections.abc import Sequence
from os import path

from simple_slurm import Slurm

from src import config
from src.config import DEFAULT_SLURM_ACCOUNT


def setup_python(job: Slurm):
    """Add command to a slurm job to activate necessary modules and environments


    Args:
        job (Slurm): slurm job to modify
    """
    job.add_cmd("mkdir -p $SLURM_TMPDIR/.triton/cache")
    job.add_cmd("export TRITON_CACHE_DIR=$SLURM_TMPDIR/.triton/cache")
    job.add_cmd('echo "Triton is setup"')

    job.add_cmd("module load python cuda httpproxy arrow r-bundle-bioconductor")
    job.add_cmd("source ~/fix_bowl/bin/activate")
    job.add_cmd('echo "python is setup"')


def cpy_extract_tar(job: Slurm, tarballs_name: Sequence[str]):
    """Copy and extract tarball files before job

    Args:
        job (Slurm): Job to modify
        tarballs_name (Sequence[str]): name of the tarball faile to use
    """
    job.add_cmd("mkdir -p $SLURM_TMPDIR/datasets")
    for ds in tarballs_name:
        job.add_cmd(
            f"tar --skip-old-file -xf ~/scratch/{ds}.tar -C $SLURM_TMPDIR/datasets"
        )
        job.add_cmd(f'echo "{ds} copied"')


def cpy_extract_pretrain(job: Slurm):
    """Copy and extract tarball files before job

    Args:
        job (Slurm): Job to modify
        tarballs_name (Sequence[str]): name of the tarball faile to use
    """
    job.add_cmd("mkdir -p $SLURM_TMPDIR/datasets")
    job.add_cmd(
        "tar --skip-old-file  -xf ~/scratch/generate_dataset.tar -C $SLURM_TMPDIR/datasets"
    )
    job.add_cmd('echo "Pretrained dataset copied"')


def copy_data_tmp_generate(job: Slurm):
    """Extract data from scratch to $SLURM_TMPDIR for pretraining dataset

    Args:
        job (Slurm): slurm job to modify
    """
    to_cpy = ["HCPEP-Preproc", "AMPSCZ-Preproc"]
    job.add_cmd("mkdir -p $SLURM_TMPDIR/datasets")
    for ds in to_cpy:
        job.add_cmd(
            f"tar --skip-old-file -xf ~/scratch/{ds}.tar home/cbricout/scratch -C $SLURM_TMPDIR/datasets --strip-components 3"
        )
        job.add_cmd(f'echo "{ds} copied"')


def get_full_cmd() -> str:
    """Reconstruct the command use to launch the current program
    It is needed to launch the cli from inside the slurm job with the arguments provided
      when using the --slurm flag.

    Returns:
        str: the command arguments
    """
    full_command = " ".join(sys.argv)
    full_command = full_command.replace(" -S", " ").replace("--slurm", "")
    if full_command.find("--run_num"):
        full_command = re.sub(
            r"(--run_num)\s?[^\s-]*", r"\1 $SLURM_ARRAY_TASK_ID", full_command
        )

    return full_command


def get_transfer_cmd_from_pretrain(cmd: str) -> str:
    """Create the corresponding transfer command from a pretrain command
    Used to queue transfer job after pretrain

    Args:
        cmd (str): the pretrain command to modify

    Returns:
        str: corresponding transfer command
    """
    full_command = cmd.replace("pretrain", "transfer")
    if full_command.find("--dropout"):
        full_command = re.sub(r"(--dropout)\s?[^\s-]*", "", full_command)
    return full_command


def get_output(prefix: str, model: str, array: Sequence[int] | int | None) -> str:
    """Return the output log path for a job

    Args:
        prefix (str): job prefix (ex.: "pretrain", "transfer",...)
        model (str): the model used in the job
        array (Sequence[int] | int | None): raw array parameter given to the function

    Returns:
        str: output log path
    """
    base = f"./logs/{prefix}-{model}"
    if array is not None:
        base += f"_{Slurm.JOB_ARRAY_ID}"
    base += f".{Slurm.JOB_ARRAY_MASTER_ID}.out"
    return base


def get_name(prefix: str, model: str, array: Sequence[int] | int | None) -> str:
    """Return the name for a job

    Args:
        prefix (str): job prefix (ex.: "pretrain", "transfer",...)
        model (str): the model used in the job
        array (Sequence[int] | int | None): raw array parameter given to the function

    Returns:
        str: job name
    """
    base = f"{prefix}_{model}"
    if array is not None:
        base += f"_{Slurm.JOB_ARRAY_ID}"
    return base


def create_job(
    name: str,
    array: Sequence[int] | int | None,
    output: str,
    n_cpus: int,
    n_gpus: int,
    account=DEFAULT_SLURM_ACCOUNT,
    mem="300G",
    time="24:00:00",
    nodes=1,
) -> Slurm:
    """Generate a basic job with requeu and python setup

    Args:
        name (str): Job name
        array (Sequence[int] | int | None): Array parameter: single id, sequence, range or nothing
        output (str): Output path
        n_cpus (int): Number of CPUs to allocate
        n_gpus (int): Number of GPUs to allocate
        account (_type_, optional): Account id to use. Defaults to DEFAULT_SLURM_ACCOUNT
            (see config.py).
        mem (str, optional): RAM to allocate. Defaults to "200G".
        time (str, optional): Time to allocate ressources for. Defaults to "24:00:00".

    Returns:
        Slurm: The job with requeu enabled and a ready python environment
    """
    job = Slurm(
        job_name=name,
        nodes=nodes,
        cpus_per_task=n_cpus,
        ntasks_per_node=max(n_gpus, 1),
        ntasks=max(n_gpus, 1),
        mem=mem,
        time=time,
        account=account,
        signal="SIGUSR1@90",
        requeue=True,
        output=output,
    )
    if n_gpus > 0:
        job.add_arguments(gpus_per_node=n_gpus)
    if not array is None:
        job.add_arguments(array=array)
    setup_python(job)
    return job


def submit_pretrain(
    model: str,
    array: Sequence[int] | int | None = None,
    cmd: str | None = None,
    account: str = config.DEFAULT_SLURM_ACCOUNT,
):
    """Submit pretrain job on SLURM cluster

    Args:
        model (str): Model to use
        array (Sequence[int] | int | None, optional): Can be single id, sequence, range or nothing.
            Defaults to None.
        cmd (str | None, optional): command to run, if None, retrieve the parameters
            used from the CLI. Defaults to None.
        send_transfer (bool, optional): flag to send transfer command. Defaults to False.
    """
    job = create_job(
        get_name("pretrain", model, array),
        array,
        get_output("pretrain", model, array),
        n_cpus=10,
        n_gpus=4,
        mem="300G",
        time="48:00:00",
        account=account,
    )

    cpy_extract_pretrain(job)
    cpy_extract_tar(job, "AMPSCZ-Preproc")

    if cmd is None:
        cmd = get_full_cmd()
    else:
        if not "--run_num" in cmd and array is not None:
            cmd += " --run_num $SLURM_ARRAY_TASK_ID"

    job_id = job.sbatch(f"srun python3 {cmd}")
    print(job)


def submit_transfer(
    pretrain_path: str,
    array: Sequence[int] | int | None = None,
    cmd: str | None = None,
    dependency: str | None = None,
):
    """Submit transfer job on SLURM cluster

    Args:
        model (str): Model to use
        array (Sequence[int] | int | None, optional): Can be single id, sequence, range or nothing.
            Defaults to None.
        cmd (str | None, optional): command to run, if None, retrieve the parameters
            used from the CLI. Defaults to None.
        dependency (str | None, optional): optionnal job_id dependency to wait for.
            Defaults to None.
        dataset (str, optional): dataset to run the transfer process on
            (used for job name and output). Defaults to "".
    """
    job = create_job(
        get_name("transfer", path.basename(pretrain_path), array),
        array,
        get_output("transfer", path.basename(pretrain_path), array),
        n_cpus=20,
        n_gpus=1,
        mem="100G",
        time="1:00:00",
    )
    cpy_extract_tar(job, "AMPSCZ-Preproc")

    if dependency:
        job.set_dependency(f"afterok:${dependency}")
    if cmd is None:
        cmd = get_full_cmd()
    if not "--run_num" in cmd and array is not None:
        cmd += " --run_num $SLURM_ARRAY_TASK_ID"

    job.sbatch(f"srun python {cmd}")


def submit_scratch(
    model: str,
    array: Sequence[int] | int | None = None,
    cmd: str | None = None,
):
    """Submit base train job on SLURM cluster

    Args:
        model (str): Model to use
        array (Sequence[int] | int | None, optional): Can be single id, sequence, range or nothing.
            Defaults to None.
        cmd (str | None, optional): command to run, if None, retrieve the parameters
            used from the CLI. Defaults to None.
        dataset (str, optional): dataset to run the training process on
            (used for job name and output). Defaults to "".
    """
    job = create_job(
        get_name("base", model, array),
        array,
        get_output("base", model, array),
        n_cpus=20,
        n_gpus=1,
        mem="100G",
        time="4:00:00",
    )
    cpy_extract_tar(job, "AMPSCZ-Preproc")

    if cmd is None:
        cmd = get_full_cmd()
    else:
        if not "--run_num" in cmd and array is not None:
            cmd += " --run_num $SLURM_ARRAY_TASK_ID"

    job.sbatch(f"srun python {cmd}")
    print(job)


def submit_generate_ds():
    """Submit job to generate synthetic motion dataset on SLURM cluster"""
    job = create_job(
        "generate-dataset",
        None,
        f"./logs/generate-dataset.{Slurm.JOB_ARRAY_MASTER_ID}.out",
        n_cpus=64,
        n_gpus=0,
        mem="200G",
        time="50:00:00",
    )
    cpy_extract_tar(job, ["HCPEP-Preproc", "AMPSCZ-Preproc"])

    job.sbatch(f"srun python {get_full_cmd()}")


def submit_tune_transfer(model: str, cmd: str = None):
    job = Slurm(
        job_name=f"tune_transfer_{model}",
        nodes=1,
        gres="gpu:3",
        cpus_per_task=48,
        ntasks_per_node=1,
        ntasks=1,
        mem="400G",
        time="3:00:00",
        account=config.DEFAULT_SLURM_ACCOUNT,
        output=f"./logs/tune_transfer.{Slurm.JOB_ARRAY_MASTER_ID}.out",
    )
    setup_python(job)
    cpy_extract_tar(job, ["AMPSCZ-Preproc"])

    if not cmd:
        cmd = get_full_cmd()

    job.sbatch(f"srun python {cmd}")


def submit_tune_scratch(model: str, cmd: str = None):
    job = Slurm(
        job_name=f"tune_scratch_{model}",
        nodes=1,
        gres="gpu:4",
        cpus_per_task=48,
        ntasks_per_node=1,
        ntasks=1,
        mem="498G",
        time="16:00:00",
        account=config.DEFAULT_SLURM_ACCOUNT,
        output=f"./logs/tune_scratch_.{Slurm.JOB_ARRAY_MASTER_ID}.out",
    )
    setup_python(job)
    cpy_extract_tar(job, ["AMPSCZ-Preproc"])

    if not cmd:
        cmd = get_full_cmd()

    job.sbatch(f"python {cmd}")
