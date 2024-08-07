"""
Module to launch different standard command through slurm jobs
"""

import logging
import sys
import re
from collections.abc import Sequence
from simple_slurm import Slurm

from src.config import DEFAULT_SLURM_ACCOUNT


def setup_python(job: Slurm):
    """Add command to a slurm job to activate necessary modules and environments


    Args:
        job (Slurm): slurm job to modify
    """
    job.add_cmd("module load python cuda httpproxy")
    job.add_cmd("source ~/bowl/bin/activate")
    job.add_cmd('echo "python is setup"')
    # job.add_cmd("export NCCL_DEBUG=INFO")


def get_full_cmd() -> str:
    """Reconstruct the command use to launch the current program
    It is needed to launch the cli from inside the slurm job with the arguments provided
      when using the --slurm flag.

    Returns:
        str: the command arguments
    """
    full_command = " ".join(sys.argv)
    full_command = full_command.replace("-S", "").replace("--slurm", "")
    if full_command.find("--run_num"):
        full_command = re.sub(
            r"(--run_num)\s?[^\s-]*", r"\1 $SLURM_ARRAY_TASK_ID", full_command
        )

    return full_command


def get_finetune_cmd_from_pretrain(cmd: str) -> str:
    """Create the corresponding finetune command from a pretrain command
    Used to queue finetune job after pretrain

    Args:
        cmd (str): the pretrain command to modify

    Returns:
        str: corresponding finetune command
    """
    full_command = cmd.replace("pretrain", "finetune")
    if full_command.find("--dropout"):
        full_command = re.sub(r"(--dropout)\s?[^\s-]*", "", full_command)
    return full_command


def get_output(prefix: str, model: str, array: Sequence[int] | int | None) -> str:
    """Return the output log path for a job

    Args:
        prefix (str): job prefix (ex.: "pretrain", "finetune",...)
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
        prefix (str): job prefix (ex.: "pretrain", "finetune",...)
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
        array=array,
        nodes=1,
        cpus_per_task=n_cpus,
        gpus_per_node=n_gpus,
        ntasks_per_node=n_gpus,
        mem=mem,
        time=time,
        account=account,
        signal="SIGUSR1@90",
        requeue=True,
        output=output,
    )
    setup_python(job)
    return job


def submit_pretrain(
    model: str,
    array: Sequence[int] | int | None = None,
    cmd: str | None = None,
    send_finetune: bool = False,
):
    """Submit pretrain job on SLURM cluster

    Args:
        model (str): Model to use
        array (Sequence[int] | int | None, optional): Can be single id, sequence, range or nothing.
            Defaults to None.
        cmd (str | None, optional): command to run, if None, retrieve the parameters
            used from the CLI. Defaults to None.
        send_finetune (bool, optional): flag to send finetune command. Defaults to False.
    """
    job = create_job(
        get_name("pretrain", model, array),
        array,
        get_output("pretrain", model, array),
        n_cpus=10,
        n_gpus=2,
    )
    if cmd is None:
        cmd = get_full_cmd()
    else:
        if not "--run_num" in cmd and array is not None:
            cmd += " --run_num $SLURM_ARRAY_TASK_ID"

    job_id = job.sbatch(f"srun python3 {cmd}")
    print(job)

    if send_finetune:
        finetune_cmd = get_finetune_cmd_from_pretrain(cmd)
        for dataset in ["MRART", "AMPSCZ"]:
            submit_finetune(
                model,
                cmd=finetune_cmd + f" --dataset ${dataset}",
                dependency=job_id,
                dataset=dataset,
            )


def submit_finetune(
    model: str,
    array: Sequence[int] | int | None = None,
    cmd: str | None = None,
    dependency: str | None = None,
    dataset: str = "",
):
    """Submit finetune job on SLURM cluster

    Args:
        model (str): Model to use
        array (Sequence[int] | int | None, optional): Can be single id, sequence, range or nothing.
            Defaults to None.
        cmd (str | None, optional): command to run, if None, retrieve the parameters
            used from the CLI. Defaults to None.
        dependency (str | None, optional): optionnal job_id dependency to wait for.
            Defaults to None.
        dataset (str, optional): dataset to run the finetune process on
            (used for job name and output). Defaults to "".
    """
    job = create_job(
        get_name("finetune", model, array),
        array,
        get_output("finetune", model, array) + f"_{dataset}",
        n_cpus=20,
        n_gpus=1,
        mem="100G",
        time="5:00:00",
    )
    if dependency:
        job.set_dependency(f"afterok:${dependency}")
    if cmd is None:
        cmd = get_full_cmd()
    else:
        if not "--run_num" in cmd and array is not None:
            cmd += " --run_num $SLURM_ARRAY_TASK_ID"

    job.sbatch(f"srun python {cmd}")


def submit_scratch(
    model: str, array: Sequence[int] | int | None = None, cmd: str | None = None
):
    """Submit base train job on SLURM cluster

    Args:
        model (str): Model to use
        array (Sequence[int] | int | None, optional): Can be single id, sequence, range or nothing.
            Defaults to None.
        cmd (str | None, optional): command to run, if None, retrieve the parameters
            used from the CLI. Defaults to None.
    """
    job = create_job(
        get_name("base", model, array),
        array,
        get_output("base", model, array),
        n_cpus=20,
        n_gpus=1,
        mem="100G",
        time="5:00:00",
    )
    if cmd is None:
        cmd = get_full_cmd()
    job.sbatch(f"srun python {cmd}")


def submit_generate_ds():
    """Submit job to generate synthetic motion dataset on SLURM cluster"""
    job = create_job(
        "generate-dataset",
        None,
        f"generate-dataset.{Slurm.JOB_ARRAY_MASTER_ID}.out",
        n_cpus=40,
        n_gpus=0,
        mem="100G",
        time="10:00:00",
    )
    job.sbatch(f"srun python {get_full_cmd()}")
