import sys
import re
from collections.abc import Sequence
from simple_slurm import Slurm

from src.config import DEFAULT_SLURM_ACCOUNT


def setup_python(job: Slurm):
    job.add_cmd("module load python cuda httpproxy")
    job.add_cmd("source ~/bowl/bin/activate")
    job.add_cmd("echo \"python is setup\"")


def get_full_cmd():
    full_command = " ".join(sys.argv)
    full_command = full_command.replace("-S", "").replace("--slurm", "")
    if full_command.find("--run_num"):
        full_command = re.sub(
            r"(--run_num)\s?[^\s-]*", r"\1 $SLURM_ARRAY_TASK_ID", full_command
        )

    return full_command


def get_finetune_cmd_from_pretrain(cmd):
    full_command = cmd.replace("pretrain", "finetune")
    if full_command.find("--dropout"):
        full_command = re.sub(r"(--dropout)\s?[^\s-]*", "", full_command)
    return full_command


def get_output(prefix: str, model: str, array):
    base = f"./logs/{prefix}-{model}"
    if array != None:
        base += f"_{Slurm.JOB_ARRAY_ID}"
    base += f".{Slurm.JOB_ARRAY_MASTER_ID}.out"
    return base


def get_name(prefix: str, model: str, array):
    base = f"{prefix}_{model}"
    if array != None:
        base += f"_{Slurm.JOB_ARRAY_ID}"
    return base


def create_job(
    name,
    array,
    output,
    n_cpus,
    n_gpus,
    account=DEFAULT_SLURM_ACCOUNT,
    mem="200G",
    time="24:00:00",
):
    job = Slurm(
        job_name=name,
        array=array,
        nodes=1,
        cpus_per_task=n_cpus,
        gpus_per_node=n_gpus,
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
    array: Sequence[int] | int = None,
    cmd: str = None,
    send_finetune: bool = False,
):
    job = create_job(
        get_name("pretrain", model, array),
        array,
        get_output("pretrain", model, array),
        n_cpus=20,
        n_gpus=2,
    )
    if cmd is None:
        cmd = get_full_cmd()
    else:
        if not "--run_num" in cmd and array!=None:
            cmd += " --run_num $SLURM_ARRAY_TASK_ID"

    job_id =job.sbatch(f"srun python {cmd}")

    if send_finetune:
        finetune_cmd = get_finetune_cmd_from_pretrain(cmd)
        for dataset in ["MRART", "AMPSCZ"]:
            submit_finetune(model, cmd=finetune_cmd + f" --dataset ${dataset}", dependency=job_id, dataset=dataset)


def submit_finetune(
    model: str,
    array: Sequence[int] | int = None,
    cmd: str = None,
    dependency: str = None,
    dataset:str=""
):
    job = create_job(
        get_name("finetune", model, array),
        array,
        get_output("finetune", model, array)+f"_{dataset}",
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
        if not "--run_num" in cmd and array!=None:
            cmd += " --run_num $SLURM_ARRAY_TASK_ID"

    job.sbatch(f"srun python {cmd}")


def submit_scratch(model: str, array: Sequence[int] | int = None, cmd: str = None):
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
