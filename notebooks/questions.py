# %%
%load_ext autoreload
%autoreload 2
import glob
import sys
from os import path 
from comet_ml import API
import pandas as pd
import matplotlib
import seaborn as sb
from matplotlib import pyplot as plt
sys.path.append("..")
sb.set_theme()
from src import config
from notebooks.int_utils import *

matplotlib.use('tkagg')
# % [markdown]
# # Are synthetic artifacts close to real artifacts

# %%
thresh_mrart_results = retrieve_thresholds()
print(
    thresh_mrart_results[["model", "task", "balanced_accuracy"]].to_latex(
        index=False, escape=True, float_format="%.2f"
    )
)

# %%
best_row = thresh_mrart_results.sort_values("balanced_accuracy", ascending=False).iloc[
    0
]
best_model = path.join(
    path_to_test,
    "pretraining",
    f"{best_row['model']}-{best_row['task']}-{best_row['run_num']}",
    "mrart.csv",
)
pred_mrart = pd.read_csv(best_model)
sb.boxplot(pred_mrart, x="pred", hue="label", dodge=True, palette=sb.color_palette())

# %% [markdown]
# # What metric should we use to pretrain on synthetic artifacts ?

# %%
selector = ["task", "balanced_accuracy"]
thresh_df = retrieve_thresholds()
thresh_df = thresh_df[selector].groupby(["task"], as_index=False).mean()
compute_norms(thresh_df)
thresh_df["origin"] = "threshold"

transfer_df = retrieve_transfer()
mean_transfer_df = (
    transfer_df[["model", "task", "run_num", "balanced_accuracy"]]
    .groupby(["model", "task"], as_index=False)
    .mean()
    .drop(columns="run_num")
)
mean_transfer_df = (
    mean_transfer_df.drop(columns="model").groupby(["task"], as_index=False).mean()
)
compute_norms(mean_transfer_df)
mean_transfer_df["origin"] = "transfer"


task_df = pd.concat([mean_transfer_df, thresh_df]).reset_index(drop=True)

# %%
mean_tasks = pd.concat([mean_transfer_df, thresh_df])
df = mean_tasks.drop(columns="origin").groupby("task", as_index=False).mean()
df["origin"] = "Mean"

full_res = pd.concat([mean_tasks, df])
full_res = full_res.set_index(["origin", "task"], drop=True, append=False)
print(full_res.to_latex(index=True, escape=True, float_format="%.3f", multirow=True))

# %% [markdown]
# # What model is the best fit for motion prediction task

# %%
selector = ["model", "balanced_accuracy"]

thresh_df = retrieve_thresholds()[selector]
thresh_df = thresh_df.groupby(["model"], as_index=False).mean()[selector]
thresh_df["setting"] = "Threshold"
compute_norms(thresh_df)

transfer_df = retrieve_transfer()[selector + ["task"]]
transfer_df = transfer_df.groupby(["model", "task"], as_index=False).mean()[selector]
transfer_df = transfer_df.groupby(["model"], as_index=False).mean()[selector]
transfer_df["setting"] = "Transfer"
compute_norms(transfer_df)

scratch_df = retrieve_scratch()[selector]
scratch_df = scratch_df.groupby(["model"], as_index=False).mean()[selector]
scratch_df["setting"] = "Scratch"
compute_norms(scratch_df)


model_df = pd.concat([thresh_df, transfer_df, scratch_df])
report_models = model_df.set_index(["setting", "model"], drop=True, append=False)
print(
    report_models.to_latex(index=True, escape=True, float_format="%.4f", multirow=True)
)

# %%
model_perf_df = model_df.drop(columns="setting").groupby("model").mean()
print(
    model_perf_df.to_latex(index=True, escape=True, float_format="%.4f", multirow=True)
)

# %% [markdown]
# # Should we consider using pretraining ?

# %% [markdown]
# ## Setting Comparison

# %%
selector = ["id", "setting", "model", "task", "balanced_accuracy"]
thresh_df = retrieve_thresholds()
thresh_df["setting"] = "Threshold"
thresh_df["id"] = "Threshold - " + thresh_df["model"] + " - " + thresh_df["task"]

transfer_df = retrieve_transfer()
transfer_df = transfer_df.groupby(["model", "task"], as_index=False).mean()
transfer_df["setting"] = "Transfer"
transfer_df["id"] = "Transfer - " + transfer_df["model"] + " - " + transfer_df["task"]

scratch_df = retrieve_scratch().drop(columns=["source"])
scratch_df = scratch_df.groupby(["model"], as_index=False).mean()
scratch_df["setting"] = "Scratch"
scratch_df["task"] = ""
scratch_df["id"] = "Scratch - " + scratch_df["model"]

full_res = pd.concat([thresh_df[selector], transfer_df[selector], scratch_df[selector]])
setting_mean = (
    full_res.drop(columns=["id", "model", "task"])
    .groupby("setting", as_index=False)
    .mean()
)
print(setting_mean.to_latex(index=False, escape=True, float_format="%.4f"))

plt.figure(figsize=(5, 6))
sb.boxplot(full_res, y="balanced_accuracy", x="setting")

# %%
select_res = full_res[full_res["model"] == "SFCN"]
select_res = select_res[(select_res["task"] == "MOTION") | (select_res["task"] == "")]
print(
    select_res[["id", "balanced_accuracy"]].to_latex(
        index=False, escape=True, float_format="%.4f"
    )
)


selector = ["id", "setting", "model", "task", "balanced_accuracy"]
thresh_df = retrieve_thresholds()
thresh_df = thresh_df[(thresh_df["task"] == "MOTION") & (thresh_df["model"] == "SFCN")]
thresh_df["setting"] = "Threshold"

transfer_df = retrieve_transfer()
transfer_df = transfer_df[
    (transfer_df["task"] == "MOTION") & (transfer_df["model"] == "SFCN")
]
transfer_df["setting"] = "Transfer"

scratch_df = retrieve_scratch().drop(columns=["source"])
scratch_df = scratch_df[scratch_df["model"] == "SFCN"]
scratch_df["setting"] = "Scratch"
scratch_df["task"] = ""

sfcn_motion = pd.concat([thresh_df, transfer_df, scratch_df])
plt.figure(figsize=(6, 3))
sb.boxplot(sfcn_motion, y="setting", x="balanced_accuracy")
plt.xlabel("Setting")
plt.title("Balanced accuracy of SFCN model on different setting")

# %% [markdown]
# ## Power Consumption Comparison

# %%
exps = api.get_experiments("mrart", "baseline-mrart")
scratch_ressources_df = get_compute_usage_df(exps)
scratch_ressources_df["task"] = ""
scratch_ressources_df["origin"] = "Scratch"

exps = api.get_experiments("mrart", "estimate-motion-pretrain", "transfer-MRART*")
transfer_ressources_df = get_compute_usage_df(exps, task_dependent=True)
transfer_ressources_df["origin"] = "Transfer"

# %%
%autoreload 2
from src.dataset.mrart.mrart_dataset import TrainMrArt
from src.transforms.load import FinetuneTransform
from src.utils.metrics import separation_capacity_train
import src.utils.task as task_utils
import src.training.eval as teval
from torch.utils.data import DataLoader
import timeit
import os
os.chdir("..")

def eval_thresh(cuda):
    module, task =task_utils.load_pretrain_from_ckpt(ckpt_path="models/pretrained/SFCN-MOTION-3.ckpt")
    dl = DataLoader(TrainMrArt.from_env(FinetuneTransform()))
    train=teval.get_pred_from_pretrain(module, dl, "train", cuda=cuda)
    x_train = train["pred"].to_numpy()
    y_train = train["label"].to_numpy()
    thresholds, _ = separation_capacity_train(
        X=x_train, y=y_train
    )

res_cpu =timeit.timeit(lambda: eval_thresh(False), number=1)
res_cuda =timeit.timeit(lambda: eval_thresh(True), number=1)

os.chdir("notebooks")
res_cpu, res_cuda

# %%
conv = lambda ms: ":".join(map(lambda x: str(int(x)).zfill(2), ms_to_time(ms)))
time_cpu = conv(res_cpu * 1000)
time_cuda = conv(res_cuda * 1000)
time_cpu, time_cuda

# %%
resources = pd.concat([scratch_ressources_df, transfer_ressources_df]).reset_index(
    drop=True
)
resources = (
    resources[["origin", "millis", "max_gpu_ram_used", "max_gpu_power_usage"]]
    .groupby(["origin"])
    .agg(["mean", "std"])
)

resources.loc["Reduced by (%)"] = (
    (resources.loc["Transfer"] - resources.loc["Scratch"])
    .div(resources.loc["Scratch"])
    .mul(100)
)

resources["duration", "mean"] = resources["millis", "mean"].apply(conv)
resources["duration", "std"] = resources["millis", "std"].apply(conv)
resources.loc["Reduced by (%)", "duration"] = resources.loc[
    "Reduced by (%)", "millis"
].to_list()
resources = resources.drop(columns="millis")

print(resources.to_latex(index=True, escape=True, float_format="%.2f"))

# %% [markdown]
# ## Unbalanced Dataset Comparison

# %%
selector = ["id", "setting", "model", "task", "balanced_accuracy"]
thresh_df = retrieve_thresholds(unbalanced=True)
thresh_df["setting"] = "Threshold"
thresh_df["id"] = "Threshold - " + thresh_df["model"] + " - " + thresh_df["task"]

transfer_df = retrieve_transfer(unbalanced=True)
transfer_df = transfer_df.groupby(["model", "task"], as_index=False).mean()
transfer_df["setting"] = "Transfer"
transfer_df["id"] = "Transfer - " + transfer_df["model"] + " - " + transfer_df["task"]

scratch_df = retrieve_scratch(unbalanced=True).drop(columns=["source"])
scratch_df = scratch_df.groupby(["model"], as_index=False).mean()
scratch_df["setting"] = "Scratch"
scratch_df["task"] = ""
scratch_df["id"] = "Scratch - " + scratch_df["model"]

full_res = pd.concat([thresh_df[selector], transfer_df[selector], scratch_df[selector]])
setting_mean = (
    full_res.drop(columns=["id", "model", "task"])
    .groupby("setting", as_index=False)
    .mean()
)
print(setting_mean.to_latex(index=False, escape=True, float_format="%.4f"))

plt.figure(figsize=(6, 3))
sb.boxplot(full_res, x="balanced_accuracy", y="setting")

# %%
select_res = full_res[full_res["model"] == "SFCN"]
select_res = select_res[(select_res["task"] == "MOTION") | (select_res["task"] == "")]
print(
    select_res[["id", "balanced_accuracy"]].to_latex(
        index=False, escape=True, float_format="%.4f"
    )
)


selector = ["id", "setting", "model", "task", "balanced_accuracy"]
thresh_df = retrieve_thresholds(unbalanced=True)
thresh_df = thresh_df[(thresh_df["task"] == "MOTION") & (thresh_df["model"] == "SFCN")]
thresh_df["setting"] = "Threshold"

transfer_df = retrieve_transfer(unbalanced=True)
transfer_df = transfer_df[
    (transfer_df["task"] == "MOTION") & (transfer_df["model"] == "SFCN")
]
transfer_df["setting"] = "Transfer"

scratch_df = retrieve_scratch(unbalanced=True).drop(columns=["source"])
scratch_df = scratch_df[scratch_df["model"] == "SFCN"]
scratch_df["setting"] = "Scratch"
scratch_df["task"] = ""

sfcn_motion = pd.concat([thresh_df, transfer_df, scratch_df])
plt.figure(figsize=(6, 3))
sb.boxplot(sfcn_motion, y="setting", x="balanced_accuracy")
plt.xlabel("Setting")
plt.title("Balanced accuracy of SFCN model on different setting")

# %% [markdown]
# ##  Accuracy Conservation

# %%
def get_full(unbalanced):
    selector = ["setting", "model", "task", "balanced_accuracy"]
    thresh_df = retrieve_thresholds(unbalanced=unbalanced)
    thresh_df["setting"] = "Threshold"

    transfer_df = retrieve_transfer(unbalanced=unbalanced)
    transfer_df = transfer_df.groupby(["model", "task"], as_index=False).mean()
    transfer_df["setting"] = "Transfer"

    scratch_df = retrieve_scratch(unbalanced=unbalanced).drop(columns=["source"])
    scratch_df = scratch_df.groupby(["model"], as_index=False).mean()
    scratch_df["setting"] = "Scratch"
    scratch_df["task"] = ""

    return pd.concat(
        [thresh_df[selector], transfer_df[selector], scratch_df[selector]]
    ).set_index(["setting", "model", "task"])


normal_df = get_full(False)
unbalanced_df = get_full(True)
loss = unbalanced_df.sub(normal_df)
normal_df["balanced_accuracy_unb"] = unbalanced_df["balanced_accuracy"]
normal_df["accuracy_loss"] = loss
normal_df["percent_loss"] = (
    normal_df["accuracy_loss"].div(normal_df["balanced_accuracy"]) * 100
)
acc_conv = (
    normal_df.reset_index().drop(columns=["model", "task"]).groupby("setting").mean()
)

print(acc_conv.to_latex(index=True, escape=True, float_format="%.2f"))
acc_conv


