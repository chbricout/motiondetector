## %
from notebooks.int_utils import (
    min_max_norm,
    retrieve_pretrain,
    retrieve_thresholds,
    retrieve_transfer,
)

# %% [markdown]
# # How to determine if a pretrained model will behave well based on pretrain metrics

# %%
motion, ssim, binary = retrieve_pretrain()
thresh_df = retrieve_thresholds()
transfer_df = retrieve_transfer()

# %%
motion["metric"] = min_max_norm(motion["r2"])
ssim["metric"] = min_max_norm(ssim["r2"])
binary["metric"] = min_max_norm(binary["balanced_accuracy"])
# %%

