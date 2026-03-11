# %% This is to answer q1: Which brain area (if any) has the highest density of ripples (i.e. “hippocampal” ripples traditionally occurring during sharp wave-ripples)?

from joblib import Parallel, delayed
from pathlib import Path
import gc

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import trange
import pandas as pd
from ripple_detection import Kay_ripple_detector

# your path to the dataset
dataset_folder = Path('data')
fs = 500  # Hz, sampling rate

# %% Your data

def process_mouse(mouse_id):
    lfp_folder = dataset_folder / str(mouse_id)
    lfps = [np.load(lfp_folder / f"lfp_{i}.npy") for i in range(1, 4)]
    time = np.arange(lfps[0].shape[1]) / fs
    speed = np.zeros(len(time))  # Animal speed

    res = []
    for lf_idx, lfp in enumerate(lfps):
        ripple_times = Kay_ripple_detector(
            time, lfp.T, speed, fs,
            minimum_duration=0.015,     # seconds
            zscore_threshold=2.0
        )
        res.append((mouse_id, lf_idx + 1, len(ripple_times)))
    
    return pd.DataFrame(res, columns=["mouse_id", "brain_area", "#ripples"])

df_list = []

for mouse_id in trange(1, 19):
    df_list.append(process_mouse(mouse_id))

res = pd.concat(df_list, ignore_index=True)

# %% plot the results

fig, ax = plt.subplots(figsize=(2, 2), layout="constrained")
sns.barplot(x="brain_area", y="#ripples", data=res, ax=ax, fill=None, color="black", errorbar=None)
sns.stripplot(x="brain_area", y="#ripples", data=res, color="black", size=3, alpha=0.5, ax=ax)
sns.despine()
[x.patch.set_visible(False) for x in (fig, ax)]
ax.set(xlabel="Brain Area", ylabel="Number of Ripples")
fig.savefig("q1.svg", transparent=True)


# %% kruskal-wallis test

from scipy.stats import kruskal

kruskal(*res.groupby("brain_area")["#ripples"].apply(list))
