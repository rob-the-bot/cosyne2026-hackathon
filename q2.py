# %% This is to answer q2: In which brain area are pairwise spike train interactions strongest at the 100 ms timescale?

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm, trange
from utils import build_spontaneous_activity_matrix, load_mouse_data


# your path to the dataset
dataset_folder = Path("data")


# %% Loading trials data

# collect all trial_data.csv into one DataFrame
all_trials = []
columns = ["trial_start", "stim_start", "outcome", "trial_end"]

for ii in range(1, 19):
    trialdata_filename = dataset_folder / str(ii) / "trial_data.csv"
    df_mouse = pd.read_csv(trialdata_filename, usecols=columns)
    df_mouse["mouse_id"] = ii
    # put `mouse_id` as the first column
    df_mouse = df_mouse[["mouse_id"] + columns]
    all_trials.append(df_mouse)

trials_df = pd.concat(all_trials, ignore_index=True)
# trails_df.to_csv(dataset_folder / 'output/trials_info.csv')


# %% get the start time and end time of each animal

animal_start_end = trials_df.groupby("mouse_id").agg(
    {"trial_start": "min", "trial_end": "max"}
)
animal_start_end


# %% spontaneous activity outside the trial period

example_mouse_id = 1
example_mouse_data = load_mouse_data(example_mouse_id, dataset_folder, verbose=False)
spontaneous_matrix, cluster_ids, spontaneous_bin_edges, interval_info = (
    build_spontaneous_activity_matrix(
        example_mouse_data,
        trials_df,
        mouse_id=example_mouse_id,
        bin_width=0.1,
        spike_feature="count",
        concatenate=True,
    )
)
print(spontaneous_matrix.shape)
print(len(cluster_ids))
print(spontaneous_bin_edges[:10])
print(interval_info)


# group spontaneous activity by brain area

cluster_area_info = pd.DataFrame(
    {
        "cluster_id": cluster_ids,
        "brain_area": [example_mouse_data[cid]["brain_area"] for cid in cluster_ids],
    }
)
cluster_area_info["row_idx"] = np.arange(len(cluster_ids))

spontaneous_by_area = {
    brain_area: spontaneous_matrix[group["row_idx"].to_numpy(), :]
    for brain_area, group in cluster_area_info.groupby("brain_area")
}

for brain_area, area_matrix in spontaneous_by_area.items():
    print(brain_area, area_matrix.shape)
    corr = np.corrcoef(area_matrix)
    # fill diagonal with nan
    np.fill_diagonal(corr, np.nan)
    print(np.nanmean(corr))


# %% loop through all mice

res = []
for mouse_id in trange(1, 19):
    mouse_data = load_mouse_data(mouse_id, dataset_folder, verbose=False)

    spontaneous_matrix, cluster_ids, spontaneous_bin_edges, interval_info = (
        build_spontaneous_activity_matrix(
            mouse_data,
            trials_df,
            mouse_id=mouse_id,
            bin_width=0.1,
            spike_feature="count",
            concatenate=True,
        )
    )

    # group spontaneous activity by brain area
    cluster_area_info = pd.DataFrame(
        {
            "cluster_id": cluster_ids,
            "brain_area": [mouse_data[cid]["brain_area"] for cid in cluster_ids],
        }
    )
    cluster_area_info["row_idx"] = np.arange(len(cluster_ids))

    spontaneous_by_area = {
        brain_area: spontaneous_matrix[group["row_idx"].to_numpy(), :]
        for brain_area, group in cluster_area_info.groupby("brain_area")
    }

    for brain_area, area_matrix in spontaneous_by_area.items():
        # remove rows with all zeros
        area_matrix = area_matrix[~np.all(area_matrix == 0, axis=1)]
        if area_matrix.shape[0] == 0:
            continue
        corr = np.corrcoef(area_matrix)
        # fill diagonal with nan
        np.fill_diagonal(corr, np.nan)
        res.append((mouse_id, brain_area, np.nanmean(corr)))

df_res = pd.DataFrame(res, columns=["mouse_id", "brain_area", "corr"])


# %% plot the results

fig, ax = plt.subplots(figsize=(2, 2.5), layout="constrained")
sns.barplot(x="brain_area", y="corr", data=df_res, fill=None, color="black", errorbar=None)
sns.stripplot(x="brain_area", y="corr", data=df_res, color="black", size=3, alpha=0.5)
sns.despine()
[x.patch.set_visible(False) for x in (fig, ax)]
ax.set(xlabel="Brain Area", ylabel="Mean Correlation")
fig.savefig("q2.svg", transparent=True)
