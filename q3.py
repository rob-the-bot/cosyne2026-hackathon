# %% This is to answer q3: Which brain area pair has the strongest directed functional connectivity?

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm, trange
import nemos as nmo

from utils import build_spontaneous_activity_matrix, load_mouse_data

# your path to the dataset
dataset_folder = Path("data")

K = 3
L = 10
raised_cos = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=K, window_size=L, label="spike-history"
)
model = nmo.glm.PopulationGLM()
area_pairs = [(1, 2), (3, 2), (3, 1)]


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
spontaneous_blocks, cluster_ids, interval_info = build_spontaneous_activity_matrix(
    example_mouse_data,
    trials_df,
    mouse_id=example_mouse_id,
    bin_width=0.1,
    spike_feature="count",
    concatenate=False,
)

# group spontaneous activity by brain area

cluster_area_info = pd.DataFrame(
    {
        "cluster_id": cluster_ids,
        "brain_area": [example_mouse_data[cid]["brain_area"] for cid in cluster_ids],
    }
)
cluster_area_info["row_idx"] = np.arange(len(cluster_ids))

spontaneous_by_area = {
    brain_area: spontaneous_blocks["pre_trial"]["matrix"][group["row_idx"].to_numpy()]
    for brain_area, group in cluster_area_info.groupby("brain_area")
}

for brain_area, area_matrix in spontaneous_by_area.items():
    print(brain_area, area_matrix.shape)

# %%

counts = spontaneous_blocks["post_trial"]["matrix"].T
print(counts.shape)
X = raised_cos.compute_features(counts)

# %%

raised_cos = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=K, window_size=L, label="spike-history"
)
X = raised_cos.compute_features(counts)
model.fit(X, counts)

# sender x basis x receiver
Wb = raised_cos.split_by_feature(model.coef_, axis=0)["spike-history"]

# basis evaluated on a uniform grid across the window
grid, B = raised_cos.evaluate_on_grid(L)  # B shape: (L, K)

# sender x lag x receiver
Wlag = np.einsum("lb,sbr->slr", B, Wb)


# %% convert to a 2D neuron-by-neuron directed connectivity matrix

conn_matrix = Wlag.mean(axis=1)

vmax = np.percentile(np.abs(conn_matrix), 99.5)
plt.imshow(conn_matrix, cmap="PiYG", vmax=vmax, vmin=-vmax)
plt.colorbar(label="Connectivity Strength")

# %% find the brain area pair with the strongest directed functional connectivity

# get connectivity matrices for area 1 to 2, 3 to 2, and 3 to 1 using a for loop


for from_area, to_area in area_pairs:
    from_idx = cluster_area_info.query("brain_area == @from_area")["row_idx"].to_numpy()
    to_idx = cluster_area_info.query("brain_area == @to_area")["row_idx"].to_numpy()
    conn_submatrix = conn_matrix[from_idx][:, to_idx]
    print(f"{from_area} to {to_area}: {conn_submatrix.mean()}")


# %% loop through all mice and both pre and post trial periods

res = []
for mouse_id in trange(1, 19):

    mouse_data = load_mouse_data(
        mouse_id, dataset_folder, verbose=False
    )
    spontaneous_blocks, cluster_ids, interval_info = build_spontaneous_activity_matrix(
        mouse_data,
        trials_df,
        mouse_id=mouse_id,
        bin_width=0.1,
        spike_feature="count",
        concatenate=False,
    )

    # group spontaneous activity by brain area
    cluster_area_info = pd.DataFrame(
        {
            "cluster_id": cluster_ids,
            "brain_area": [
                mouse_data[cid]["brain_area"] for cid in cluster_ids
            ],
        }
    )
    cluster_area_info["row_idx"] = np.arange(len(cluster_ids))

    conn_matrix = {}
    for period in ["pre_trial", "post_trial"]:
        counts = spontaneous_blocks[period]["matrix"].T
        X = raised_cos.compute_features(counts)
        model.fit(X, counts)
        # sender x basis x receiver
        Wb = raised_cos.split_by_feature(model.coef_, axis=0)["spike-history"]
        # basis evaluated on a uniform grid across the window
        grid, B = raised_cos.evaluate_on_grid(L)  # B shape: (L, K)
        # sender x lag x receiver
        Wlag = np.einsum("lb,sbr->slr", B, Wb)
        conn_matrix[period] = Wlag.mean(axis=1)

    # weighted average of the two periods
    conn_matrix = np.average(
        list(conn_matrix.values()),
        axis=0,
        weights=[
            spontaneous_blocks["pre_trial"]["matrix"].sum(),
            spontaneous_blocks["post_trial"]["matrix"].sum(),
        ],
    )

    for from_area, to_area in area_pairs:
        from_idx = cluster_area_info.query("brain_area == @from_area")[
            "row_idx"
        ].to_numpy()
        to_idx = cluster_area_info.query("brain_area == @to_area")["row_idx"].to_numpy()
        conn_submatrix = conn_matrix[from_idx][:, to_idx]
        res.append((mouse_id, f"{from_area}->{to_area}", conn_submatrix.mean()))

# %%

df = pd.DataFrame(res, columns=["mouse_id", "area_pair", "conn_strength"])

fig, ax = plt.subplots(figsize=(2, 2.5), layout="constrained")
sns.boxplot(x="area_pair", y="conn_strength", fill=None, color="black", data=df, ax=ax)
# sns.stripplot(x="area_pair", y="conn_strength", color="black", size=3, alpha=0.5, data=df, ax=ax)
sns.despine()
[x.patch.set_visible(False) for x in (fig, ax)]
ax.set(xlabel="Brain Area Pair", ylabel="Mean Connectivity Strength")
fig.savefig("q3.svg", transparent=True)
