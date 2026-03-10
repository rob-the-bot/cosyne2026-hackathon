# %% This is to answer q4: During which trial segment is variable C best decoded?
# Trial start -> Stim start
# Stim start -> Outcome
# Outcome -> Trial end
# Not enough data / no differences

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm, trange

from utils import (
    build_sklearn_trial_dataset,
    load_mouse_data,
    make_leave_one_group_majority_scorer,
    spikes_to_firing_rate_matrix,
)

# your path to the dataset
dataset_folder = Path("data")

cv = LeaveOneGroupOut()
clf = LDA(solver="eigen", shrinkage="auto", priors=[0.5, 0.5])
# use logistic regression instead
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty="l2", class_weight="balanced")

pipeline = make_pipeline(StandardScaler(), clf)
trial_level_scorer = make_leave_one_group_majority_scorer(random_state=0)

# %% Loading trials data

# collect all trial_data.csv into one DataFrame
all_trials = []
columns = ["variable_C", "trial_start", "stim_start", "outcome", "trial_end"]

for ii in range(1, 19):
    trialdata_filename = dataset_folder / str(ii) / "trial_data.csv"
    df_mouse = pd.read_csv(trialdata_filename, usecols=columns)
    df_mouse["mouse_id"] = ii
    # put `mouse_id` as the first column
    df_mouse = df_mouse[["mouse_id"] + columns]
    all_trials.append(df_mouse)

trials_df = pd.concat(all_trials, ignore_index=True)
# trails_df.to_csv(dataset_folder / 'output/trials_info.csv')

trials_df


# %% the dataset is quite unbalanced, especially class `1` in `variable_C`

for mouse_id, group in trials_df.groupby("mouse_id"):
    unique_counts = group["variable_C"].value_counts()
    print(f"Mouse ID: {mouse_id}")
    print("Unique elements and their counts:")
    print(unique_counts)
    print("-" * 40)


# %% drop all rows where `variable_C` is `1`

trials_df = trials_df.query("variable_C != 1")


# %% convert spikes to firing rate matrix

mouse_id: int = 1
cluster_data = load_mouse_data(mouse_id, dataset_folder, verbose=False)
rates_matrix, cluster_ids, bin_edges = spikes_to_firing_rate_matrix(cluster_data)
print(rates_matrix.shape)
print(cluster_ids)
print(bin_edges)


# %% build sklearn trial dataset

section_datasets = build_sklearn_trial_dataset(
    cluster_data,
    trials_df,
    mouse_id=mouse_id,
    bin_width=0.1,
    spike_feature="rate",
)
for section_name, section_data in section_datasets.items():
    print(section_name, section_data["X"].shape, section_data["y"].shape)
    print(section_data["groups"][:8])
    print(section_data["feature_names"][:8])


# %% standard decoding

# loop through "pre_stim", "stim_to_outcome", "outcome_to_end"

for phase in ["pre_stim", "stim_to_outcome", "outcome_to_end"]:
    X = section_datasets[phase]["X"]
    y = section_datasets[phase]["y"]
    groups = section_datasets[phase]["groups"]
    scores = cross_val_score(
        pipeline,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=trial_level_scorer,
        n_jobs=-1,
    )
    print(f"{phase}: {scores.mean()}")

# %% run with all mice

results = []
# Run decoding for all available mice and print results
for mouse_id in trange(1, 19):
    cluster_data = load_mouse_data(mouse_id, dataset_folder, verbose=False)

    section_datasets = build_sklearn_trial_dataset(
        cluster_data,
        trials_df,
        mouse_id=mouse_id,
        bin_width=0.1,
        spike_feature="rate",
    )

    mouse_scores = {"mouse_id": mouse_id}
    for phase in ["pre_stim", "stim_to_outcome", "outcome_to_end"]:
        X = section_datasets[phase]["X"]
        y = section_datasets[phase]["y"]
        groups = section_datasets[phase]["groups"]
        if X.shape[0] == 0:
            print(f"{phase}: No data.")
            mouse_scores[phase] = None
            continue
        scores = cross_val_score(
            pipeline,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=trial_level_scorer,
            n_jobs=-1,
        )
        mean_score = scores.mean()
        print(f"{phase}: {mean_score:.4f}")
        mouse_scores[phase] = mean_score
    results.append(mouse_scores)

df_results = pd.DataFrame(results)
df_results.to_csv("results.csv", index=False)
