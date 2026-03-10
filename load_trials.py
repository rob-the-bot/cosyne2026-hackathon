# %% This is to answer q4: During which trial segment is variable C best decoded?
# Trial start -> Stim start
# Stim start -> Outcome
# Outcome -> Trial end
# Not enough data / no differences

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

# your path to the dataset
dataset_folder = Path("data")

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


# %%


def load_mouse_data(mouse_id: int, dataset_folder: Path, verbose: bool = True):
    """
    Load spikes, cluster IDs, and brain areas for a given mouse folder.

    Parameters
    ----------
    mouse_id : int
        Identifier of the mouse (folder name).
    dataset_folder : Path
        Root dataset folder.
    verbose : bool
        If True, print dataset shapes/info.

    Returns
    -------
    dict
        key is cluster_id, value is { 'spikes': [...], 'brain_area': int}
    """
    mouse_folder = dataset_folder / str(mouse_id)
    if verbose:
        print(f"Loading mouse {mouse_id} from {mouse_folder}")

    # file paths
    brain_areas_file = mouse_folder / "brain_area.npy"
    cluster_ids_file = mouse_folder / "clusters.npy"
    spikes_file = mouse_folder / "spikes.npy"

    # load arrays
    brain_areas = np.load(brain_areas_file, allow_pickle=True).item()
    cluster_ids = np.load(cluster_ids_file)
    spikes = np.load(spikes_file)

    if verbose:
        print(f"brain_areas keys: {list(brain_areas.keys())}")
        print(f"brain_area cluster_id shape: {brain_areas['cluster_id'].shape}")
        print(f"brain_area names shape: {brain_areas['brain_area'].shape}")
        print(
            f"cluster_ids shape: {cluster_ids.shape}, "
            f"unique clusters: {len(np.unique(cluster_ids))}"
        )
        print(f"spikes shape: {spikes.shape}")

    # build brain area map
    brain_area_map = dict(zip(brain_areas["cluster_id"], brain_areas["brain_area"]))

    # organize spikes per cluster
    cluster_data = {}
    for cid, spike in zip(cluster_ids, spikes):
        cluster_data.setdefault(cid, {"spikes": [], "brain_area": brain_area_map[cid]})
        cluster_data[cid]["spikes"].append(spike)

    return cluster_data


def spikes_to_firing_rate_matrix(cluster_data, bin_width=0.1, duration=None):
    """
    Converts a dictionary of spike trains into a neuron x time matrix of firing rates.

    Parameters
    ----------
    cluster_data : dict
        key is cluster_id, value is { 'spikes': [...], 'brain_area': int}. Spike times are in seconds.
    bin_width : float, optional
        Width of the bin for spike counting (in seconds). Default is 0.1 (100 ms).
    duration : float, optional
        Total duration of session in seconds. If None, uses the maximum spike time.

    Returns
    -------
    rates_matrix : np.ndarray
        2D array (neurons x timebins) with firing rates (Hz).
    cluster_ids : list
        List of cluster_ids, row order matches rates_matrix.
    bin_edges : np.ndarray
        Array of bin edges in seconds.
    """

    cluster_ids = list(cluster_data.keys())
    # Concatenate all spike times to determine the max time if duration is not provided
    if duration is None:
        all_spike_times = np.concatenate(
            [np.array(cluster_data[cid]["spikes"]) for cid in cluster_ids]
        )
        duration = all_spike_times.max() if len(all_spike_times) > 0 else 0

    # Compute bin edges
    bin_edges = np.arange(0, duration + bin_width, bin_width)
    n_bins = len(bin_edges) - 1
    n_neurons = len(cluster_ids)

    rates_matrix = np.zeros((n_neurons, n_bins), dtype=float)

    for i, cid in enumerate(cluster_ids):
        spike_times = np.array(cluster_data[cid]["spikes"])
        counts, _ = np.histogram(spike_times, bins=bin_edges)
        rates_matrix[i, :] = counts / bin_width  # Firing rate in Hz

    return rates_matrix, cluster_ids, bin_edges


def build_sklearn_trial_dataset(
    cluster_data,
    trials_df,
    mouse_id=None,
    bin_width=0.1,
    spike_feature="count",
):
    """
    Convert raw spikes and trial event times into section-wise sklearn datasets.

    Returns one dataset per trial section:
    ``pre_stim``, ``stim_to_outcome`` and ``outcome_to_end``.
    Within each section, spike times are re-aligned so that binning starts at
    section onset, using non-overlapping 100 ms bins by default.
    Each output row is one time bin, and all bins from the same trial share the
    same entry in ``groups``.

    Parameters
    ----------
    cluster_data : dict
        Output of ``load_mouse_data`` for a single mouse.
    trials_df : pd.DataFrame
        Trial table containing ``variable_C``, ``trial_start``, ``stim_start``,
        ``outcome`` and ``trial_end``. If it also contains ``mouse_id``, the
        optional ``mouse_id`` argument can be used to select one mouse.
    mouse_id : int, optional
        Mouse identifier used to filter ``trials_df`` when a ``mouse_id`` column
        is present.
    bin_width : float, optional
        Width of the time bins in seconds. Default is 0.1 (100 ms).
    spike_feature : {"count", "rate"}, optional
        Whether to use raw spike counts or firing rates (Hz) within each bin.

    Returns
    -------
    dict
        Mapping each section name to a dictionary with keys:
        ``X``, ``y``, ``groups``, ``feature_names``, ``bin_width``,
        ``n_trials``, ``trial_info`` and ``sample_info``.
    """
    required_columns = [
        "variable_C",
        "trial_start",
        "stim_start",
        "outcome",
        "trial_end",
    ]
    missing_columns = [col for col in required_columns if col not in trials_df.columns]
    if missing_columns:
        raise ValueError(f"trials_df is missing required columns: {missing_columns}")

    trial_table = trials_df.copy()
    if "mouse_id" in trial_table.columns:
        if mouse_id is not None:
            trial_table = trial_table.query("mouse_id == @mouse_id")
        elif trial_table["mouse_id"].nunique() > 1:
            raise ValueError(
                "trials_df contains multiple mice. Pass mouse_id or pre-filter the DataFrame."
            )

    if spike_feature not in {"count", "rate"}:
        raise ValueError("spike_feature must be either 'count' or 'rate'")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    cluster_ids = sorted(cluster_data)
    section_names = ["pre_stim", "stim_to_outcome", "outcome_to_end"]

    spike_lookup = {
        cid: np.asarray(cluster_data[cid]["spikes"], dtype=float) for cid in cluster_ids
    }

    valid_trials = []
    section_durations = {section_name: [] for section_name in section_names}
    for row_idx, trial in trial_table.reset_index().rename(
        columns={"index": "trial_idx"}
    ).iterrows():
        trial_edges = np.array(
            [
                trial["trial_start"],
                trial["stim_start"],
                trial["outcome"],
                trial["trial_end"],
            ],
            dtype=float,
        )

        if not np.all(np.diff(trial_edges) >= 0):
            warnings.warn(
                f"Trial {row_idx} has non-monotonic timestamps: {trial_edges.tolist()}. Skipping this trial.",
                UserWarning,
            )
            continue

        valid_trials.append(trial.copy())
        for section_name, duration in zip(section_names, np.diff(trial_edges)):
            section_durations[section_name].append(float(duration))

    valid_trials = pd.DataFrame(valid_trials).reset_index(drop=True)
    if valid_trials.empty:
        empty_trial_info = pd.DataFrame(columns=["trial_idx"] + list(trial_table.columns))
        empty_sample_info = pd.DataFrame(
            columns=["trial_idx", "section", "bin_idx", "bin_start", "bin_end", "variable_C"]
        )
        return {
            section_name: {
                "X": np.empty((0, 0), dtype=float),
                "y": np.empty((0,), dtype=int),
                "groups": np.empty((0,), dtype=int),
                "feature_names": [],
                "bin_width": bin_width,
                "n_trials": 0,
                "trial_info": empty_trial_info.copy(),
                "sample_info": empty_sample_info.copy(),
            }
            for section_name in section_names
        }

    section_datasets = {}
    for section_idx, section_name in enumerate(section_names):
        X_rows = []
        y_rows = []
        group_rows = []
        sample_rows = []
        feature_names = [f"cluster_{cid}" for cid in cluster_ids]

        for trial_idx, trial in valid_trials.reset_index(drop=True).iterrows():
            section_start = float(trial[required_columns[1 + section_idx]])
            section_end = float(trial[required_columns[2 + section_idx]])
            section_duration = section_end - section_start
            n_bins = int(np.ceil(section_duration / bin_width)) if section_duration > 0 else 0
            if n_bins == 0:
                continue

            trial_matrix = np.zeros((n_bins, len(cluster_ids)), dtype=float)
            bin_starts = section_start + np.arange(n_bins, dtype=float) * bin_width
            bin_ends = np.minimum(bin_starts + bin_width, section_end)

            for cluster_offset, cid in enumerate(cluster_ids):
                spike_times = spike_lookup[cid]
                in_section = (spike_times >= section_start) & (spike_times < section_end)
                relative_spikes = spike_times[in_section] - section_start
                bin_counts = np.bincount(
                    np.floor(relative_spikes / bin_width).astype(int),
                    minlength=n_bins,
                )[:n_bins].astype(float)

                if spike_feature == "rate":
                    bin_durations = np.full(n_bins, bin_width, dtype=float)
                    last_bin_duration = section_duration - bin_width * (n_bins - 1)
                    bin_durations[-1] = (
                        last_bin_duration if last_bin_duration > 0 else bin_width
                    )
                    bin_counts = bin_counts / bin_durations

                trial_matrix[:, cluster_offset] = bin_counts

            X_rows.append(trial_matrix)
            y_rows.extend([int(trial["variable_C"])] * n_bins)
            group_rows.extend([int(trial["trial_idx"])] * n_bins)
            sample_rows.extend(
                {
                    "trial_idx": int(trial["trial_idx"]),
                    "section": section_name,
                    "bin_idx": bin_idx,
                    "bin_start": float(bin_start),
                    "bin_end": float(bin_end),
                    "variable_C": int(trial["variable_C"]),
                }
                for bin_idx, (bin_start, bin_end) in enumerate(zip(bin_starts, bin_ends))
            )

        X = (
            np.vstack(X_rows)
            if X_rows
            else np.empty((0, len(cluster_ids)), dtype=float)
        )
        y = np.asarray(y_rows, dtype=int)
        groups = np.asarray(group_rows, dtype=int)
        sample_info = pd.DataFrame(sample_rows)

        section_datasets[section_name] = {
            "X": X,
            "y": y,
            "groups": groups,
            "feature_names": feature_names,
            "bin_width": bin_width,
            "n_trials": len(valid_trials),
            "trial_info": valid_trials.reset_index(drop=True).copy(),
            "sample_info": sample_info,
        }

    return section_datasets


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


# %% scorer for LeaveOneGroupOut

def make_leave_one_group_majority_scorer(random_state=None):
    """
    Build a scorer for LeaveOneGroupOut that converts bin-wise predictions into
    one trial-level prediction by majority vote.

    This scorer assumes each test fold contains exactly one group, which is true
    for ``LeaveOneGroupOut``. If there is a tie for the most frequent predicted
    label, one of the tied labels is chosen at random.
    """
    rng = np.random.default_rng(random_state)

    def scorer(estimator, X, y_true):
        y_pred = estimator.predict(X)
        unique_true = np.unique(y_true)
        if len(unique_true) != 1:
            raise ValueError(
                "This scorer expects exactly one true label in the test fold. "
                "Use it with LeaveOneGroupOut where each held-out group belongs "
                "to a single trial."
            )

        predicted_labels, counts = np.unique(y_pred, return_counts=True)
        winners = predicted_labels[counts == counts.max()]
        final_pred = rng.choice(winners) if len(winners) > 1 else winners[0]
        return float(final_pred == unique_true[0])

    return scorer


# %% standard decoding

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from tqdm.auto import tqdm


pipeline = make_pipeline(StandardScaler(), LDA(priors=[0.5, 0.5]))
trial_level_scorer = make_leave_one_group_majority_scorer(random_state=0)

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
        cv=LeaveOneGroupOut(),
        scoring=trial_level_scorer,
        n_jobs=-1,
    )
    print(f"{phase}: {scores.mean()}")



# %% load one example mouse

# cluster_data = load_mouse_data(mouse_id, dataset_folder, verbose=False)
spikes_file = dataset_folder / str(mouse_id) / "spikes.npy"
spikes = np.load(spikes_file)
print(spikes.shape)

# brain_area
brain_area_file = dataset_folder / str(mouse_id) / "brain_area.npy"
brain_area = np.load(brain_area_file, allow_pickle=True).item()
print(brain_area.keys())


# %% cluster_ids

cluster_ids_file = dataset_folder / str(mouse_id) / "clusters.npy"
cluster_ids = np.load(cluster_ids_file)
print(cluster_ids.shape)
