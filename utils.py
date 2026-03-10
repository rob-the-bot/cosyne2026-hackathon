import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def load_mouse_data(mouse_id: int, dataset_folder: Path, verbose: bool = True):
    """
    Load spikes, cluster IDs, and brain areas for a given mouse folder.
    Taken from https://pre-cosyne-brainhack.github.io/hackathon2026/assets/downloads/code-snippets.zip

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


def build_spontaneous_activity_matrix(
    cluster_data,
    trials_df,
    mouse_id=None,
    bin_width=0.1,
    spike_feature="count",
    concatenate=True,
):
    """
    Build a neuron x time matrix for spontaneous activity outside the trial period.

    Spontaneous activity is defined here as the time before the first trial start
    and after the last trial end for a given mouse. The two periods are binned
    separately and then concatenated along the time axis.

    Parameters
    ----------
    cluster_data : dict
        Output of ``load_mouse_data`` for a single mouse.
    trials_df : pd.DataFrame
        Trial table containing at least ``trial_start`` and ``trial_end``. If it
        also contains ``mouse_id``, the optional ``mouse_id`` argument can be used
        to select one mouse.
    mouse_id : int, optional
        Mouse identifier used to filter ``trials_df`` when a ``mouse_id`` column
        is present.
    bin_width : float, optional
        Width of the time bins in seconds. Default is 0.1 (100 ms).
    spike_feature : {"count", "rate"}, optional
        Whether to return raw spike counts or firing rates (Hz) within each bin.
    concatenate : bool, optional
        If True, concatenate pre-trial and post-trial spontaneous activity along
        the time axis. If False, return them separately.

    Returns
    -------
    If ``concatenate=True``:
        spontaneous_matrix : np.ndarray
            2D array with shape ``(n_neurons, n_spontaneous_bins)``.
        cluster_ids : list
            Cluster IDs matching the row order in ``spontaneous_matrix``.
        bin_edges : np.ndarray
            Bin edges for the concatenated spontaneous periods, expressed in a
            new concatenated time axis starting at 0.
        interval_info : pd.DataFrame
            Metadata describing the pre-trial and post-trial spontaneous segments.

    If ``concatenate=False``:
        spontaneous_blocks : dict
            Dictionary keyed by ``pre_trial`` and ``post_trial``. Each value is a
            dictionary with keys ``matrix`` and ``bin_edges``.
        cluster_ids : list
            Cluster IDs matching the row order in each returned matrix.
        interval_info : pd.DataFrame
            Metadata describing the pre-trial and post-trial spontaneous segments.
    """
    required_columns = ["trial_start", "trial_end"]
    missing_columns = [col for col in required_columns if col not in trials_df.columns]
    if missing_columns:
        raise ValueError(f"trials_df is missing required columns: {missing_columns}")
    if spike_feature not in {"count", "rate"}:
        raise ValueError("spike_feature must be either 'count' or 'rate'")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    trial_table = trials_df.copy()
    if "mouse_id" in trial_table.columns:
        if mouse_id is not None:
            trial_table = trial_table.query("mouse_id == @mouse_id")
        elif trial_table["mouse_id"].nunique() > 1:
            raise ValueError(
                "trials_df contains multiple mice. Pass mouse_id or pre-filter the DataFrame."
            )

    if trial_table.empty:
        raise ValueError("No trials available after filtering.")

    cluster_ids = sorted(cluster_data)
    all_spike_times = [
        np.asarray(cluster_data[cid]["spikes"], dtype=float) for cid in cluster_ids
    ]
    non_empty_spike_times = [spikes for spikes in all_spike_times if spikes.size > 0]
    if not non_empty_spike_times:
        empty_interval_info = pd.DataFrame(
            columns=[
                "segment",
                "start",
                "end",
                "duration",
                "n_bins",
                "concat_start",
                "concat_end",
            ]
        )
        if concatenate:
            return (
                np.empty((len(cluster_ids), 0), dtype=float),
                cluster_ids,
                np.array([0.0]),
                empty_interval_info,
            )
        return ({}, cluster_ids, empty_interval_info)

    session_start = min(spikes.min() for spikes in non_empty_spike_times)
    session_end = max(spikes.max() for spikes in non_empty_spike_times)
    first_trial_start = float(trial_table["trial_start"].min())
    last_trial_end = float(trial_table["trial_end"].max())

    spontaneous_intervals = [
        ("pre_trial", session_start, first_trial_start),
        ("post_trial", last_trial_end, session_end),
    ]

    interval_rows = []
    binned_blocks = []
    concatenated_edges = [0.0]
    spontaneous_blocks = {}
    concat_time = 0.0

    for segment_name, start, end in spontaneous_intervals:
        duration = float(end - start)
        if duration <= 0:
            continue

        bin_edges = np.arange(start, end, bin_width)
        if len(bin_edges) == 0 or bin_edges[-1] < end:
            bin_edges = np.append(bin_edges, end)
        if len(bin_edges) < 2:
            continue

        segment_matrix = np.zeros((len(cluster_ids), len(bin_edges) - 1), dtype=float)
        for row_idx, cid in enumerate(cluster_ids):
            spike_times = np.asarray(cluster_data[cid]["spikes"], dtype=float)
            in_segment = spike_times[(spike_times >= start) & (spike_times < end)]
            counts, _ = np.histogram(in_segment, bins=bin_edges)
            segment_matrix[row_idx, :] = counts

        if spike_feature == "rate":
            bin_durations = np.diff(bin_edges)
            segment_matrix = segment_matrix / bin_durations[np.newaxis, :]

        binned_blocks.append(segment_matrix)
        spontaneous_blocks[segment_name] = {
            "matrix": segment_matrix,
            "bin_edges": bin_edges,
        }
        segment_widths = np.diff(bin_edges)
        for width in segment_widths:
            concatenated_edges.append(concatenated_edges[-1] + float(width))
        concat_end = concat_time + duration
        interval_rows.append(
            {
                "segment": segment_name,
                "start": start,
                "end": end,
                "duration": duration,
                "n_bins": segment_matrix.shape[1],
                "concat_start": concat_time,
                "concat_end": concat_end,
            }
        )
        concat_time = concat_end

    if not binned_blocks:
        interval_info = pd.DataFrame(interval_rows)
        if concatenate:
            return (
                np.empty((len(cluster_ids), 0), dtype=float),
                cluster_ids,
                np.array([0.0]),
                interval_info,
            )
        return ({}, cluster_ids, interval_info)

    spontaneous_matrix = np.concatenate(binned_blocks, axis=1)
    concatenated_bin_edges = np.asarray(concatenated_edges, dtype=float)
    interval_info = pd.DataFrame(interval_rows)

    if concatenate:
        return spontaneous_matrix, cluster_ids, concatenated_bin_edges, interval_info
    return spontaneous_blocks, cluster_ids, interval_info


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

        for _, trial in valid_trials.reset_index(drop=True).iterrows():
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


def make_leave_one_group_majority_scorer(random_state=None):
    """
    Build a scorer for LeaveOneGroupOut that converts bin-wise predictions into
    one trial-level prediction.

    This scorer assumes each test fold contains exactly one group, which is true
    for ``LeaveOneGroupOut``. When available, ``predict_proba`` is used and
    class probabilities are averaged across bins in the held-out trial. If the
    estimator does not expose ``predict_proba``, the scorer falls back to a
    majority vote on ``predict`` outputs. Ties are broken at random.
    """
    rng = np.random.default_rng(random_state)

    def scorer(estimator, X, y_true):
        unique_true = np.unique(y_true)
        if len(unique_true) != 1:
            raise ValueError(
                "This scorer expects exactly one true label in the test fold. "
                "Use it with LeaveOneGroupOut where each held-out group belongs "
                "to a single trial."
            )

        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X)
            mean_proba = y_proba.mean(axis=0)
            winners = estimator.classes_[mean_proba == mean_proba.max()]
        else:
            y_pred = estimator.predict(X)
            predicted_labels, counts = np.unique(y_pred, return_counts=True)
            winners = predicted_labels[counts == counts.max()]

        final_pred = rng.choice(winners) if len(winners) > 1 else winners[0]
        return float(final_pred == unique_true[0])

    return scorer
