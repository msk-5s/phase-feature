# SPDX-License-Identifier: BSD-3-Clause

"""
This script runs a set of parameters on windows across the entire year of data. The results are
saved to a JSON file in the `results` folder.
"""

import json
import sys
import time

from rich.progress import track

import numpy as np
import pyarrow.feather
import sklearn.metrics

import data_factory
import mskutil as mu
import research

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals,too-many-statements
    """
    The main function.

    Arguments
    ---------
    fuser_name : str
        The name of the fuer to use. See `research.parameters.fuser_names` for the valid
        names.
    clusterer_name : str
        The name of the clusterer to use. See `research.parameters.clusterer_names` for the valid
        names.
    days : int
        The width of the data window in days.
    denoiser_name : str
        The name of the denoiser to use. See `research.parameters.denoiser_names` for the valid
        names.
    filterer_name : str
        The name of the filterer to use. See `research.parameters.filterer_names` for the valid
        names.
    noise_injector_name : str
        The noise injector to use. See `research.parameters.noise_names` for the valid names.
    reducer_name : str
        The reducer to use. See `research.parameters.reducer_names` for the valid names.

    Examples
    --------
    The following will run a case using the mean fuser, kmeans clusterer, 15-day window, SVD
    denoiser, ideal filter, Gaussian noise model, and PCA reducer:

    python3 run_case_plc.py mean kmeans 15 svd ideal_0p05 gauss pca
    """
    circuit_name = research.parameters.circuit_name
    kwargs_atlas = research.make_kwargs_atlas()

    #***********************************************************************************************
    # Get command line arguemnts.
    #***********************************************************************************************
    fuser_name = str(sys.argv[1])
    clusterer_name = str(sys.argv[2])
    days = int(sys.argv[3])
    denoiser_name = str(sys.argv[4])
    filterer_name = str(sys.argv[5])
    noise_injector_name = str(sys.argv[6])
    reducer_name = str(sys.argv[7])

    #***********************************************************************************************
    # Set the random state.
    #***********************************************************************************************
    # Since only the noise is randomized, we want to use the same random state for all cases. This
    # allows the same noisy data to be used for all cases.
    random_state_base = research.parameters.random_state_base

    #***********************************************************************************************
    # Get noise injector kwargs.
    #***********************************************************************************************
    noise_injector_kwargs = kwargs_atlas["noise_injector"][noise_injector_name]
    run_inject_noise = research.parameters.run_inject_noise_map[noise_injector_name]

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    (data_raw, channel_map) = data_factory.make_data_raw()

    data_raw_n = run_inject_noise(
        data=data_raw, random_state=random_state_base, **noise_injector_kwargs
    )

    # We no longer need the original data.
    del data_raw

    #***********************************************************************************************
    # Get the fuser and kwargs.
    #***********************************************************************************************
    fuser_kwargs = kwargs_atlas["fuser"][fuser_name]
    run_fuse = research.parameters.run_fuse_map[fuser_name]

    #***********************************************************************************************
    # Get the clusterer and kwargs.
    #***********************************************************************************************
    clusterer_kwargs = kwargs_atlas["clusterer"][clusterer_name]
    run_cluster = research.parameters.run_cluster_map[clusterer_name]

    #***********************************************************************************************
    # Get the denoiser and kwargs.
    #***********************************************************************************************
    denoiser_kwargs = kwargs_atlas["denoiser"][denoiser_name]
    run_denoise = research.parameters.run_denoise_map[denoiser_name]

    denoiser_scaler_kwargs = kwargs_atlas["denoiser_scaler"][denoiser_name]
    run_denoiser_scale = research.parameters.denoiser_run_scale_map[denoiser_name]

    #***********************************************************************************************
    # Get the filterer and kwargs.
    #***********************************************************************************************
    filterer_kwargs = kwargs_atlas["filterer"][filterer_name]
    run_filter = research.parameters.run_filter_map[filterer_name]

    #***********************************************************************************************
    # Get the reducer and kwargs.
    #***********************************************************************************************
    reducer_kwargs = kwargs_atlas["fuser_reducer_atlas"][fuser_name][reducer_name]
    run_reduce = research.parameters.run_reduce_map[reducer_name]

    reducer_scaler_kwargs = kwargs_atlas["reducer_scaler"][reducer_name]
    run_reducer_scale = research.parameters.reducer_run_scale_map[reducer_name]

    #***********************************************************************************************
    # Calculate the index for each window in the data.
    #***********************************************************************************************
    timesteps_per_day = 96
    width = days * timesteps_per_day

    start_indices = np.arange(
        start=0, stop=data_raw_n.shape[0] - timesteps_per_day * (days - 1), step=timesteps_per_day
    )

    # Use this block of code to run the case for smaller number of windows.
    #window_count = 20
    #start_indices = np.arange(
    #    start=0, stop=96 * window_count, step=timesteps_per_day
    #)

    #***********************************************************************************************
    # Run the simulation case.
    #***********************************************************************************************
    rng = np.random.default_rng(seed=random_state_base)

    labels_df = pyarrow.feather.read_feather(f"data/{circuit_name}-labels.feather")
    labels_true = labels_df["phase_value"].to_numpy(dtype=int)
    labels_error = mu.label_modifier.run_error_change(
        labels=labels_true, percent=0.4, random_state=random_state_base
    )

    accuracies = []
    clusterer_times = []
    denoiser_times = []
    filterer_times = []
    metadata_denoisers = []
    metadata_reducers = []
    random_states = []
    reducer_times = []
    window_starts = []

    for start in track(start_indices, "Processing..."):
        random_state_local = int(rng.integers(np.iinfo(np.int32).max))
        window_raw_n = data_raw_n[start:(start + width), :]

        # Denoise the data.
        window_raw_n = run_denoiser_scale(data=window_raw_n, **denoiser_scaler_kwargs)

        denoiser_time = time.perf_counter()
        denoiser_result = run_denoise(data=window_raw_n, **denoiser_kwargs)
        denoiser_time = time.perf_counter() - denoiser_time

        window_raw_nd = denoiser_result.data

        # Filter the denoised data.
        filterer_time = time.perf_counter()
        window_raw_nds = run_filter(data=window_raw_nd, **filterer_kwargs)
        filterer_time = time.perf_counter() - filterer_time

        # Fuse the channels.
        window_nds = run_fuse(
            data=window_raw_nds, channel_map=channel_map, **fuser_kwargs
        )

        # Dimensionality reduction.
        window_nds = run_reducer_scale(data=window_nds, **reducer_scaler_kwargs)

        reducer_time = time.perf_counter()
        reducer_result = run_reduce(
            data=window_nds.T, random_state=random_state_local, **reducer_kwargs
        )
        reducer_time = time.perf_counter() - reducer_time

        window_nds_dr = reducer_result.data

        # Phase label correction.
        clusterer_time = time.perf_counter()
        predicted_clusters = run_cluster(
            data=window_nds_dr, random_state=random_state_local, **clusterer_kwargs
        )
        clusterer_time = time.perf_counter() - clusterer_time

        predicted_labels = mu.label_modifier.run_majority_vote_correction(
            clusters=predicted_clusters, labels=labels_error
        )

        # Calculate the accuracy.
        accuracy = sklearn.metrics.accuracy_score(y_true=labels_true, y_pred=predicted_labels)

        # Save the results.
        accuracies.append(accuracy)
        clusterer_times.append(clusterer_time)
        denoiser_times.append(denoiser_time)
        filterer_times.append(filterer_time)
        metadata_denoisers.append(denoiser_result.other_to_metadata(denoiser_result.other))
        metadata_reducers.append(reducer_result.other_to_metadata(reducer_result.other))
        random_states.append(random_state_local)
        reducer_times.append(reducer_time)

        # We need to cast `start` to an `int` so that it can be saved by the JSON parser. `track`
        # returns an `int64`, which isn't supported.
        window_starts.append(int(start))

    #***********************************************************************************************
    # Save results.
    #***********************************************************************************************
    json_dict = {
        "random_state_base": random_state_base,
        "accuracies": accuracies,
        "denoiser_times": denoiser_times,
        "filterer_times": filterer_times,
        "reducer_times": reducer_times,
        "window_starts": window_starts,
        "random_states": random_states,
        "denoiser": metadata_denoisers,
        "reducer": metadata_reducers
    }

    filepath = "-".join([
        "results/result",
        f"{days}-days",
        f"{fuser_name}-fuser",
        f"{clusterer_name}-clusterer",
        f"{denoiser_name}-denoiser",
        f"{filterer_name}-filterer",
        f"{noise_injector_name}-noise_injector",
        f"{reducer_name}-reducer.json"
    ])

    with open(file=filepath, mode="w", encoding="utf-8") as handle:
        json.dump(obj=json_dict, fp=handle, indent=4)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
