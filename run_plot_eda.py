# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots the correlation matrix heatmap for the data before and after processing.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.feather
import sklearn.metrics

import data_factory
import mskutil as mu
import research

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main():
    # pylint: disable=too-many-locals,too-many-statements
    """
    The main function.
    """
    circuit_name = research.parameters.circuit_name

    random_state_base = research.parameters.random_state_base

    #***********************************************************************************************
    # Load labels.
    #***********************************************************************************************
    labels_df = pyarrow.feather.read_feather(f"data/{circuit_name}-labels.feather")
    labels_phase = labels_df["phase_value"].to_numpy(dtype=int)
    base_kvs = labels_df["base_kv"].to_numpy(dtype=float)

    #***********************************************************************************************
    # Make noise parameters.
    #***********************************************************************************************
    base_kv_partials = (0.01 * base_kvs)

    centers_list = [[-value, value] for value in base_kv_partials]
    percents_list = [[0.002, 0.002] for _ in range(len(base_kv_partials))]

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    days = 15
    timesteps_per_day = 96
    width = days * timesteps_per_day
    start = 0

    (data_raw, channel_map) = data_factory.make_data_raw()
    data_raw = data_raw[start:(start + width), :]

    data_raw_n = {
        "gauss": mu.noise_injector.inject_gauss(
            data=data_raw, random_state=random_state_base, percent=0.002
        ),
        "gmm": mu.noise_injector.inject_gmm(
            data=data_raw, random_state=random_state_base, centers_list=centers_list,
            percents_list=percents_list
        ),
        "laplace": mu.noise_injector.inject_laplace(
            data=data_raw, random_state=random_state_base, percent=0.002
        ),
        "mixed": mu.noise_injector.inject_mixed(
            data=data_raw, random_state=random_state_base, base_kv_partials=base_kv_partials,
            channel_map=channel_map, percent_cauchy=0.0035, percent_gauss=0.002
        )
    }["gauss"]

    #***********************************************************************************************
    # Processor names.
    #***********************************************************************************************
    denoiser_scaler_name = "normalize"
    denoiser_name = "svd"
    filterer_name = "none"
    fuser_name = "cross_time"
    reducer_scaler_name = "standardize"
    reducer_name = "tsne"

    #***********************************************************************************************
    # Select denoiser.
    #***********************************************************************************************
    run_denoiser_scale = {
        "normalize": mu.scaler.run_normalize,
        "standardize": mu.scaler.run_standardize
    }[denoiser_scaler_name]

    denoiser_scaler_kwargs = {
        "normalize": {"feature_range": [0, 1]},
        "standardize": {}
    }[denoiser_scaler_name]

    run_denoise = {
        "svd": mu.denoiser.run_svd,
        "iqr_svd": mu.denoiser.run_iqr_svd
    }[denoiser_name]

    denoiser_kwargs = {
        "svd": {},
        "iqr_svd": {"filterer_kwargs": {"cutoff": 0.05}, "run_filter": mu.filterer.run_ideal}
    }[denoiser_name]

    #***********************************************************************************************
    # Select filterer.
    #***********************************************************************************************
    run_filter = {
        "butterworth": mu.filterer.run_butterworth,
        "difference": mu.filterer.run_difference,
        "ideal": mu.filterer.run_ideal,
        "none": mu.filterer.run_none,
        "unit_magnitude": mu.filterer.run_unit_magnitude
    }[filterer_name]

    filterer_kwargs = {
        "butterworth": {"cutoff": 0.05, "order": 10},
        "difference": {"order": 2},
        "ideal": {"cutoff": 0.05},
        "none": {},
        "unit_magnitude": {"will_zero_first": True}
    }[filterer_name]

    #***********************************************************************************************
    # Select fuser.
    #***********************************************************************************************
    run_fuse = {
        "cross_func": mu.fuser.run_cross_func,
        "cross_time": mu.fuser.run_cross_time,
        "mean": mu.fuser.run_mean
    }[fuser_name]

    fuser_kwargs = {
        "cross_func": {"will_flip": True},
        "cross_time": {"will_flip": True},
        "mean": {}
    }[fuser_name]

    #***********************************************************************************************
    # Select reducer.
    #***********************************************************************************************
    run_reducer_scale = {
        "normalize": mu.scaler.run_normalize,
        "standardize": mu.scaler.run_standardize
    }[reducer_scaler_name]

    reducer_scaler_kwargs = {
        "normalize": {"feature_range": [0, 1]},
        "standardize": {}
    }[reducer_scaler_name]

    run_reduce = {
        "nmf": mu.reducer.run_nmf,
        "pca": mu.reducer.run_pca,
        "le": mu.reducer.run_le,
        "tsne": mu.reducer.run_tsne
    }[reducer_name]

    reducer_kwargs = {
        "nmf": {"n_components": 3},
        "pca": {"n_components": 3},
        "le": {"n_components": 3, "affinity": "rbf"},
        "tsne": {"n_components": 2, "perplexity": 50}
    }[reducer_name]

    #***********************************************************************************************
    # Preprocess the data.
    #***********************************************************************************************
    data_raw_n = run_denoiser_scale(data=data_raw_n, **denoiser_scaler_kwargs)
    data_raw_nd = run_denoise(data=data_raw_n, **denoiser_kwargs).data
    data_raw_nds = run_filter(data=data_raw_nd, **filterer_kwargs)
    data_nds = run_fuse(data=data_raw_nds, channel_map=channel_map, **fuser_kwargs)

    #***********************************************************************************************
    # Select a subset of the data based on meter channels.
    #***********************************************************************************************
    # 1: wye - 2: delta
    meter_counts = [1, 2]
    meter_indices_list = [np.where(labels_df["meter_count"] == count)[0] for count in meter_counts]

    meter_indices = [index for sublist in meter_indices_list for index in sublist]

    data_nds = data_nds[:, meter_indices]
    labels_phase = labels_phase[meter_indices]

    #***********************************************************************************************
    # Dimensionality reduction.
    #***********************************************************************************************
    data_nds = run_reducer_scale(data=data_nds, **reducer_scaler_kwargs)
    data_nds_dr = run_reduce(data=data_nds.T, random_state=random_state_base, **reducer_kwargs).data

    print(data_nds_dr.shape)

    #***********************************************************************************************
    # Print metrics.
    #***********************************************************************************************
    corr_i = mu.matrix_factory.make_ideal_phase_correlation_matrix(labels=labels_phase)
    corr_load = mu.matrix_factory.make_phase_sorted_correlation_matrix(
        data=data_nds, labels=labels_phase
    )

    error = np.linalg.norm((corr_i - corr_load), ord="fro")
    score = sklearn.metrics.calinski_harabasz_score(data_nds_dr, labels=labels_phase)

    print(f"Frobenius Error: {error}")
    print(f"CH Score: {score}")

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    fontsize = 80
    legend_fontsize = 30

    plt.rc("legend", fontsize=legend_fontsize)
    plt.rc("axes", labelsize=fontsize)
    plt.rc("axes", titlesize=fontsize)
    plt.rc("figure", titlesize=fontsize)
    plt.rc("xtick", labelsize=fontsize)
    plt.rc("ytick", labelsize=fontsize)

    font = {"family": "monospace", "monospace": ["Times New Roman"]}
    plt.rc("font", **font)

    #***********************************************************************************************
    # Plot correlation matrices.
    #***********************************************************************************************
    (_, axs_corr, cbar) = mu.plot_factory.make_correlation_heatmap(
        data=data_nds, labels=labels_phase, cmap="afmhot"
    )
    axs_corr.set_xlabel(None)
    axs_corr.set_ylabel(None)
    cbar.ax.set_ylabel(None)

    (_, axs_dr) = mu.plot_factory.make_scatter_dr_plot(
        data=data_nds_dr, labels=labels_phase, dot_size=200
    )
    #axs_dr.set_xlabel(None)
    #axs_dr.set_ylabel(None)
    if data_nds_dr.shape[1] > 2:
        axs_dr.set_zlabel(None)

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
