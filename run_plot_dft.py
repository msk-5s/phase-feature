# SPDX-License-Identifier: BSD-3-Clause

"""
This script makes a scatter plot of the DFT-angle and magnitude terms of a data window.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.feather

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
    random_state_base = research.parameters.random_state_base

    #***********************************************************************************************
    # Load labels.
    #***********************************************************************************************
    circuit_name = research.parameters.circuit_name
    labels_df = pyarrow.feather.read_feather(f"data/{circuit_name}-labels.feather")
    labels = labels_df["phase_value"].to_numpy(dtype=int)

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    days = 15
    timesteps_per_day = 96
    width = days * timesteps_per_day
    start = 0

    (data_raw, channel_map) = data_factory.make_data_raw()

    data_raw = data_raw[0:(start + width), :]

    #***********************************************************************************************
    # Perform denoising.
    #***********************************************************************************************
    # Inject 0.02% Gaussian noise into the data window.
    data_n_raw = mu.noise_injector.inject_gauss(
        data=data_raw, random_state=random_state_base, percent=0.002
    )

    # Scale the window.
    data_n_raw = mu.scaler.run_normalize(data=data_n_raw)

    # Denoise the window.
    data_nd_raw = mu.denoiser.run_svd(data=data_n_raw).data

    # Fuse the channels.
    data = mu.fuser.run_cross_time(data=data_nd_raw, channel_map=channel_map)

    # Get the DFT magnitudes and angles.
    data_dft = np.fft.fft(data, axis=0)
    data_dft_angles = np.arctan(data_dft.imag / data_dft.real)
    data_dft_magnitudes = np.abs(data_dft)

    # Apply dimensionality reduction to the windows.
    data_dft_angles_dr = mu.reducer.run_tsne(data=data_dft_angles.T, n_components=2).data
    data_dft_magnitudes_dr = mu.reducer.run_tsne(data=data_dft_magnitudes.T, n_components=2).data

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    fontsize = 60
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
    # Plot scatter plots.
    #***********************************************************************************************
    _ = mu.plot_factory.make_scatter_dr_plot(
        data=data_dft_angles_dr, labels=labels, dot_size=200
    )

    _ = mu.plot_factory.make_scatter_dr_plot(
        data=data_dft_magnitudes_dr, labels=labels, dot_size=200
    )

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
