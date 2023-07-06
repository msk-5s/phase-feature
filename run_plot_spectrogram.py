# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots the average of the spectrograms of all load voltage magnitude measurements across
the year of data.
"""

import matplotlib.pyplot as plt

import data_factory
import mskutil as mu
import research

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.
    """
    random_state_base = research.parameters.random_state_base

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    days = 15
    timesteps_per_day = 96
    width = timesteps_per_day * days

    (data, _) = data_factory.make_data_raw()
    data_n = mu.noise_injector.inject_gauss(
        data=data, percent=0.002, random_state=random_state_base
    )

    run_denoise = mu.denoiser.run_svd

    spectrogram = mu.matrix_factory.make_average_spectrogram_matrix(
        data_n=data_n, denoiser_kwargs={}, run_denoise=run_denoise, stride=timesteps_per_day,
        width=width
    )

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    fontsize = 40
    plt.rc("axes", labelsize=fontsize)
    plt.rc("axes", titlesize=fontsize)
    plt.rc("figure", titlesize=fontsize)
    plt.rc("xtick", labelsize=fontsize)
    plt.rc("ytick", labelsize=fontsize)

    font = {"family": "monospace", "monospace": ["Times New Roman"]}
    plt.rc("font", **font)

    #***********************************************************************************************
    # Plot the virtual measurement spectrograms.
    #***********************************************************************************************
    (fig, _) = mu.plot_factory.make_spectrogram_plot(spectrogram=spectrogram)
    #fig.suptitle("Unfiltered Series")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
