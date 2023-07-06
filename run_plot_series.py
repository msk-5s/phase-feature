# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots the periodograms of filters.
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
    # Make the data.
    #***********************************************************************************************
    days = 15
    timesteps_per_day = 96
    width = days * timesteps_per_day
    start = 0

    (data, _) = data_factory.make_data_raw()
    data = data[start:(start + width), :]

    data_n = mu.noise_injector.inject_gauss(
        data=data, percent=0.002, random_state=random_state_base
    )

    data_nd = mu.denoiser.run_svd(data=data_n).data

    #data_nd_s = mu.filterer.run_difference(data=data_nd, order=2)
    #data_nd_s = mu.filterer.run_ideal(data=data_nd, cutoff=0.05)
    #data_nd_s = mu.filterer.run_none(data=data_nd)
    data_nd_s = mu.filterer.run_unit_magnitude(data=data_nd)

    series = data_nd[:, 0]
    series_s = data_nd_s[:, 0]

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    fontsize = 80
    plt.rc("axes", labelsize=fontsize)
    plt.rc("axes", titlesize=fontsize)
    plt.rc("figure", titlesize=fontsize)
    plt.rc("xtick", labelsize=fontsize)
    plt.rc("ytick", labelsize=fontsize)

    font = {"family": "monospace", "monospace": ["Times New Roman"]}
    plt.rc("font", **font)

    #***********************************************************************************************
    # Plot the filter spectrograms.
    #***********************************************************************************************
    (fig, axs) = plt.subplots()
    (fig_s, axs_s) = plt.subplots()
    (fig_a, _) = mu.plot_factory.make_autocorrelation_plot(series=series)
    (fig_as, _) = mu.plot_factory.make_autocorrelation_plot(series=series_s)

    fig.tight_layout()
    fig_s.tight_layout()
    fig_a.tight_layout()
    fig_as.tight_layout()

    axs.plot(series, linewidth=3)
    axs_s.plot(series_s, linewidth=3)

    axs.set_xlabel("Timestep (15-min)")
    axs.set_ylabel("Voltage (V)")
    axs_s.set_xlabel("Timestep (15-min)")
    axs_s.set_xlabel("Voltage (V)")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
