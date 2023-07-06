# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots the periodograms of filters.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import mskutil as mu

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.
    """
    #***********************************************************************************************
    # Make the filters.
    #***********************************************************************************************
    width = 96 * 15

    frequencies = np.fft.fftfreq(width) + 1e-6

    # Ideal parameters.
    cutoff = 0.05

    difference_1_dft = frequencies
    difference_2_dft = frequencies ** 2
    ideal_dft = np.array([1 if abs(frequency) > cutoff else 0 for frequency in frequencies])

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    sns.set_theme(style="whitegrid")

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
    linewidth = 10

    (fig_difference_1, axs_difference_1) = mu.plot_factory.make_periodogram_filter_plot(
        filter_dft=difference_1_dft, frequencies=frequencies, linewidth=linewidth
    )

    (fig_difference_2, axs_difference_2) = mu.plot_factory.make_periodogram_filter_plot(
        filter_dft=difference_2_dft, frequencies=frequencies, linewidth=linewidth
    )

    (fig_ideal, axs_ideal) = mu.plot_factory.make_periodogram_filter_plot(
        filter_dft=ideal_dft, frequencies=frequencies, linewidth=linewidth
    )

    fig_difference_1.tight_layout()
    fig_difference_2.tight_layout()
    fig_ideal.tight_layout()

    #fig_difference_1.suptitle("Difference (First-Order)")
    #fig_difference_2.suptitle("Difference (Second-Order)")
    #fig_ideal.suptitle("Ideal")

    axs_difference_1.set_ylim([0, 0.6])
    axs_difference_2.set_ylim([0, 0.3])
    axs_ideal.set_ylim([0, 1.2])

    axs_difference_2.set_yticks([0, 0.25])

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
