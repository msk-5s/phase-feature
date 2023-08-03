# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots the a summary statistic of the Frobenius error in the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import result_factory

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main():
    # pylint: disable=too-many-locals,too-many-statements,consider-iterating-dictionary
    """
    The main function.
    """
    #***********************************************************************************************
    # Set up parameters and make results.
    #***********************************************************************************************
    stat_name = ["mean", "std"][0]
    run_stat = {"mean": np.mean, "std": np.std}[stat_name]

    #config = result_factory.make_fuser_reducer_config(
    #    denoiser_name = ["none", "svd"][1],
    #)

    config = result_factory.make_denoiser_fuser_config(
        reducer_name = ["pca", "le", "tsne"][0]
    )

    #config = result_factory.make_denoiser_filterer_config(
    #    fuser_name=["cross_time", "mean"][1],
    #    reducer_name=["pca", "le", "tsne"][0]
    #)

    #config = result_factory.make_denoiser_reducer_config(
    #    fuser_name=["cross_time", "mean"][1],
    #    filterer_name=["difference_1", "difference_2", "ideal_0p05", "none", "unit_magnitude"][0]
    #)

    (scores_atlas, results_atlas) = result_factory.make_results(
        subfilenames=config.subfilenames, run_stat=run_stat
    )

    print(scores_atlas)

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    sns.set_theme(style="whitegrid")

    fontsize = 40
    legend_fontsize = 20
    plt.rc("axes", labelsize=fontsize)
    plt.rc("legend", title_fontsize=legend_fontsize)
    plt.rc("legend", fontsize=legend_fontsize)
    plt.rc("xtick", labelsize=fontsize)
    plt.rc("ytick", labelsize=fontsize)

    font = {"family": "monospace", "monospace": ["Times New Roman"]}
    plt.rc("font", **font)

    #***********************************************************************************************
    # Make the plot.
    #***********************************************************************************************
    (_, axs) = plt.subplots()

    linewidth = 3
    marker_size = 100

    stat_pretty_name = {
        "mean": "Mean",
        "std": "STD"
    }[stat_name]

    days_range = range(1, 31)

    for(secondary_name, secondary_pretty_name) in config.secondary_pretty_names.items():
        for (primary_name, primary_pretty_name) in config.primary_pretty_names.items():
            axs.plot(
                days_range, results_atlas[primary_name][secondary_name],
                color=config.colors[secondary_name], linestyle=config.linestyles[primary_name],
                linewidth=linewidth,
                label=f"{secondary_pretty_name}: {primary_pretty_name}"
            )

            axs.scatter(
                days_range, results_atlas[primary_name][secondary_name],
                c=config.colors[secondary_name], marker=config.markers[secondary_name],
                s=marker_size
            )
        #axs.plot(
        #    days_range, results_atlas["none"][secondary_name],
        #    color=config.colors[secondary_name], linestyle="dashed", linewidth=linewidth,
        #    label=f"{secondary_pretty_name}: {config.primary_pretty_names['none']}"
        #)

        #axs.plot(
        #    days_range, results_atlas["svd"][secondary_name],
        #    color=config.colors[secondary_name], linestyle="solid", linewidth=linewidth,
        #    label=f"{secondary_pretty_name}: {config.primary_pretty_names['svd']}"
        #)

        #axs.scatter(
        #    days_range, results_atlas["none"][secondary_name],
        #    c=config.colors[secondary_name], marker=config.markers[secondary_name], s=marker_size
        #)

        #axs.scatter(
        #    days_range, results_atlas["svd"][secondary_name],
        #    c=config.colors[secondary_name], marker=config.markers[secondary_name], s=marker_size
        #)

    # Change linewidth of the lines shown in the legend.
    _ = [line.set_linewidth(linewidth) for line in axs.legend().get_lines()]

    axs.set_xlabel("Window Width (Days)")
    axs.set_ylabel(f"Accuracy ({stat_pretty_name})")
    axs.legend(loc="best", fontsize=40)
    #axs.set_ylim([0, 0.13])
    axs.set_ylim([0, 1])

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
