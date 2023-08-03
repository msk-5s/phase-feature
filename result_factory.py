# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains factory functions for making results.
"""

from typing import Callable, Dict, Mapping, NamedTuple, Sequence

import json

import sklearn.metrics

_fuser_graphics_atlas = {
    "colors": {"cross_time": "blue", "cross_time_difference": "black", "mean": "red"},
    "markers": {"cross_time": "o", "cross_time_difference": "v", "mean": "X"},
    "linestyles": {"cross_time": "solid", "cross_time_difference": "dotted", "mean": "dashed"}
}

_fuser_pretty_names = {
    "cross_time": "Cross-Product (Time)",
    "cross_time_difference": "Cross-Product (Time) + Filter",
    "mean": "Mean"
}

_denoiser_pretty_names = {
    "none": "Noise",
    "svd": "Denoised"
}

_denoiser_graphics_atlas = {
    "linestyles": {"none": "dashed", "svd": "solid"}
}

_filterer_graphics_atlas = {
    "colors": {
        "difference_1": "green", "difference_2": "orange", "none": "red", "ideal_0p05": "black",
        "unit_magnitude": "blue"
    },
    "markers" : {
        "difference_1": "o", "difference_2": "v", "none": "s", "ideal_0p05": "D",
        "unit_magnitude": "X"
    }
}

_filterer_pretty_names = {
    "difference_1": r"$H(\omega)_{D, 1}$",
    "difference_2": r"$H(\omega)_{D, 2}$",
    "ideal_0p05": r"$H(\omega)_{I}$",
    "none": "None",
    "unit_magnitude": r"$H(\omega)_{U}$"
}

_reducer_graphics_atlas = {
    "colors": {"pca": "red", "le": "green", "tsne": "blue"},
    "markers": {"pca": "o", "le": "v", "tsne": "X"}
}

_reducer_pretty_names = {
    "pca": "PCA",
    "le": "LE",
    "tsne": "t-SNE"
}

#===================================================================================================
#===================================================================================================
class Config(NamedTuple):
    """

    Attributes
    ----------
    colors : dict of {str, str}
        The plot line colors for the secondary target.
    linestyles : dict of {str, str}
        The plot line styles for the primary target.
    markers : dict of {str, str}
        The plot markers for the secondary target.
    primary_pretty_names : dict of {str: str}
        The pretty names for the primary target.
    secondary_pretty_names : dict of {str: str}
        The pretty names for the secondary target.
    subfilenames : dict of {str: dict of {str: str}}
        The name of the subfiles to get the results for.
    """
    colors: Mapping[str, str]
    linestyles: Mapping[str, str]
    markers: Mapping[str, str]
    primary_pretty_names: Mapping[str, str]
    secondary_pretty_names: Mapping[str, str]
    subfilenames: Mapping[str, Mapping[str, str]]

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_fuser_reducer_config(denoiser_name: str) -> Dict[str, Dict[str, str]]:
    """
    Make the configuration that compares the different reducers with fused data.

    Parameters
    ----------
    denoiser_name : str
        The name of the denoiser to use.

    Returns
    -------
    Config
        The configuration for making results data.
    """
    subfilenames = {
        "mean": {
            "pca":
                f"mean-fuser-kmeans-clusterer-{denoiser_name}-denoiser-unit_magnitude-" +\
                "filterer-gauss_0p002-noise_injector-pca-reducer",
            "le":
                f"mean-fuser-kmeans-clusterer-{denoiser_name}-denoiser-unit_magnitude-" +\
                "filterer-gauss_0p002-noise_injector-le-reducer",
            "tsne":
                f"mean-fuser-kmeans-clusterer-{denoiser_name}-denoiser-difference_2-" +\
                "filterer-gauss_0p002-noise_injector-tsne-reducer"
        },
        "cross_time": {
            "pca":
                f"cross_time-fuser-kmeans-clusterer-{denoiser_name}-denoiser-" +\
                "difference_2-filterer-gauss_0p002-noise_injector-pca-reducer",
            "le":
                f"cross_time-fuser-kmeans-clusterer-{denoiser_name}-denoiser-" +\
                "difference_2-filterer-gauss_0p002-noise_injector-le-reducer",
            "tsne":
                f"cross_time-fuser-kmeans-clusterer-{denoiser_name}-denoiser-" +\
                "none-filterer-gauss_0p002-noise_injector-tsne-reducer"
        }
    }

    # pylint: disable=consider-iterating-dictionary
    linestyles = {key: _fuser_graphics_atlas["linestyles"][key] for key in subfilenames.keys()}
    pretty_names = {key: _fuser_pretty_names[key] for key in subfilenames.keys()}

    config = Config(
        colors=_reducer_graphics_atlas["colors"],
        linestyles=linestyles,
        markers=_reducer_graphics_atlas["markers"],
        primary_pretty_names=pretty_names,
        secondary_pretty_names=_reducer_pretty_names,
        subfilenames=subfilenames
    )

    return config

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_denoiser_fuser_config(reducer_name: str) -> Dict[str, Dict[str, str]]:
    """
    Make the configuration that compares the fusers with noise/denoised data.

    Parameters
    ----------
    reducer_name : str
        The name of the reducer to use.

    Returns
    -------
    Config
        The configuration for making results data.
    """
    subfilenames = {
        "none": {
            "cross_time":
                "cross_time-fuser-kmeans-clusterer-none-denoiser-none-filterer-" +\
                f"gauss_0p002-noise_injector-{reducer_name}-reducer",
            "cross_time_difference":
                "cross_time-fuser-kmeans-clusterer-none-denoiser-difference_2-filterer-" +\
                f"gauss_0p002-noise_injector-{reducer_name}-reducer",
            "mean":
                "mean-fuser-kmeans-clusterer-none-denoiser-unit_magnitude-filterer-" +\
                f"gauss_0p002-noise_injector-{reducer_name}-reducer"
        },
        "svd": {
            "cross_time":
                "cross_time-fuser-kmeans-clusterer-svd-denoiser-none-filterer-" +\
                f"gauss_0p002-noise_injector-{reducer_name}-reducer",
            "cross_time_difference":
                "cross_time-fuser-kmeans-clusterer-svd-denoiser-difference_2-filterer-" +\
                f"gauss_0p002-noise_injector-{reducer_name}-reducer",
            "mean":
                "mean-fuser-kmeans-clusterer-svd-denoiser-unit_magnitude-filterer-" +\
                f"gauss_0p002-noise_injector-{reducer_name}-reducer"
        }
    }

    config = Config(
        colors=_fuser_graphics_atlas["colors"],
        linestyles=_denoiser_graphics_atlas["linestyles"],
        markers=_fuser_graphics_atlas["markers"],
        primary_pretty_names=_denoiser_pretty_names,
        secondary_pretty_names=_fuser_pretty_names,
        subfilenames=subfilenames
    )

    return config

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_denoiser_filterer_config(
    fuser_name: str, reducer_name: str
) -> Dict[str, Dict[str, str]]:
    """
    Make the configuration that compares the different filters with noise/denoised data.

    Parameters
    ----------
    fuser_name : str
        The name of the fuser to use.
    reducer_name : str
        The name of the reducer to use.

    Returns
    -------
    Config
        The configuration for making results data.
    """
    subfilenames = {
        "none": {
            "difference_1":
                f"{fuser_name}-fuser-kmeans-clusterer-none-denoiser-difference_1-" +\
                f"filterer-gauss_0p002-noise_injector-{reducer_name}-reducer",
            "difference_2":
                f"{fuser_name}-fuser-kmeans-clusterer-none-denoiser-difference_2-" +\
                f"filterer-gauss_0p002-noise_injector-{reducer_name}-reducer",
            "ideal_0p05":
                f"{fuser_name}-fuser-kmeans-clusterer-none-denoiser-ideal_0p05-" +\
                f"filterer-gauss_0p002-noise_injector-{reducer_name}-reducer",
            "none":
                f"{fuser_name}-fuser-kmeans-clusterer-none-denoiser-none-" +\
                f"filterer-gauss_0p002-noise_injector-{reducer_name}-reducer",
            "unit_magnitude":
                f"{fuser_name}-fuser-kmeans-clusterer-none-denoiser-unit_magnitude-" +\
                f"filterer-gauss_0p002-noise_injector-{reducer_name}-reducer",
        },
        "svd": {
            "difference_1":
                f"{fuser_name}-fuser-kmeans-clusterer-svd-denoiser-difference_1-" +\
                f"filterer-gauss_0p002-noise_injector-{reducer_name}-reducer",
            "difference_2":
                f"{fuser_name}-fuser-kmeans-clusterer-svd-denoiser-difference_2-" +\
                f"filterer-gauss_0p002-noise_injector-{reducer_name}-reducer",
            "ideal_0p05":
                f"{fuser_name}-fuser-kmeans-clusterer-svd-denoiser-ideal_0p05-" +\
                f"filterer-gauss_0p002-noise_injector-{reducer_name}-reducer",
            "none":
                f"{fuser_name}-fuser-kmeans-clusterer-svd-denoiser-none-" +\
                f"filterer-gauss_0p002-noise_injector-{reducer_name}-reducer",
            "unit_magnitude":
                f"{fuser_name}-fuser-kmeans-clusterer-svd-denoiser-unit_magnitude-" +\
                f"filterer-gauss_0p002-noise_injector-{reducer_name}-reducer",
        }
    }

    config = Config(
        colors=_filterer_graphics_atlas["colors"],
        linestyles=_denoiser_graphics_atlas["linestyles"],
        markers=_filterer_graphics_atlas["markers"],
        primary_pretty_names=_denoiser_pretty_names,
        secondary_pretty_names=_filterer_pretty_names,
        subfilenames=subfilenames
    )

    return config

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_denoiser_reducer_config(
    fuser_name: str, filterer_name: str
) -> Dict[str, Dict[str, str]]:
    """
    Make the configuration that compares the different reducers with noise/denoised data.

    Parameters
    ----------
    fuser_name : str
        The name of the fuser to use.
    filterer_name : str
        The name of the filterer to use.

    Returns
    -------
    Config
        The configuration for making results data.
    """
    subfilenames = {
        "none": {
            "pca":
                f"{fuser_name}-fuser-kmeans-clusterer-none-denoiser-" +\
                f"{filterer_name}-filterer-gauss_0p002-noise_injector-pca-reducer",
            "le":
                f"{fuser_name}-fuser-kmeans-clusterer-none-denoiser-" +\
                f"{filterer_name}-filterer-gauss_0p002-noise_injector-le-reducer",
            "tsne":
                f"{fuser_name}-fuser-kmeans-clusterer-none-denoiser-" +\
                f"{filterer_name}-filterer-gauss_0p002-noise_injector-tsne-reducer",
        },
        "svd": {
            "pca":
                f"{fuser_name}-fuser-kmeans-clusterer-svd-denoiser-" +\
                f"{filterer_name}-filterer-gauss_0p002-noise_injector-pca-reducer",
            "le":
                f"{fuser_name}-fuser-kmeans-clusterer-svd-denoiser-" +\
                f"{filterer_name}-filterer-gauss_0p002-noise_injector-le-reducer",
            "tsne":
                f"{fuser_name}-fuser-kmeans-clusterer-svd-denoiser-" +\
                f"{filterer_name}-filterer-gauss_0p002-noise_injector-tsne-reducer",
        }
    }

    config = Config(
        colors=_reducer_graphics_atlas["colors"],
        linestyles=_denoiser_graphics_atlas["linestyles"],
        markers=_reducer_graphics_atlas["markers"],
        primary_pretty_names=_denoiser_pretty_names,
        secondary_pretty_names=_reducer_pretty_names,
        subfilenames=subfilenames
    )

    return config

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_results(
    subfilenames: Mapping[str, Mapping[str, str]], run_stat: Callable[[Sequence[float]], float]
) -> Dict[str, Dict[str, float]]:
    # pylint: disable=too-many-locals
    """
    Make the results.

    Parameters
    ----------
    subfilenames : dict of {str: dict of {str: str}}
        The name of the subfiles to get the results for.
    run_stat : callable
        The function to calculate the summary statistic.

    Returns
    -------
    scores_atlas : dict of {str: dict of {str: float}}
        The results.
    results_atlas : dict of {str: dict of {str: str}}
        The results.
    """
    days_range = range(1, 31)

    scores_atlas = {
        primary_name: {secondary_name: 0.0 for secondary_name in secondary_results.keys()}
    for (primary_name, secondary_results) in subfilenames.items()}

    results_atlas = {
        primary_name: {secondary_name: [] for secondary_name in secondary_results.keys()}
    for (primary_name, secondary_results) in subfilenames.items()}

    for (primary_name, secondary_results) in results_atlas.items():
        for (secondary_name, results) in secondary_results.items():
            for days in days_range:
                subfilename = subfilenames[primary_name][secondary_name]
                filepath = f"results/result-{days}-days-{subfilename}.json"

                with open(file=filepath, mode="r", encoding="utf8") as handle:
                    json_data = json.load(handle)
                    metrics = json_data["accuracies"]
                    result = run_stat(metrics)

                    results.append(result)

            # Calculate the normalized AUC.
            score = sklearn.metrics.auc(x=days_range, y=results) / (1.0 * len(days_range))
            scores_atlas[primary_name][secondary_name] = score

    return (scores_atlas, results_atlas)
