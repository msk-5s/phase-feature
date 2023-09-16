# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains the parameters used to generate results in the research paper.
"""

import dataclasses

from typing import Any, Callable, Dict, Mapping, Sequence
from nptyping import Float, NDArray, Shape

import mskutil as mu

#===================================================================================================
#===================================================================================================
@dataclasses.dataclass
class Parameters:
    """
    Parameters used to generate results in the research paper.
    """
    # pylint: disable=too-many-instance-attributes
    circuit_name: str
    data_delimiter: str
    random_state_base: int

    #***********************************************************************************************
    # Fuser parameters.
    #***********************************************************************************************
    fuser_names: Sequence[str]

    run_fuse_map: Mapping[
        str,
        Callable[[NDArray[Shape["*,*"], Float], Mapping[str, Any]], NDArray[Shape["*,*"], Float]]
    ]

    #***********************************************************************************************
    # Clusterer parameters.
    #***********************************************************************************************
    clusterer_names: Sequence[str]

    run_cluster_map: Mapping[
        str,
        Callable[
            [NDArray[Shape["*,*"], Float], int, Mapping[str, Any]], NDArray[Shape["*,*"], Float]
        ]
    ]

    #***********************************************************************************************
    # Denoiser parameters.
    #***********************************************************************************************
    denoiser_names: Sequence[str]

    denoiser_run_scale_map: Mapping[
        str,
        Callable[[NDArray[Shape["*,*"], Float], Mapping[str, Any]], NDArray[Shape["*,*"], Float]]
    ]

    run_denoise_map: Mapping[
        str, Callable[[NDArray[Shape["*,*"], Float], Mapping[str, Any]], mu.denoiser.Result]
    ]

    #***********************************************************************************************
    # Filterer parameters.
    #***********************************************************************************************
    filterer_names: Sequence[str]

    run_filter_map: Mapping[
        str,
        Callable[[NDArray[Shape["*,*"], Float], Mapping[str, Any]], NDArray[Shape["*,*"], Float]]
    ]

    #***********************************************************************************************
    # Noise injector parameters.
    #***********************************************************************************************
    noise_injector_names: Sequence[str]

    run_inject_noise_map: Mapping[
        str,
        Callable[
            [NDArray[Shape["*,*"], Float], int, Mapping[str, Any]], NDArray[Shape["*,*"], Float]
        ]
    ]

    #***********************************************************************************************
    # Reducer parameters.
    #***********************************************************************************************
    reducer_names: Sequence[str]

    reducer_run_scale_map: Mapping[
        str,
        Callable[[NDArray[Shape["*,*"], Float], Mapping[str, Any]], NDArray[Shape["*,*"], Float]]
    ]

    run_reduce_map: Mapping[
        str,
        Callable[
            [NDArray[Shape["*,*"], Float], int, Mapping[str, Any]], NDArray[Shape["*,*"], Float]
        ]
    ]

#***************************************************************************************************
#***************************************************************************************************
# Changes made to the `parameters` should be propagated to the `make_kwargs_map` function.
parameters = Parameters(
    circuit_name="lvna",
    data_delimiter=";",
    random_state_base=1337,

    #***********************************************************************************************
    # Fuser parameters.
    #***********************************************************************************************
    fuser_names=["cross_time", "mean"],

    run_fuse_map={
        "cross_time": mu.fuser.run_cross_time,
        "mean": mu.fuser.run_mean
    },

    #***********************************************************************************************
    # Clusterer parameters.
    #***********************************************************************************************
    clusterer_names=["kmeans"],

    run_cluster_map={
        "kmeans": mu.clusterer.run_kmeans
    },

    #***********************************************************************************************
    # Denoiser parameters.
    #***********************************************************************************************
    denoiser_names=["none", "svd"],

    denoiser_run_scale_map={
        "none": mu.scaler.run_none,
        "svd": mu.scaler.run_normalize
    },

    run_denoise_map={
        "none": mu.denoiser.run_none,
        "svd": mu.denoiser.run_svd
    },

    #***********************************************************************************************
    # Filterer parameters.
    #***********************************************************************************************
    filterer_names=["difference_1", "difference_2", "ideal_0p05", "none", "unit_magnitude"],

    run_filter_map={
        "difference_1": mu.filterer.run_difference,
        "difference_2": mu.filterer.run_difference,
        "ideal_0p05": mu.filterer.run_ideal,
        "none": mu.filterer.run_none,
        "unit_magnitude": mu.filterer.run_unit_magnitude
    },

    #***********************************************************************************************
    # Noise injector parameters.
    #***********************************************************************************************
    noise_injector_names=["gauss_0p002", "none"],

    run_inject_noise_map={
        "gauss_0p002": mu.noise_injector.inject_gauss,
        "none": mu.noise_injector.inject_none
    },

    #***********************************************************************************************
    # Reducer parameters.
    #***********************************************************************************************
    reducer_names=["none", "pca", "le", "tsne"],

    reducer_run_scale_map={
        "none": mu.scaler.run_none,
        "pca": mu.scaler.run_standardize,
        "le": mu.scaler.run_standardize,
        "tsne": mu.scaler.run_standardize
    },

    run_reduce_map={
        "none": mu.reducer.run_none,
        "pca": mu.reducer.run_pca,
        "le": mu.reducer.run_le,
        "tsne": mu.reducer.run_tsne
    }
)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_kwargs_atlas() -> Dict[str, Dict[str, Any]]:
    """
    Make the kwargs atlas.

    Since some kwargs will require additional processing to obtain, we cannot include the kwargs as
    part of the `parameters`. Changes to the `parameters` should be propagated to the kwargs in this
    function as needed.

    Returns
    -------
    dict of {str: dict of {str: any}}
        The new kwargs atlas.
    """
    #***********************************************************************************************
    # Make the fuser kwargs.
    #***********************************************************************************************
    fuser_kwargs_map = {
        "cross_time": {"will_flip": True},
        "mean": {}
    }

    # The fuser used can affect the number of meaningful components to use in the reducers.
    fuser_reducer_kwargs_atlas = {
        "cross_time": {
            "none": {},
            "pca": {"n_components": 3},
            "le": {"n_components": 3, "affinity": "rbf"},
            "tsne": {"n_components": 2}
        },
        "mean": {
            "none": {},
            "pca": {"n_components": 2},
            "le": {"n_components": 2, "affinity": "rbf"},
            "tsne": {"n_components": 2}
        }
    }

    #***********************************************************************************************
    # Make the clusterer kwargs.
    #***********************************************************************************************
    clusterer_kwargs_map = {
        "kmeans": {"n_clusters": 6, "n_init": 10}
    }

    #***********************************************************************************************
    # Make the denoiser kwargs.
    #***********************************************************************************************
    denoiser_kwargs_map = {
        "none": {},
        "svd": {}
    }

    denoiser_scaler_kwargs_map = {
        "none": {},
        "svd": {"feature_range": (0, 1)}
    }

    #***********************************************************************************************
    # Make the filterer kwargs.
    #***********************************************************************************************
    filterer_kwargs_map = {
        "difference_1": {"order": 1},
        "difference_2": {"order": 2},
        "ideal_0p05": {"cutoff": 0.05, "filter_type": "highpass"},
        "none": {},
        "unit_magnitude": {}
    }

    #***********************************************************************************************
    # Make the noise injector kwargs.
    #***********************************************************************************************
    noise_injector_kwargs_map = {
        "gauss_0p002": {"percent": 0.002},
        "none": {}
    }

    #***********************************************************************************************
    # Make the reducer kwargs.
    #***********************************************************************************************
    # See `fuser_reducer_kwargs_atlas` for reducer related kwargs.
    reducer_scaler_kwargs_map={
        "none": {},
        "pca": {},
        "le": {},
        "tsne": {}
    }

    #***********************************************************************************************
    # Make the kwargs atlas.
    #***********************************************************************************************
    kwargs_atlas = {
        "fuser": fuser_kwargs_map,
        "fuser_reducer_atlas": fuser_reducer_kwargs_atlas,
        "clusterer": clusterer_kwargs_map,
        "denoiser": denoiser_kwargs_map,
        "denoiser_scaler": denoiser_scaler_kwargs_map,
        "filterer": filterer_kwargs_map,
        "noise_injector": noise_injector_kwargs_map,
        "reducer_scaler": reducer_scaler_kwargs_map
    }

    return kwargs_atlas
