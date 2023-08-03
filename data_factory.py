# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains factory functions for making data.
"""

import json

from typing import List, Tuple
from nptyping import Float, NDArray, Shape

import pyarrow.feather

import research

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_data_raw() -> Tuple[NDArray[Shape["*,*"], Float], List[Tuple[str, List[int]]]]:
    """
    Make the raw load data.

    The raw load data will have multiple meter readings (columns) for polyphase loads.

    Returns
    -------
    data : numpy.ndarray of float, (n_timestep, n_meter_channel)
        The raw load voltage magnitude data.
    channel_map : dict of {str: list of int}
        The list of channel indices for each meter. These mappings are used to fuse multiple meter
        channels. Each item in the mapping will be {meter_name: list of channel indices}, where the
        indices are the columns in the returned data.
    """
    data_df = pyarrow.feather.read_feather(
        f"data/{research.parameters.circuit_name}-load-voltage_magnitudes-raw.feather"
    )

    filepath_channel_map = f"data/{research.parameters.circuit_name}-channel_map-load.json"

    with open(file=filepath_channel_map, mode="r", encoding="utf8") as handle:
        channel_map = json.load(fp=handle)

    data = data_df.to_numpy(dtype=float)

    return (data, channel_map)
