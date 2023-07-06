# **phase-feature**

This repository contains the source code for recreating the research in "Phase Identification in Power Distribution Systems via Feature Engineering". The dataset can be downloaded from [Kaggle](https://www.kaggle.com/msk5sdata/arima-ercot-2021). Alternatively, the dataset in this work can be recreated from scratch using the [arima-ercot-2021-profile](https://github.com/msk-5s/arima-ercot-2021-profile.git) and [arima-ercot-2021-opendss](https://github.com/msk-5s/arima-ercot-2021-opendss.git) repositories.

## Requirements
    - Python 3.8+ (64-bit)
    - See requirements.txt file for the required python packages.

## Folders
`data/`
: The `lvna` dataset from the [arima-ercot-2021](https://www.kaggle.com/msk5sdata/arima-ercot-2021) data suite should be placed in this folder.

`mskutil/`
: The [mskutil](https://github.com/msk-5s/mskutil) repository as a submodule.

`results/`
: This folder contains the phase identification results for different parameters. These are the results reported in the paper.

## Running
The `run_case.py` script can be used to run phase identification across the entire year of data for different parameters. If you have access to a computing cluster, then use the `submit_job.sh`, which will run `run_case.py` as an array job via the `run_array_job.py` script. `submit_job.sh` will need to be modified for the software being used by the computing cluster (see the comments in the script).
> **NOTE: `run_case.py` will save results to the `results/` folder.**
