#!/bin/bash
# <Place HPC tool specific headers here>

# SPDX-License-Identifier: BSD-3-Clause

# This bash script is for submitting an array job to a High Performance Computing (HPC) tool such
# as SLURM. Depending on the tool being used, you may only need to change `SLURM_ARRAY_TASK_ID` in
# the last section to the environment variable that is approriate for your HPC tool. Prepend any
# tool specific headers at line 2 above.

# Run `run_array_job.py` to get the total number of cases. This will be the number of array jobs.
# Total Suite Cases: 900

CASES_PER_INDEX=1

# Insert commands here.
python3 run_array_job.py $SLURM_ARRAY_TASK_ID $CASES_PER_INDEX
