# SPDX-License-Identifier: BSD-3-Clause

"""
This script takes an array job index from some HPC tool and runs a simulation case for a given set
of parameters, based on the array job index.

Use the `run_case.py` script to directly run a simulation case with a given set of parameters.
"""

import gc
import itertools
import os
import sys

import json

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.

    The reason for using command line arguments is so this script can be used in an array job on
    high performance computing resources, if available.

    Arguments
    ---------
    array_index : int
        The array job ID provided by the HPC tool. It is assumed that the index starts at 1.
    cases_per_index : int
        The number of cases to run for each `array_index`. For example, a value of 10 means that
        cases 0-9 will run for index 1, cases 10-19 for index 2, etc. This is useful for situations
        where an HPC user might be limited in the number of jobs that can be ran in a day, where
        a single job corresponds to a single `array_index`.

    Examples
    --------
    The following will run with array job ID 1 and five cases per array job ID.

    python3 run_array_job.py 1 5
    """
    #***********************************************************************************************
    # Get command line arguemnts.
    #***********************************************************************************************
    array_index = int(sys.argv[1])
    cases_per_index = int(sys.argv[2])

    #***********************************************************************************************
    # Get the simulation parameters and cases.
    #***********************************************************************************************
    with open(file="simulation_cases.json", mode="r", encoding="utf-8") as handle:
        data_json = json.load(handle)

    cases = data_json["cases"]
    day_range = data_json["day_range"]

    #***********************************************************************************************
    # Create the base parameters.
    #***********************************************************************************************
    days_list = list(range(day_range[0], day_range[1] + 1))

    # Create an ordered list of tuples, of all possible combinations of the case parameters
    # (cartesian product).
    combinations = list(itertools.product(days_list, cases))

    #***********************************************************************************************
    # Split the combinations into groups by index according to the `case_per_index`.
    #***********************************************************************************************
    group_count = sum(divmod(len(combinations), cases_per_index))

    i_range = iter(range(0, len(combinations)))
    group_indices_list = [
        list(itertools.islice(i_range, cases_per_index)) for _ in range(group_count)
    ]
    group_indices = group_indices_list[array_index - 1]

    #***********************************************************************************************
    # Print top level info.
    #***********************************************************************************************
    print("="*50)
    print(f"Array Index: {array_index}")
    print(f"Group Count: {group_count}")
    print(f"Total Count: {len(combinations)}")
    print("="*50)

    #***********************************************************************************************
    # Get the respective parameters and run the simulation case.
    #***********************************************************************************************
    for i in group_indices:
        (days, parameters) = combinations[i]

        parameter_string = " ".join([
            f"{parameters['fuser']}",
            f"{parameters['clusterer']}",
            f"{days}",
            f"{parameters['denoiser']}",
            f"{parameters['filterer']}",
            f"{parameters['noise_injector']}",
            f"{parameters['reducer']}"
        ])

        #*******************************************************************************************
        # Print out info.
        #*******************************************************************************************
        print("-"*25)
        print(f"Group Index: {i}")
        print(f"Window Width: {days}")
        print(f"Case Parameters: {parameters}")
        print(f"python3 run_case.py {parameter_string}")
        print("-"*25)

        #*******************************************************************************************
        # Run the simulation case using the given parameters.
        #*******************************************************************************************
        os.system(f"python3 run_case.py {parameter_string}")

        # Force garbage collection. Running extensive simulation cases for large groups can cause
        # the running hardware to go out of memory (this can cause results to be lost).
        gc.collect()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
