"""
Author: Kole Phillips

Retrieves the activity counts on the x-axis of each participant of a given study
 from the Table of Power. Then simulates a loss of data by overwriting X minutes
 of data for every 1 minute kept, and stores the resulting sets of activity counts
 per minute in a JSON. The values of X are 0, 1, 2, 4, and 8.

Usage: python visualize_dists.py CITY WAVE OUTPUT_DIRECTORY
  CITY: The city which the study took place in.
  WAVE: The number of the study.
  OUTPUT_DIRECTORY: The directory in which the JSON files are stored. The same
  directory used as an input directory for visualize_dists.py.
"""


import filtering
import interact_tools as it
import pandas as pd
from sys import argv


if __name__ == "__main__":
    city, wave = it.get_command_args("individual_counts.py")
    participants = it.top_iid_list(city, wave)
    if len(argv) > 3:
        output_dir = argv[3]
    else:
        output_dir = '.'
    results = {}

    for iid in participants:
        print(iid)
        out_fname = output_dir + '/' + str(iid) + '_pa_specificity_analysis.json'
        # Retrieve the necessary information from the ToP
        data = it.top_counts(iid, '1second', 'utcdate', 'x_count')
        data['utcdate'] = pd.to_datetime(data.utcdate)
        new_row = pd.Series()
        if data.empty:
            continue
        new_row['iid'] = iid
        for cycle in [0, 1, 2, 4, 8]:
            # Skip X minutes for every 1 minute kept by overwriting them with the kept minute
            counts = filtering.skip_data(data.drop_duplicates('utcdate'), cycle)
            if counts.empty:
                continue
            # Resample to the 1 minute level
            minute_counts = counts.set_index('utcdate').resample('1min').sum().dropna()
            results['counts_' + str(cycle)] = minute_counts.x_count.tolist()
        with open(out_fname, 'w') as f:
            f.write(str(results))
        results = {}
