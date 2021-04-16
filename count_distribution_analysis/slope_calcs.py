"""
Author: Kole Phillips

Uses the distributions of x-axis activity counts to fit piecewise exponential functions, and stores the resulting
breakpoints and slopes of these functions as a metric to describe physical activity in a CSV

Usage: python visualize_dists.py CITY WAVE CYCLE
  CITY: The city which the study took place in.
  WAVE: The number of the study.
  CYCLE: How many minutes are to be overwritten for every 1 minute kept. If 0, all data is used
"""


import filtering
import interact_tools as it
import pandas as pd
from collections import Counter, OrderedDict
from accel_profiling_working import fit_and_plot_activity
import numpy as np
from sys import argv


if __name__ == "__main__":
    city, wave = it.get_command_args("top_generation.py")
    cycle = int(argv[3])
    participants = it.top_iid_list(city, wave)

    # The number of breakpoints to use in each cycle, as well as where to start looking for breakpoints
    start_point_sets = {
        2: [100, ],
        3: [100, 1951],
        4: [100, 1951, 5725],
    }

    point_bound_sets = {
        2: [[20, 2000], ],
        3: [[20, 200], [1500, 2500]],
        4: [[20, 200], [1500, 2500], [4000, 7000]],
    }
    binsize = 100

    out_fname = "pa_specificity_" + city + "_" + str(wave) + "_skip-" + str(cycle) + "-minutes.csv"
    results = pd.DataFrame()
    for iid in participants:
        new_row = pd.Series()
        new_row['iid'] = iid

        original_counts = it.top_counts(iid, utcas='utcdate')
        original_counts['utcdate'] = pd.to_datetime(original_counts['utcdate'])
        # Ensure the participant has sufficient data
        original_counts = filtering.filter_data(original_counts)
        if original_counts.empty:
            continue

        for pbs in point_bound_sets:
            start_points = start_point_sets[pbs]
            point_bounds = point_bound_sets[pbs]
            # Overwrite the appropriate number of minutes, then resample to the 1-minute level
            counts = filtering.skip_data(original_counts, cycle)
            minute_counts = counts.set_index('utcdate').resample('1min').sum().dropna().counts.tolist()
            data = OrderedDict(sorted(Counter(np.round(np.array(it.trim_10k(minute_counts)) / binsize) * binsize).items()))
            x = list(data.keys())
            y = list(data.values())
            # Plot the data and fit the function appropriately
            line_info = fit_and_plot_activity(x, y, point_bounds, start=start_points)

            # Store the data in the appropriate columns
            for i in range(1, 1 + pbs):
                if i == pbs:
                    line_info[i + pbs] = line_info[i + pbs]
                    new_row[str(pbs) + '-seg_tail'] = max(x)
                else:
                    new_row[str(pbs) + '-seg_breakpoint_' + str(i)] = line_info[i]
                new_row[str(pbs) + '-seg_slope_' + str(i)] = line_info[i + pbs]
            new_row[str(pbs) + '-seg_r_square'] = round(line_info[1 + 3*pbs], 5)
        if len(new_row.tolist()) > 3:
            results = results.append(new_row, ignore_index=True)
    results['best_r_square'] = results[[str(x) + '-seg_r_square' for x in point_bound_sets]].idxmax(axis=1).str[0]
    results['iid'] = results.iid.astype(int)
    results.set_index('iid').to_csv(out_fname)

