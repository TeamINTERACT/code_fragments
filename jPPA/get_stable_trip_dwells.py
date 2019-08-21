"""
Author: Antoniu Vadan, summer 2019
Description: extract trips and dwells for all participants
"""
import os
import pandas as pd
from spatial_metrics import get_grid_MODIFIED, movement_detection


directory = '../jppa_participant_dfs/victoria/preprocessed/every5'
directory = os.fsdecode(directory)
res = 15.625

i = 1

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print('Participant number:', i)

    part_path = directory + '/' + filename
    part_df = pd.read_csv(part_path, parse_dates=['utc_date'])

    part_df['grid_cell'] = get_grid_MODIFIED(part_df['easting'], part_df['northing'], res)

    trips_dwells = movement_detection(part_df, 3)  # 3 corresponds to being at the same location for at least 15 minutes

    trips_dwells.to_csv('../jppa_participant_dfs/victoria/preprocessed/stable_trip_det_every5_trips_dwells/' + filename + '_trips_dwells')
    i += 1
