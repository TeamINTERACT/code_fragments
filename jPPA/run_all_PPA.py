from ppa_jppa_stable_trip_det import ppa_person
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import os


participant_directory = '../jppa_participant_dfs/victoria/preprocessed/every5'
participant_directory = os.fsdecode(participant_directory)
trips_dwells_directory = '../jppa_participant_dfs/victoria/preprocessed/stable_trip_det_every5_trips_dwells'
trips_dwells_directory = os.fsdecode(trips_dwells_directory)

cost_histogram_path = '../jppa_participant_dfs/victoria/histograms/histogram_ANALYSIS_every5_sd_wave1_15_625_cost'

def wrap_func(participant_path, trips_dwells_path):
    ppa_person(participant_path, cost_histogram_path, trips_dwells_path, version='per_trip', vmax=2)

parallelize = True

if parallelize:
    part_path_l = [participant_directory + '/' + os.fsdecode(file) for file in os.listdir(participant_directory)]
    trips_dwells_path_l = [trips_dwells_directory + '/' + os.fsdecode(file)  + '_trips_dwells' for file in os.listdir(participant_directory)]

    with PoolExecutor(max_workers=2) as executor:
        for _ in executor.map(wrap_func, part_path_l, trips_dwells_path_l):
            pass



else:
    for file in os.listdir(participant_directory):
        filename = os.fsdecode(file)
        part_path = participant_directory + '/' + filename
        trips_dwells_path = trips_dwells_directory + '/' + filename + '_trips_dwells'

        ppa_person(part_path, cost_histogram_path, trips_dwells_path, version='per_trip', vmax=2)


