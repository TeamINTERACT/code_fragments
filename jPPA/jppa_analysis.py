import pandas as pd
import numpy as np
import math
import os, sys, pickle
from ast import literal_eval
from ppa_jppa_stable_trip_det import city_kdtree


def jppa_per_trip(part_trip_path, bus_days_path, city_tree, city_points, bus_stop_dictionary_dir):
    """
    Compute dataframe indicating whether a participant, during a specific trip, intersected the path of a bus route
        at a given time. Rows are timestamps and columns are bus route ids. Entries are 1s if they intersect, otherwise.
    :param part_trip_path: path to participant trip csv file
    :param bus_days_path: path to directory containing 7 sub directories, one for each day of the week, which contain
                            the directories of routes which operate on that day
    :param city_grid: city_grid dataframe (not the path)
    :param bus_stop_dictionary_dir: path to directory containing 7 pickle files -- dictionaries mapping location
                                    tuples to a list of route_ids

    :return: dataframe indexed from the start time to the end time of the participant trip with column names
                being names of bus routes that are relevant to that participant (within 600m of the starting point of
                the trip and 600m within the ending point of the trip). Entries in the dataframe are 1s (participant
                path intersects that of the bus route at that point in time) and 0s (participant path does not intersect
                that of the bus route at that point in time)
    """

    def compute_jppa(row):
        try:
            part_cells = participant.at[row.name, 'cells']
            bus_cells = one_bus_trip.at[row.name, 'cells']  # this step may raise an error if there is no entry in
            #   one_bus_trip at row_name
            if part_cells.intersection(bus_cells) == set():
                if jppa.at[row.name, route_id] == 1:
                    return 1
                else:
                    return 0
            else:
                print('Found intersection with trip', one_bus_trip.at[row.name, 'trip_id'], route_id, 'at', row.name)
                return 1
        except:  # if there is no entry in one_bus_trip at row_name
            if jppa.at[row.name, route_id] == 1:
                return 1
            else:
                return 0

    def compute_probabilities(col):
        # keep slice starting from first in_main timestamp to last in_main timestamp
        # col = col.loc[: col[(col['in_main'] == 1)].index[-1], :]
        vals = col.values
        idx = [index for index, val in enumerate(vals) if val == 1]  # get indices of occurences of 1

        k = 2 # must potentially be on a bus for at least 3 minutes -- preferably more when using Saskatoon data

        if len(vals[idx[0]: (idx[-1] + 1)]) > k:
            p = sum(vals[idx[0] : (idx[-1] + 1)]) / len(vals[idx[0] : (idx[-1] + 1)])
            return p

    # get interact_id
    part_path_separated = part_trip_path.split('/')
    for i in part_path_separated:
        try:
            x = int(i)
            if len(str(x)) == 9:
                interact_id = x
                break
        except:
            continue

    participant = pd.read_csv(part_trip_path, usecols=['utc_date', 'cells'], parse_dates=['utc_date'])
    date = participant.at[0, 'utc_date'].date().strftime('%Y-%m-%d')

    stamp = pd.Timestamp(date)
    day_of_week = stamp.day_name().lower()
    day_of_week = 'monday'  # TODO: remove this line as it artificially sets day to monday

    # initialize jppa dataframe for this trip
    jppa = pd.DataFrame(participant['utc_date'])
    jppa = jppa.set_index('utc_date')
    participant = participant.set_index('utc_date')

    # read in the cells column as sets instead of strings
    # some entries are set() instead of {...}
    participant['cells'] = participant['cells'].apply(lambda x: literal_eval(x) if x[0] == '{' else set())

    # TODO: filter out irrelevant bus routes HERE

    starting_point = participant.iloc[0]['cells'].pop() # (northing, easting) tuple
    ending_point = participant.iloc[-1]['cells'].pop() # (northing, easting) tuple

    radius = 600  # in meters -- distance travelled at 2m/s if travelling for 5 mins
    print('done with the setup, now querying')
    indices_start = city_tree.query_ball_point([starting_point[0], starting_point[1]], radius)  # return indices of points within radius
    coordinates_start = [city_points[m] for m in indices_start]  # list of actual coordinate tuples

    indices_end = city_tree.query_ball_point([ending_point[0], ending_point[1]], radius)
    coordinates_end = [city_points[n] for n in indices_end]

    coordinates_total = coordinates_start + coordinates_end

    print('reading in routes dictionary')
    with open(bus_stop_dictionary_dir + '/' + day_of_week + '.pickle', 'rb') as handle:
        routes_dict = pickle.load(handle)

    print('finding relevant routes')
    relevant_routes = [route for coord in coordinates_total if coord in routes_dict for route in routes_dict[coord]]
    # remove duplicates -- does not preserve order
    relevant_routes = list(set(relevant_routes))
    print(relevant_routes)

    for route_id in relevant_routes:
        one_bus_route_dir = bus_days_path + '/' + day_of_week + '/' + route_id
        one_bus_route_dir_encoded = os.fsdecode(one_bus_route_dir)
        jppa[route_id] = 0

        print('beginning work on route', route_id)
        for file in os.listdir(one_bus_route_dir_encoded):
            filename = os.fsdecode(file)
            one_bus_trip_path = one_bus_route_dir + '/' + filename

            one_bus_trip = pd.read_csv(one_bus_trip_path, parse_dates=['time'])

            # change date of bus trip from 1900-01-01 to date of participant data
            start_datetime = one_bus_trip.at[0, 'time']
            start_time = start_datetime.time()
            start_time= start_time.strftime('%H:%M:%S')

            new_start_datetime = date + ' ' + start_time

            times = pd.date_range(new_start_datetime, periods=len(one_bus_trip), freq='1min')
            times_series = pd.Series(times)
            one_bus_trip['utc_date'] = times_series

            one_bus_trip = one_bus_trip.set_index('utc_date')
            one_bus_trip = one_bus_trip.drop('time', axis=1)
            one_bus_trip['cells'] = one_bus_trip['cells'].apply(lambda x: literal_eval(x))  # interpret set from string format back to set
            jppa[route_id] = jppa.apply(compute_jppa, axis=1)

        if not 1 in jppa[route_id].values:
            jppa.drop(route_id, axis=1, inplace=True)

    probabilities = jppa.apply(compute_probabilities).values
    probabilities = [x for x in probabilities if not math.isnan(x)]
    p = max(probabilities)

    trip_start_time = jppa.index.values[0]
    data = {'interact_id': interact_id, 'trip_start_time' : [trip_start_time], 'probability': [p]}
    jppa = pd.DataFrame(data)

    return jppa

