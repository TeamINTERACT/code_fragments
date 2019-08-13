"""
Author: Antoniu Vadan
Description: Functions to compute Potential Path Area (PPA) and joint Potential Path Area (jPPA)
"""

import osmnx as ox
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy import spatial
import math
import datetime
import time
from street_network import points_along_path
from astar import astar
import networkx as nx
import sys
import dask.dataframe as dd
from shapely.geometry import Point, LineString, mapping
import os
from threading import Lock

"""
Description:
    - in this version of jPPA, we will compute cross-sectional PPA pixels for every minute in which
        we lack participant data
    - the plan is to use ArcGIS/arcpy to interpolate
        location of buses between stops accurately
"""


# df_bus = pd.read_pickle('bus_saskatoon_6jan_27april/df_main2')
# df_participant = pd.read_pickle('../jppa_participant_dfs/saskatoon_ethica_gps_250m_1min')


def city_kdtree(city_grid):
    """
    create KDTree with city coordinates -- needed for computation of overlap of circles
    :param city_grid: pandas dataframe of city grid -- contains 1s (in city) and np.NaNs
    :return: kdtree composed of city coordinates
    """
    city_grid_rows = list(city_grid.index.values)
    city_grid_cols = list(city_grid.columns.values)
    res = city_grid_cols[1] - city_grid_cols[0]

    rows, cols = np.mgrid[city_grid_rows[0]:city_grid_rows[-1]+1:res,
                 city_grid_cols[0]:city_grid_cols[-1]+1:res]
    points = list(zip(rows.ravel(), cols.ravel()))
    tree = spatial.KDTree(points)  # city KDTree
    return tree, points


def interpolate_along_path(start, end, n, histogram):
    """
    Adapted from https://stackoverflow.com/a/35025274

    Given 2 points, compute shortest path. Along that path, get n evenly spaced points
    :param start: starting point (northing, easting)
    :param end: destination (northing, easting)
    :param n: number of evenly spaced points BETWEEN starting point and destination
    :param histogram: probability distribution of participant location, as generated in pre_trip_detection.py
    :return: list of (northing, easting) tuples representing n + 2 points along path
    """

    cols = list(histogram.columns.values)
    res = cols[1] - cols[0]

    t5 = time.time()
    path = astar(histogram, start, end)
    t6 = time.time()
    print(str(datetime.timedelta(seconds=(t6 - t5))), '------- A*')
    path_points = [Point(x) for x in path]

    path_line = LineString(path_points)

    def redistribute_vertices(geom, num_vert):
        if geom.geom_type == 'LineString':
            if num_vert == 0:
                num_vert = 1
            num_vert += 1  # so that we get <num_vert> vertices IN BETWEEN p1 and p2
            return LineString(
                [geom.interpolate(float(i) / num_vert, normalized=True)
                 for i in range(num_vert + 1)])
        else:
            raise ValueError('unhandled geometry %s', (geom.geom_type,))

    multiline_r = redistribute_vertices(path_line, n)
    map_dict = mapping(multiline_r)  # dictionary. 'type':'LineString', 'coordinates':((.....))
    multiline_r = list(map_dict['coordinates'])  # list of coordinate tuples unrounded
    multiline_r = [((round(x[0] / res) * res), (round(x[1] / res) * res)) for x in multiline_r]  # (northing, easting) tuples

    return multiline_r


def jppa_person_person(df_part1, df_part2):
    """
    Compute dataframe indexed by timestamp and with three columns
    :param part1: PPA dataframe of participant with interact id 1 -- contains utc_date as index and one 'cells' column
    :param part2: PPA dataframe of participant with interact id 2 -- contains utc_date as index and one 'cells' column
    :return: dataframe indexed by timestamp and with three columns
             Two columns are interact_id_1 and interact_id_2
             One column is "cells" representing intersecting cells at a given timestamp
    """

    # Find max starting time and min ending time
    part1_start = df_part1.index.values[0]  # starting datetime of first period
    part1_end = df_part1.index.values[-1]   # ending datetime of first period
    part2_start = df_part2.index.values[0]  # starting datetime of second period
    part2_end = df_part2.index.values[-1]   # ending datetime of second period

    # convert all to pandas datetimes
    part1_start = pd.to_datetime(str(part1_start))
    part2_start = pd.to_datetime(str(part2_start))
    part1_end = pd.to_datetime(str(part1_end))
    part2_end = pd.to_datetime(str(part2_end))

    df_start_end = pd.DataFrame([[part1_start, 'start'], [part2_start, 'start'],
                                [part1_end, 'end'], [part2_end, 'end']],
                                columns=['utc_date', 'start/end'])

    # from df_start_end, create dataframe with start and end dates labelled as max and min
    # we want to create a time range from the latest starting period to earliest ending period
    #   to evaluate overlap
    df_start_end = df_start_end.groupby('start/end')['utc_date'].agg({'earliest':'min', 'latest':'max'})

    starting = df_start_end.at['start', 'latest']
    ending = df_start_end.at['end', 'earliest']

    time_range = pd.date_range(starting, ending + pd.Timedelta(days=1),  # this line corrects the range --
                                        freq='T')                   # otherwise it does not include last day

    # trim ends of time_range -- e.g. it starts at time 00:00:00 while participant data starts at 04:02:00
    for i, t in  enumerate(time_range):
        if t == starting:
            time_range = time_range[i:]
            break

    for i, t in reversed(list(enumerate(time_range))):
        if t == ending:
            time_range = time_range[0:i+1]
            break

    df_jppa = pd.DataFrame(time_range, columns=['utc_date'])
    df_jppa.set_index('utc_date', inplace=True)
    df_jppa['cells'] = np.NaN

    def f(row):
        return df_part1.at[row.name, 'cells'].intersection(df_part2.at[row.name, 'cells'])

    df_jppa['cells'] = df_jppa.apply(f, axis=1)
    return df_jppa


def ppa_person(df_part_csv_path, df_city_path, trips_dwells_path=None, version=1, road_graph_path=None,
               vmax=2, vwalk=2, pre_trip_detection=False):
    """
    Compute dataframe -- timestamp is every minute of period with participant data as index -- with
        one column 'cells' containing either 1 set of easting and northing (if a recording was taken
        at that time) or a set of tuples (PPA at that time)
    Dataframe represents a person's PPA
    Different versions of PPA are documented below
    May also interpolate location when pre_trip_detection=True. In that case, only df_part_csv_path, df_city_path
    and road_graph_path also need to be provided.

    :param df_part_csv_path: path to participant dataframe csv file -- not indexed by timestamp
        - this dataframe is preprocessed by participant_df_preprocessing.py
    :param df_city_path: path to pickled pandas dataframe of city grid -- contains 1s (in city) and np.NaNs
                      OR path to csv file of city histogram which will be used for pre_trip_detection
                      - created in pre_trip_detection.py
    :param trips_dwells_path: path to participant visits AND dwells dataframe as computed by spatial_metrics.py
    :param version: version 1: assumption: people do not wander at all during their dwells
                    version 2: assumption: people wander in between the recordings at walking speed
                    version 3: assumption: people can wander from the beginning of the dwell until the end
                                            at walking speed (ignoring the measurements in between)
    :param road_graph_path: path to gpickled road network of city
    :param vmax: maximum speed of person when travelling through the city in meters/second
    :param vwalk: maximum speed of person when walking
    :param pre_trip_detection: returns dataframe with minute-wise timestamps for entire duration of study
                        containing easting and northing interpolated (using the a* algorithm) for
                        times when it is not recorded.
                        Gaps longer than 60 minutes are not filled
                        ***IMPORTANT***: When pre_trip_detection=True, df_city_path is the path to the dataframe
                                        where each entry represents the cost of stepping into that cell

    :return: dataframe with timestamp as index and cells as column
    """

    def f1(row):
        """
        Function to apply on df_times to create 'in_participant' column
        """
        if row['utc_date'] in part_times:
            return True
        else:
            return False

    def f1_inner(row, part_times_inner, dummy):
        """
        Function to apply on df_times to create 'in_participant' column
        """
        if row['utc_date'] in part_times_inner:
            return True
        else:
            return False

    def f2(row):
        """
        Function to apply on df_times to create 'location_tuples_e_n' column
        If a timestamp is contained in the participant dataframe, location_tuples_e_n is a tuple
            of the easting and northing of the location of the participant at that time
        Otherwise, it contains a tuple of tuples -- first tuple contains the easting and northing of last
            recorded location, along with how many minutes away it was recorded
            Second tuple contains the same data for the second point
        """

        if row['cells']:  # check if set is not empty
            return row['cells']

        j = 1
        k = 1

        # after how many minutes to stop computing PPA and interpolating path
        threshold = 15

        # find how far a timestamp (which is not also in the participant dataframe),
        #   is from a timestamp which IS in the participant dataframe
        while not df_times.at[row.name - datetime.timedelta(minutes=j), 'cells']:  # check if set is empty
            j += 1
            if j == threshold:  # set threshold: if data points are more than an hour apart, return None
                return

        while not df_times.at[row.name + datetime.timedelta(minutes=k), 'cells']:
            k += 1
            if k == threshold:
                return

        if j + k >= threshold:
            return

        t1 = row.name - datetime.timedelta(minutes=j)
        t2 = row.name + datetime.timedelta(minutes=k)
        # keep track of the coordinates of the point whose distance is stored
        temp1 = list(df_times.loc[t1, 'cells'])[0]
        temp2 = list(df_times.loc[t2, 'cells'])[0]

        return (temp1[0], temp1[1], j), (temp2[0], temp2[1], k)


    def f2_inner(row, times, dummy):
        """
        Function to apply on df_times to create 'location_tuples_e_n' column
        If a timestamp is contained in the participant dataframe, location_tuples_e_n is a tuple
            of the easting and northing of the location of the participant at that time
        Otherwise, it contains a tuple of tuples -- first tuple contains the easting and northing of last
            recorded location, along with how many minutes away it was recorded
            Second tuple contains the same data for the second point
        """

        if row['cells']:  # check if set is not empty
            return row['cells']

        j = 1
        k = 1

        # after how many minutes to stop computing PPA and interpolating path
        threshold = 15

        # find how far a timestamp (which is not also in the participant dataframe),
        #   is from a timestamp which IS in the participant dataframe
        while not times.at[row.name - datetime.timedelta(minutes=j), 'cells']:  # check if set is empty
            j += 1
            if j == threshold:  # set threshold: if data points are more than an hour apart, return None
                return

        while not times.at[row.name + datetime.timedelta(minutes=k), 'cells']:
            k += 1
            if k == threshold:
                return

        if j + k >= threshold:
            return

        t1 = row.name - datetime.timedelta(minutes=j)
        t2 = row.name + datetime.timedelta(minutes=k)
        # keep track of the coordinates of the point whose distance is stored
        temp1 = list(times.loc[t1, 'cells'])[0]
        temp2 = list(times.loc[t2, 'cells'])[0]

        return (temp1[0], temp1[1], j), (temp2[0], temp2[1], k)

    def f2_dwells(row):
        """
        Similar to f2. The difference is that loc_tuples_e_n exists for durations longer than 60mins
            (as dwells may take longer)
        """
        if row['cells']:  # check if set is not empty
            return row['cells']

        j = 1
        k = 1
        # find how far a timestamp (which is not also in the participant dataframe),
        #   is from a timestamp which IS in the participant dataframe
        while not df_times.at[row.name - datetime.timedelta(minutes=j), 'cells']:  # check if set is empty
            j += 1

        while not df_times.at[row.name + datetime.timedelta(minutes=k), 'cells']:
            k += 1

        t1 = row.name - datetime.timedelta(minutes=j)
        t2 = row.name + datetime.timedelta(minutes=k)
        # keep track of the coordinates of the point whose distance is stored
        temp1 = df_part.loc[df_part['utc_date'] == t1].reset_index()
        temp2 = df_part.loc[df_part['utc_date'] == t2].reset_index()

        return (temp1.at[0, 'northing'], temp1.at[0, 'easting'], j), \
               (temp2.at[0, 'northing'], temp2.at[0, 'easting'], k)


    def f3(row):
        """
        Insert known location of participant in df_times
        """
        if row['in_participant']:
            result = set()
            temp = df_part.loc[df_part['utc_date'] == row.name].reset_index()
            result.add((temp.at[0, 'northing'], temp.at[0, 'easting']))
            return result
        else:
            result = set()
            return result

    def set_easting(row):  # exception handles the case in which the first cells in a participant dataframe
                           #    represent a dwell
                           # reason: querying result_paths[result_paths['end_time'] == row['start_time']]
                           #    results in an error because there is no such entry if the first dwell is at the
                           #    beginning of the dataframe
        # try:
        #     easting = result_paths[result_paths['end_time'] == row['start_time']]['path_end_x']
        #     easting = float(easting)  # step is included because now easting is a Pandas Series object
        #     easting = round(easting / res) * res
        #     df_times.loc[row['start_time']:row['end_time'], 'temp_easting'] = easting
        # except:

        easting = list(df_times.at[row['start_time'], 'cells'])
        easting = easting[0][1]
        easting = float(easting)
        easting = round(easting / res) * res
        df_times.loc[row['start_time']:row['end_time'], 'temp_easting'] = easting

    def set_northing(row):
        # try:
        #     northing = result_paths[result_paths['end_time'] == row['start_time']]['path_end_y']
        #     northing = float(northing)  # step is included because now easting is a Pandas Series object
        #     northing = round(northing / res) * res
        #     df_times.loc[row['start_time']:row['end_time'], 'temp_northing'] = northing
        # except:
        northing = list(df_times.at[row['start_time'], 'cells'])
        northing = northing[0][0]
        northing = float(northing)
        northing = round(northing / res) * res
        df_times.loc[row['start_time']:row['end_time'], 'temp_northing'] = northing

    def combine_e_n(row):
        if not math.isnan(row['temp_easting']):
            row['cells'].clear()
            row['cells'].add((row['temp_northing'], row['temp_easting']))

    def filter_long_dwells(row):
        if (row['end_time'] - row['start_time']) / np.timedelta64(1, 'm') > 45:  # equivalent to three trips, according to Luana's logic
            df_times.loc[row['start_time']:row['end_time'], 'cells'] = set()

    def f4(row):
        """
        Compute circle intersections
        """
        if not row['cells'] and row['loc_tuples_e_n'] is not None:
            point1 = row['loc_tuples_e_n'][0]  # tuple of tuples with location data
            point2 = row['loc_tuples_e_n'][1]

            r1 = point1[2] * 60 * vmax  # point1[2] indicates how many timestamps away (back in time) is the next
            r2 = point2[2] * 60 * vmax  # measurement taken; point2[2] looks at future timestamp
            indices1 = tree.query_ball_point([point1[0], point1[1]], r1)  # return indices of points within radius
            indices2 = tree.query_ball_point([point2[0], point2[1]], r2)
            coords1 = [points[m] for m in indices1]  # list of actual coordinates
            coords2 = [points[m] for m in indices2]

            set_coords1 = set(coords1)  # convert to set for (assumed) faster intersections (not proven)
            set_coords2 = set(coords2)

            return set_coords1.intersection(set_coords2)
        else:
            return row['cells']

    def f4_walking(row):
        """
        Compute circle intersections with walking speed
        """
        if not row['cells'] and row['loc_tuples_e_n'] is not None:
            point1 = row['loc_tuples_e_n'][0]  # tuple with location data
            point2 = row['loc_tuples_e_n'][1]

            r1 = point1[2] * 60 * vwalk  # point1[2] indicates how many timestamps away (back in time) is the next
            r2 = point2[2] * 60 * vwalk  # measurement taken; point2[2] looks at future timestamp
            indices1 = tree.query_ball_point([point1[0], point1[1]], r1)  # return indices of points within radius
            indices2 = tree.query_ball_point([point2[0], point2[1]], r2)
            coords1 = [points[m] for m in indices1]  # list of actual coordinates
            coords2 = [points[m] for m in indices2]

            set_coords1 = set(coords1)  # convert to set for (assumed) faster intersections (not proven)
            set_coords2 = set(coords2)

            return set_coords1.intersection(set_coords2)
        else:
            return row['cells']

    def small_wine_glass(row):
        """
        assumption: people wander in between the recordings at walking speed
        This function computes intersection of circles between timestamps within dwells
        """
        start = row['start_time']
        end = row['end_time']
        columns = list(df_times.columns.values)
        # get location tuples for the dwell period
        df_times.loc[start:end, 'loc_tuples_e_n'] = \
            df_times.loc[start:end, columns].apply(f2, axis=1)
        df_times.loc[start:end, 'cells'] = \
            df_times.loc[start:end, columns].apply(f4_walking, axis=1)

    def large_wine_glass(row):
        """
        assumption: people can wander from the beginning of the dwell until the end
                    at walking speed (ignoring the measurements in between)
        This function computes intersection of circles between timestamps which bound dwells
        Results in one large space-time prism (STP)
        """
        start = row['start_time']
        end = row['end_time']
        shifted_start = start + pd.Timedelta(minutes=1)
        shifted_end = end - pd.Timedelta(minutes=1)
        columns = list(df_times.columns.values)
        # clear all sets in the cells column between start and end
        #   as f2_dwells relies on them being empty
        df_times.loc[shifted_start:shifted_end, columns].apply(lambda x: x['cells'].clear(), axis=1)
        df_times.loc[start:end, 'loc_tuples_e_n'] = \
            df_times.loc[start:end, columns].apply(f2_dwells, axis=1)
        df_times.loc[start:end, 'cells'] = \
            df_times.loc[start:end, columns].apply(f4_walking, axis=1)

    def fill_in(row):
        """
        For each gap in the participant dataframe, use street_network.py function to interpolate
        """
        if not row['cells'] and row['loc_tuples_e_n'] is not None:
            point1_data = row['loc_tuples_e_n'][0]  # tuple with location data; e.g. (northing, easting, n)
            point2_data = row['loc_tuples_e_n'][1]
            start = (point1_data[0], point1_data[1])
            end = (point2_data[0], point2_data[1])
            if start == end:
                return {start}
            n1 = point1_data[2]
            n2 = point2_data[2]
            path = points_along_path(start, end, graph_utm, (n1 + n2 - 1))
            return {(int(round(path[n1][1]) / res) * res, int(round(path[n1][0]) / res) * res)}  # reverse easting and
            # return {(path[n1][1], path[n1][0])}
        else:
            return row['cells']

    def fill_in_astar(row, memo_dict, dummy):
        """
        For each gap shorter than an hour in the participant dataframe, use A* to interpolate location
        """
        if not row['cells'] and row['loc_tuples_e_n'] is not None:
            point1_data = row['loc_tuples_e_n'][0]  # tuple with location data; e.g. (northing, easting, n)
            point2_data = row['loc_tuples_e_n'][1]
            start = (point1_data[0], point1_data[1])
            end = (point2_data[0], point2_data[1])
            if start == end:
                return {start}
            n1 = point1_data[2]
            n2 = point2_data[2]

            print(memo_dict)
            print(len(memo_dict))
            if (start, end, (n1 + n2 -1)) in memo_dict:
                path = memo_dict[(start, end, (n1 + n2 -1))]
                return {(path[n1][0], path[n1][1])}

            path = interpolate_along_path(start, end, (n1 + n2 -1), df_city)
            memo_dict[(start, end, (n1 + n2 -1))] = path
            return {(path[n1][0], path[n1][1])}
        else:
            return row['cells']

    def extract_n(row):
        """Apply function after the fill_in function. This function gets easting and northing
        from cells column in separate columns as preparation for trip_detection"""
        if row['cells']:
            coords = list(row['cells'])[0]
            return int(coords[0])

    def extract_e(row):
        """Apply function after the fill_in function. This function gets easting and northing
        from cells column in separate columns as preparation for trip_detection"""
        if row['cells']:
            coords = list(row['cells'])[0]
            return int(coords[1])

    def apply_fill_in(df):
        return df.apply((lambda row: fill_in(row)), axis=1)

    def apply_fill_in_astar(df):
        return df.apply((lambda row: fill_in_astar(row)), axis=1)

    def get_df_times(df):
        # time range from the beginning of
        time_range = pd.date_range(df.loc[df.index[0], 'utc_date'].strftime('%Y-%m-%d'),
                                   df.loc[df.index[-1], 'utc_date']
                                   + pd.Timedelta(days=1),  # this line corrects the range -- otherwise it does not
                                   freq='T')  # include last day

        start_time = df.at[0, 'utc_date']
        end_time = df.at[df.index[-1], 'utc_date']

        # trim ends of time_range -- e.g. it starts at time 00:00:00 while participant data starts at 04:02:00
        for i, t in enumerate(time_range):
            if t == start_time:
                time_range = time_range[i:]
                break

        for i, t in reversed(list(enumerate(time_range))):
            if t == end_time:
                time_range = time_range[0:i + 1]
                break

        part_times_list = df['utc_date'].tolist()
        part_times_set = set(part_times_list)  # set used for f1
        times = pd.DataFrame(time_range, columns=['utc_date'])
        return times, part_times_set


    versions = (1, 2, 3, 'per_trip')

    if version not in versions:
        raise ValueError('Existing versions: ' + str(versions))

    df_part = pd.read_csv(df_part_csv_path, parse_dates=['utc_date'])

    # TODO: **********++++++++++ Remove following line when NOT testing -- takes every 5th entry of participant data ++++++++++**********
    # df_part = df_part.iloc[0:68, :]
    # print(df_part.to_string())
    # sys.exit()

    try:
        df_city = pd.read_pickle(df_city_path)
    except:
        df_city = pd.read_csv(df_city_path, dtype=np.float)
        df_city = df_city.set_index('Unnamed: 0')
        df_city.columns = df_city.columns.astype(float)
        df_city.index = df_city.index.astype(float)

    cols = list(df_city.columns.values)
    res = cols[1] - cols[0]

    pd.to_numeric(df_part['easting'])
    pd.to_numeric(df_part['northing'])

    # if pre_trip_detection:
    #     graph_utm = nx.read_gpickle(road_graph_path)  # needed for the fill_in function
    #     print('getting location tuples')
    #     df_times['loc_tuples_e_n'] = df_times.apply(f2, axis=1)
    #     print('filling in cells using road network')
    #     # df_times['cells'] = df_times.apply(fill_in, axis=1)  # without parallelization
    #     ddata = dd.from_pandas(df_times, npartitions=16)
    #     df_times['cells'] = ddata.map_partitions(apply_fill_in).compute(scheduler='processes')
    #
    #     # extract Easting and Northing in separate columns for trip detection
    #     print('separating eastings and northings')
    #     df_times['utm_n'] = df_times.apply(extract_n, axis=1)
    #     df_times['utm_e'] = df_times.apply(extract_e, axis=1)
    #     print('dropping NaNs')
    #     # remove times with no data for location (times when location is missing for longer than 60 mins)
    #     df_times = df_times.dropna()
    #     return df_times

    if pre_trip_detection:
        df_times, part_times = get_df_times(df_part)

        print('Creating in_participant column')
        df_times['in_participant'] = df_times.apply(f1, axis=1)  # temporary column
        df_times.set_index('utc_date', inplace=True)
        print('Inserting known locations')
        df_times['cells'] = df_times.apply(f3, axis=1)

        print('getting location tuples')
        df_times['loc_tuples_e_n'] = df_times.apply(f2, axis=1)
        # print(df_times.iloc[0:100].to_string())
        # print(df_times.iloc[9900:10200, :].to_string())
        # sys.exit()
        print('filling in cells using astar')

        df_times['cells'] = df_times.apply(fill_in_astar, axis=1)  # without parallelization
        # ddata = dd.from_pandas(df_times, npartitions=16)
        # df_times['cells'] = ddata.map_partitions(apply_fill_in_astar).compute(scheduler='processes')

        # extract Easting and Northing in separate columns for trip detection
        print('separating eastings and northings')
        df_times['utm_n'] = df_times.apply(extract_n, axis=1)
        df_times['utm_e'] = df_times.apply(extract_e, axis=1)
        print('dropping NaNs')
        # remove times with no data for location (times when location is missing for longer than 60 mins)
        df_times = df_times.dropna()
        return df_times

    trips_dwells = pd.read_csv(trips_dwells_path, parse_dates=['start_time', 'end_time'])
    result_visits = trips_dwells[trips_dwells['trip_length'] == 1]
    result_visits= result_visits[result_visits['duration'] != 1]
    result_trips = trips_dwells[trips_dwells['trip_length'] > 1]
    # print(trips_dwells.iloc[0:20, :].to_string())
    # print(result_visits.iloc[0:20, :].to_string())
    # print(result_trips.iloc[0:3, :].to_string())
    # result_trips = result_trips.iloc[0:3, :]

    # sys.exit()

    def process_trips(row):
        trip_start_time = row['start_time']
        trip_end_time = row['end_time']
        difference = (trip_end_time - trip_start_time) / np.timedelta64(1, 'm')
        if difference < 10:
            return

        lock = Lock()

        lock.acquire()  # will block if lock is already held
        print("something")
        lock.release()

        t1 = time.time()
        trip = df_part.loc[(df_part['utc_date'] >= row['start_time']) & (df_part['utc_date'] <= row['end_time'])]
        trip = trip.reset_index()
        df_times, part_times_inner = get_df_times(trip)
        part_times_argument = (part_times_inner, None)
        print('Creating in_participant column')
        df_times['in_participant'] = df_times.apply(f1_inner, args=part_times_argument, axis=1)  # temporary column
        df_times.set_index('utc_date', inplace=True)
        print('Inserting known locations')
        df_times['cells'] = df_times.apply(f3, axis=1)
        print('Get location of nearest two measurements in time')
        df_times_argument = (df_times, None)
        df_times['loc_tuples_e_n'] = df_times.apply(f2_inner, args=df_times_argument, axis=1)
        # print('Computing PPA for remaining time steps')
        df_times['cells'] = df_times.apply(f4, axis=1)
        print('Interpolating location for missing intervals')
        memo = dict()
        memo_argument = (memo, None)
        df_times['cells'] = df_times.apply(fill_in_astar, args=memo_argument, axis=1)
        memo.clear()

        df_times = df_times.drop(['in_participant', 'loc_tuples_e_n'])

        date = trip.at[0, 'utc_date'].date().strftime('%Y-%m-%d')
        time_day = trip.at[0, 'utc_date'].time().strftime('%H:%M:%S').replace(':', '_')

        interact_id = str(trip.at[0, 'interact_id'])
        main_dir = '../jppa_participant_dfs/victoria/preprocessed/daily_ppa_parallel_test'

        if not os.path.exists(main_dir + '/'+ interact_id):
            os.makedirs(main_dir + '/'+ interact_id)

        if not os.path.exists(main_dir + '/'+ interact_id + '/' + date):
            os.makedirs(main_dir + '/'+ interact_id + '/' + date)

        path = main_dir + '/'+ interact_id + '/' + date
        df_times.to_csv(path + '/' + 'start_time_' + time_day)

        t2 = time.time()
        print(str(datetime.timedelta(seconds=(t2 - t1))), ' -- one trip for participant', interact_id)
        print('++++++++++++++++++++++++++++++++++++++++ TRIP SAVED **************************************************')
        print('++++++++++++++++++++++++++++++++++++++++**************************************************++++++++++++')

        # TODO: remove following commented code: it is for plotting the PPA
        # all_cells = df_times['cells']
        # all_cells = all_cells.to_list()
        #
        # all_cells_lists = [list(x) for x in all_cells if len(x) != 0]
        # eastings = [float(y[1]) for x in all_cells_lists for y in x]
        # northings = [float(y[0]) for x in all_cells_lists for y in x]
        #
        # graph = nx.read_gpickle('graphs/victoria_not_simplified_road_utm_undirected')
        #
        # fig, ax = ox.plot_graph(graph, show=False, close=False)
        # plt.scatter(eastings, northings, c='red', s=30)
        # plt.show()
        #
        # print(df_times.to_string())
        # sys.exit()

    # set up KD Tree -- will be used for finding intersection of circles
    tree, points = city_kdtree(df_city)

    if version == 'per_trip':
        result_trips.apply(process_trips, axis=1)

    elif version == 1:  # when dwelling, assume person is not wandering at all
        df_times, part_times = get_df_times(df_part)

        print('Creating in_participant column')
        df_times['in_participant'] = df_times.apply(f1, axis=1)  # temporary column
        df_times.set_index('utc_date', inplace=True)
        print('Inserting known locations')
        df_times['cells'] = df_times.apply(f3, axis=1)

        memo = dict()
        print('Set easting for dwell times')
        result_visits.apply(set_easting, axis=1)
        print('Set northing for dwell times')
        result_visits.apply(set_northing, axis=1)
        print('Combining easting and northing')
        df_times.apply(combine_e_n, axis=1)
        print('Get location of nearest two measurements in time')
        df_times['loc_tuples_e_n'] = df_times.apply(f2, axis=1)
        print('Computing PPA for remaining time steps')
        df_times['cells'] = df_times.apply(f4, axis=1)
        # print('Interpolating location for missing intervals')
        # df_times['cells'] = df_times.apply(fill_in_astar, axis=1)
        print('Filtering long dwells')
        result_visits.apply(filter_long_dwells, axis=1)

        # df_times =  df_times.drop(['in_participant', 'loc_tuples_e_n', 'temp_easting', 'temp_northing'], axis=1)
        # TODO: delete next line -- it is for testing purposes only
        df_times['len_cells'] = df_times.apply(lambda x: len(x['cells']), axis=1)
        return df_times


    elif version == 2:
        df_times, part_times = get_df_times(df_part)

        print('Creating in_participant column')
        df_times['in_participant'] = df_times.apply(f1, axis=1)  # temporary column
        df_times.set_index('utc_date', inplace=True)
        print('Inserting known locations')
        df_times['cells'] = df_times.apply(f3, axis=1)

        # insert small "wine glasses"
        print('Creating small wine glasses')
        result_visits.apply(small_wine_glass, axis=1)
        print('Get location of nearest two measurements in time')
        df_times['loc_tuples_e_n'] = df_times.apply(f2, axis=1)
        print('Computing PPA for remaining time steps')
        df_times['cells'] = df_times.apply(f4, axis=1)
        df_times = df_times.drop(['in_participant', 'loc_tuples_e_n'], axis=1)
        # TODO: delete next line -- it is for testing purposes only
        df_times['len_cells'] = df_times.apply(lambda x: len(x['cells']), axis=1)
        return df_times

    elif version == 3:
        df_times, part_times = get_df_times(df_part)

        print('Creating in_participant column')
        df_times['in_participant'] = df_times.apply(f1, axis=1)  # temporary column
        df_times.set_index('utc_date', inplace=True)
        print('Inserting known locations')
        df_times['cells'] = df_times.apply(f3, axis=1)

        # insert large "wine glass"
        print('Creating large wine glass')
        result_visits.apply(large_wine_glass, axis=1)
        print('Get location of nearest two measurements in time')
        df_times['loc_tuples_e_n'] = df_times.apply(f2, axis=1)
        print('Computing PPA for remaining time steps')
        df_times['cells'] = df_times.apply(f4, axis=1)
        df_times = df_times.drop(['in_participant', 'loc_tuples_e_n'], axis=1)
        # TODO: delete next line -- it is for testing purposes only
        df_times['len_cells'] = df_times.apply(lambda x: len(x['cells']), axis=1)
        return df_times





