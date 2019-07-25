"""
Author: Antoniu Vadan
Description: Functions to compute Potential Path Area (PPA) and joint Potential Path Area (jPPA)
    -
"""


import pandas as pd
import numpy as np
from scipy import spatial
import math
import datetime
import time
from street_network import points_along_path
import networkx as nx
import sys
import dask.dataframe as dd


# TODO: prototype: demonstrate retrieving jppa pixels (with timestamps) for one participant with
# TODO:     ANY/ALL buses -- after implementing arcpy functionality to interpolate bus location

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


# TODO: test jppa_person_person()
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


def ppa_person(df_part_csv_path, df_city_path, paths_csv=None, visits_csv=None, version=1, road_graph_path=None,
               vmax=10, vwalk=2, pre_trip_detection=False):
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
    :param paths_csv: path to participant path dataframe as computed by trip_detection.py
    :param visits_csv: path to participant visits dataframe as computed by trip_detection.py
    :param version: version 1: assumption: people do not wander at all during their dwells
                    version 2: assumption: people wander in between the recordings at walking speed
                    version 3: assumption: people can wander from the beginning of the dwell until the end
                                            at walking speed (ignoring the measurements in between)
    :param road_graph_path: path to gpickled road network of city
    :param vmax: maximum speed of person when travelling through the city in meters/second
    :param vwalk: maximum speed of person when walking
    :param pre_trip_detection: returns dataframe with minute-wise timestamps for entire duration of study
                        containing easting and northing interpolated (using the street network graph) for
                        times when it is not recorded.
                        Gaps longer than 60 minutes are not filled

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
        # find how far a timestamp (which is not also in the participant dataframe),
        #   is from a timestamp which IS in the participant dataframe
        while not df_times.at[row.name - datetime.timedelta(minutes=j), 'cells']:  # check if set is empty
            j += 1
            if j == 60:  # set threshold: if data points are more than an hour apart, return None
                return

        while not df_times.at[row.name + datetime.timedelta(minutes=k), 'cells']:
            k += 1
            if k == 60:
                return

        if j + k >= 60:
            return

        t1 = row.name - datetime.timedelta(minutes=j)
        t2 = row.name + datetime.timedelta(minutes=k)
        # keep track of the coordinates of the point whose distance is stored
        temp1 = list(df_times.loc[t1, 'cells'])[0]
        temp2 = list(df_times.loc[t2, 'cells'])[0]

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
        try:
            easting = result_paths[result_paths['end_time'] == row['start_time']]['path_end_x']
            easting = float(easting)  # step is included because now easting is a Pandas Series object
            easting = round(easting / 250) * 250
            df_times.loc[row['start_time']:row['end_time'], 'temp_easting'] = easting
        except:
            easting = list(df_times.at[row['start_time'], 'cells'])
            easting = easting[0][1]
            easting = float(easting)
            easting = round(easting / 250) * 250
            df_times.loc[row['start_time']:row['end_time'], 'temp_easting'] = easting

    def set_northing(row):
        try:
            northing = result_paths[result_paths['end_time'] == row['start_time']]['path_end_y']
            northing = float(northing)  # step is included because now easting is a Pandas Series object
            northing = round(northing / 250) * 250
            df_times.loc[row['start_time']:row['end_time'], 'temp_northing'] = northing
        except:
            northing = list(df_times.at[row['start_time'], 'cells'])
            northing = northing[0][0]
            northing = float(northing)
            northing = round(northing / 250) * 250
            df_times.loc[row['start_time']:row['end_time'], 'temp_northing'] = northing

    def combine_e_n(row):
        if not math.isnan(row['temp_easting']):
            row['cells'].clear()
            row['cells'].add((int(row['temp_northing']), int(row['temp_easting'])))

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

    if version not in (1, 2, 3):
        print('There are three versions available (1, 2, 3)')
        sys.exit()

    df_part = pd.read_csv(df_part_csv_path, parse_dates=['utc_date'])
    df_city = pd.read_pickle(df_city_path)
    cols = list(df_city.columns.values)
    res = cols[1] - cols[0]

    pd.to_numeric(df_part['easting'])
    pd.to_numeric(df_part['northing'])

    # time range from the beginning of
    time_range = pd.date_range(df_part.loc[df_part.index[0], 'utc_date'].strftime('%Y-%m-%d'),
                               df_part.loc[df_part.index[-1], 'utc_date']
                               + pd.Timedelta(days=1),  # this line corrects the range -- otherwise it does not
                               freq='T')  # include last day

    start_time = df_part.at[0, 'utc_date']
    end_time = df_part.at[df_part.index[-1], 'utc_date']

    # trim ends of time_range -- e.g. it starts at time 00:00:00 while participant data starts at 04:02:00
    for i, t in enumerate(time_range):
        if t == start_time:
            time_range = time_range[i:]
            break

    for i, t in reversed(list(enumerate(time_range))):
        if t == end_time:
            time_range = time_range[0:i + 1]
            break

    # next step: go through all time steps in the time range
    #   and generate PPA/step for participant
    part_times = df_part['utc_date'].tolist()
    part_times = set(part_times)
    df_times = pd.DataFrame(time_range, columns=['utc_date'])

    print('Creating in_participant column')
    df_times['in_participant'] = df_times.apply(f1, axis=1)  # temporary
    df_times.set_index('utc_date', inplace=True)
    print('Inserting known locations')
    df_times['cells'] = df_times.apply(f3, axis=1)

    def apply_fill_in(df):
        return df.apply((lambda row: fill_in(row)), axis=1)

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
        graph_utm = nx.read_gpickle(road_graph_path)  # needed for the fill_in function
        print('getting location tuples')
        df_times['loc_tuples_e_n'] = df_times.apply(f2, axis=1)
        print('filling in cells using road network')
        # df_times['cells'] = df_times.apply(fill_in, axis=1)  # without parallelization
        ddata = dd.from_pandas(df_times, npartitions=16)
        df_times['cells'] = ddata.map_partitions(apply_fill_in).compute(scheduler='processes')

        # extract Easting and Northing in separate columns for trip detection
        print('separating eastings and northings')
        df_times['utm_n'] = df_times.apply(extract_n, axis=1)
        df_times['utm_e'] = df_times.apply(extract_e, axis=1)
        print('dropping NaNs')
        # remove times with no data for location (times when location is missing for longer than 60 mins)
        df_times = df_times.dropna()
        return df_times

    result_paths = pd.read_csv(paths_csv, parse_dates=['start_time', 'end_time'])
    result_visits = pd.read_csv(visits_csv, parse_dates=['start_time', 'end_time'])
    df_city = pd.read_pickle(df_city_path)

    # set up KD Tree -- will be used for finding intersection of circles
    tree, points = city_kdtree(df_city)

    result_paths.drop(['duration', 'start_zone', 'end_zone', 'end_zone', 'num_segments', 'segments'], axis=1,
                      inplace=True)

    result_paths['path_start_x'] = result_paths['path_start_x'].apply(lambda x: round(x / res) * res)
    result_paths['path_start_y'] = result_paths['path_start_y'].apply(lambda x: round(x / res) * res)
    result_paths['path_end_x'] = result_paths['path_end_x'].apply(lambda x: round(x / res) * res)
    result_paths['path_end_y'] = result_paths['path_end_y'].apply(lambda x: round(x / res) * res)

    # get resolution of grid from df_city
    cols = list(df_city.columns.values)
    res = cols[1] - cols[0]

    if version == 1:
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
        df_times =  df_times.drop(['in_participant', 'loc_tuples_e_n', 'temp_easting', 'temp_northing'], axis=1)
        # TODO: delete next line -- it is for testing purposes only
        df_times['len_cells'] = df_times.apply(lambda x: len(x['cells']), axis=1)

    if version == 2:
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

    if version == 3:
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


if __name__ == '__main__':
    # part_file = '../jppa_participant_dfs/saskatoon/301802247/301802247_ethica_preprocessed_250'
    df_saskatoon = 'city_grids/saskatoon_15_625'
    graph = 'graphs/saskatoon_statca_road_utm_undirected'

    df = pd.read_csv('../jppa_participant_dfs/saskatoon/301802247/301802247_ethica_preprocessed_250')
    print(df.head(200).to_string())

    # t1 = time.time()
    pre_trip = ppa_person(part_file, df_saskatoon, result_paths, result_visits, version=1)
    # t2 = time.time()
    # print(str(datetime.timedelta(seconds=(t2 - t1))), '------- ppa v1')
    # pre_trip.to_csv('../jppa_participant_dfs/saskatoon/301802247/301802247_v1_ethica_250')


