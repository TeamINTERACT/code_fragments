"""
Author: Antoniu Vadan
"""

import pandas as pd
from datetime import datetime, timedelta
from ppa_jppa import city_kdtree
from street_network import points_along_path
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import time
import sys, os, pickle
from concurrent.futures import ThreadPoolExecutor as PoolExecutor


# trips = pd.read_csv('bus_saskatoon_6jan_27april/main1.csv')
# print(trips.head(40).to_string())


def ppa_bus_date(bus_main_path, calendar_path, df_city_path, date_str, vmax=10):
    """
    Function computes PPA dataframe of a bus system.
    :param bus_main_path: path to dataframe (in csv format) containing all bus data from transitfeeds as
        organized by bus_df_extraction.py (columns are ['route_id', 'trip_id', 'service_id', 'time',
        'stop_sequence', 'easting', 'northing'])
    :param calendar_path: path to calendar csv file which maps service_id to days of week
    :param df_city_path: path to pickled pandas dataframe of city grid -- contains 1s (in city) and np.NaNs
    :param date_str: YYYY-MM-DD string
    :param vmax: maximum speed of bus when travelling through the city in meters/second
    :return: dataframe with 'cells' column added, which contains a Python set of tuples of
        (easting, northing) data
    """

    def process_trip_df(df):
        def in_main(row):
            if row['time'].time() in main_times:
                return True
            else:
                return False

        def insert_known(row):
            if row['in_main']:
                result = set()
                temp = df.loc[df['time'].dt.time == row['time'].time()].reset_index()
                result.add((temp.at[0, 'northing'], temp.at[0, 'easting']))
                return result
            else:
                result = set()
                return result

        def loc_tuples(row):
            if row['cells']:
                return row['cells']
            j = 1
            k = 1
            # find how far a timestamp (which is not also in the participant dataframe),
            #   is from a timestamp which IS in the participant dataframe
            while not df_times.at[row.name - timedelta(minutes=j), 'cells']:  # check if set is empty
                j += 1
            while not df_times.at[row.name + timedelta(minutes=k), 'cells']:
                k += 1

            t1 = row.name - timedelta(minutes=j)
            t2 = row.name + timedelta(minutes=k)
            # keep track of the coordinates of the point whose distance is stored
            temp1 = list(df_times.loc[t1, 'cells'])[0]
            temp2 = list(df_times.loc[t2, 'cells'])[0]

            return (temp1[0], temp1[1], j), (temp2[0], temp2[1], k)

        def circles(row):
            if not row['cells'] and row['loc_tuples'] is not None:
                point1 = row['loc_tuples'][0]  # tuple of tuples with location data
                point2 = row['loc_tuples'][1]

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



        # create dataframe which has times starting at the time when the df starts at
        start_date = date_str
        start_time = df.at[0, 'time'].time().strftime('%H:%M:%S')
        start = str(start_date + ' '+ start_time)
        # ASSUMPTION: no bus trip is longer than 90 minutes -- extra time will be trimmed off
        times = pd.date_range(start, periods=90, freq='1min')
        df_times = pd.DataFrame(times)
        df_times.columns = ['time']

        df_times['route_id'] = df.at[0, 'route_id']
        df_times['trip_id'] = df.at[0, 'trip_id']
        df_times['service_id'] = df.at[0, 'service_id']
        df.set_index('time')

        # in_main column: True if at that time in df_times there is a corresponding entry in df
        main_times = df['time'].dt.time.to_list()
        df_times['in_main'] = df_times.apply(in_main, axis=1)

        # keep slice starting from first in_main timestamp to last in_main timestamp
        df_times = df_times.loc[: df_times[(df_times['in_main'] == True)].index[-1], :]
        df_times['cells'] = df_times.apply(insert_known, axis=1)

        # find how far a row with no easting/northing entries is from the nearest rows that DO contain
        #   easting and northing
        df_times = df_times.set_index('time')
        df_times['loc_tuples'] = df_times.apply(loc_tuples, axis=1)

        # compute circle intersections
        df_times['cells'] = df_times.apply(circles, axis=1)
        df_times = df_times.drop(['in_main', 'loc_tuples'], axis=1)

        return df_times

    df_city = pd.read_pickle(df_city_path)
    # set up KD Tree -- will be used for finding intersection of circles
    tree, points = city_kdtree(df_city)

    date = datetime.strptime(date_str, '%Y-%m-%d')
    weekday = date.weekday()
    weekday = days_of_week[weekday]

    calendar = pd.read_csv(calendar_path)[['service_id', 'monday', 'tuesday', 'wednesday',
                                           'thursday', 'friday', 'saturday', 'sunday']]
    main = pd.read_csv(bus_main_path, parse_dates=['time'])

    # get working service_id on that day of the week
    working = calendar[calendar[weekday] == 1]['service_id']
    main = main[main['service_id'].isin(working)]

    # separate all trips in their own dataframes
    trip_groupby = main.groupby(['route_id', 'trip_id'])
    all_trips = [trip_groupby.get_group(x).sort_values(by=['stop_sequence'])
                     .drop('stop_sequence', axis=1).reset_index() for x in trip_groupby.groups]
    # execute function on each separate dataframe
    all_results = list(map(process_trip_df, all_trips))
    day_trips = pd.concat(all_results)
    day_trips = day_trips.reset_index()
    # rename "time" column to "utc_date"
    day_trips.columns = ['utc_date', 'route_id', 'trip_id', 'service_id', 'cells']

    return day_trips


def ppa_bus_day(bus_main_path, calendar_path, df_city_path, day_of_week, vmax=10):
    """
    Function computes PPA dataframe of a bus system.
    :param bus_main_path: path to dataframe (in csv format) containing all bus data from transitfeeds as
        organized by bus_df_extraction.py (columns are ['route_id', 'trip_id', 'service_id', 'time',
        'stop_sequence', 'easting', 'northing'])
    :param calendar_path: path to calendar csv file which maps service_id to days of week
    :param df_city_path: path to pickled pandas dataframe of city grid -- contains 1s (in city) and np.NaNs
    :param day_of_week: day of week string (e.g. 'monday', 'tuesday', etc.)
    :param vmax: maximum speed of bus when travelling through the city in meters/second
    :return: dataframe with 'cells' column added, which contains a Python set of tuples of
        (easting, northing) data
    """


    def process_trip_df(df):
        def in_main(row):
            if row['time'].time() in main_times:
                return True
            else:
                return False

        def insert_known(row):
            if row['in_main']:
                result = set()
                temp = df.loc[df['time'].dt.time == row['time'].time()].reset_index()
                result.add((temp.at[0, 'northing'], temp.at[0, 'easting']))
                return result
            else:
                result = set()
                return result

        def loc_tuples(row):
            if row['cells']:
                return row['cells']
            j = 1
            k = 1
            # find how far a timestamp (which is not also in the participant dataframe),
            #   is from a timestamp which IS in the participant dataframe
            while not df_times.at[row.name - timedelta(minutes=j), 'cells']:  # check if set is empty
                j += 1
            while not df_times.at[row.name + timedelta(minutes=k), 'cells']:
                k += 1

            t1 = row.name - timedelta(minutes=j)
            t2 = row.name + timedelta(minutes=k)
            # keep track of the coordinates of the point whose distance is stored
            temp1 = list(df_times.loc[t1, 'cells'])[0]
            temp2 = list(df_times.loc[t2, 'cells'])[0]

            return (temp1[0], temp1[1], j), (temp2[0], temp2[1], k)

        def circles(row):
            if not row['cells'] and row['loc_tuples'] is not None:
                point1 = row['loc_tuples'][0]  # tuple of tuples with location data
                point2 = row['loc_tuples'][1]

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



        # create dataframe which has times starting at the time when the df starts at
        # start_date = date_str
        # start_time = df.at[0, 'time'].time().strftime('%H:%M:%S')
        # start = str(start_date + ' '+ start_time)

        start = df.at[0, 'time'].strftime('%Y-%m-%d %H:%M:%S')

        # ASSUMPTION: no bus trip is longer than 90 minutes -- extra time will be trimmed off
        times = pd.date_range(start, periods=90, freq='1min')
        df_times = pd.DataFrame(times)
        df_times.columns = ['time']

        df_times['route_id'] = df.at[0, 'route_id']
        df_times['trip_id'] = df.at[0, 'trip_id']
        df_times['service_id'] = df.at[0, 'service_id']
        df.set_index('time')

        # in_main column: True if at that time in df_times there is a corresponding entry in df
        main_times = df['time'].dt.time.to_list()
        df_times['in_main'] = df_times.apply(in_main, axis=1)

        # keep slice starting from first in_main timestamp to last in_main timestamp
        df_times = df_times.loc[: df_times[(df_times['in_main'] == True)].index[-1], :]
        df_times['cells'] = df_times.apply(insert_known, axis=1)

        # find how far a row with no easting/northing entries is from the nearest rows that DO contain
        #   easting and northing
        df_times = df_times.set_index('time')
        df_times['loc_tuples'] = df_times.apply(loc_tuples, axis=1)

        # compute circle intersections
        df_times['cells'] = df_times.apply(circles, axis=1)
        df_times = df_times.drop(['in_main', 'loc_tuples'], axis=1)

        return df_times

    df_city = pd.read_pickle(df_city_path)
    # set up KD Tree -- will be used for finding intersection of circles
    tree, points = city_kdtree(df_city)

    # date = datetime.strptime(date_str, '%Y-%m-%d')
    # weekday = date.weekday()
    # weekday = days_of_week[weekday]

    calendar = pd.read_csv(calendar_path)[['service_id', 'monday', 'tuesday', 'wednesday',
                                           'thursday', 'friday', 'saturday', 'sunday']]
    main = pd.read_csv(bus_main_path, parse_dates=['time'])

    # get working service_id on that day of the week
    working = calendar[calendar[day_of_week] == 1]['service_id']
    main = main[main['service_id'].isin(working)]

    # separate all trips in their own dataframes
    trip_groupby = main.groupby(['route_id', 'trip_id'])
    all_trips = [trip_groupby.get_group(x).sort_values(by=['stop_sequence'])
                     .drop('stop_sequence', axis=1).reset_index() for x in trip_groupby.groups]

    # execute function on each separate dataframe
    all_results = list(map(process_trip_df, all_trips))
    day_trips = pd.concat(all_results)
    day_trips = day_trips.reset_index()
    # rename "time" column to "utc_date"
    day_trips.columns = ['utc_date', 'route_id', 'trip_id', 'service_id', 'cells']

    return day_trips


def interpolate_bus_route(bus_main_path, calendar_path, road_graph_path, day_of_week, route_id, res, destination):
    """
    Interpolates the location of a bus for all trips of a given route and stores the trips separately in csv files
    :param bus_main_path: path to csv file containing all bus trips for all routes
    :param calendar_path: path to csv file containing information on what days bus routes operate on (from GTFS)
    :param road_graph_path: path to undirected, not simplified graph of the city's road network
    :param day_of_week: e.g. 'monday'
    :param route_id: from GTFS
    :param res: resolution of city grid
    :param destination: path to directory where the routes and their trips are to be stored
    Post conditions:
        Saves all trips with interpolated gaps in separate csv files
    :return: None
    """

    def interpolate(df):
        def in_main(row):
            if row['time'].time() in main_times:
                return True
            else:
                return False

        def insert_known(row):
            if row['in_main']:
                result = set()
                temp = df.loc[df['time'].dt.time == row['time'].time()].reset_index()
                result.add((temp.at[0, 'northing'], temp.at[0, 'easting']))
                return result
            else:
                result = set()
                return result

        def loc_tuples(row):
            if row['cells']:
                return row['cells']
            j = 1
            k = 1
            # find how far a timestamp (which is not also in the participant dataframe),
            #   is from a timestamp which IS in the participant dataframe
            while not df_times.at[row.name - timedelta(minutes=j), 'cells']:  # check if set is empty
                j += 1
            while not df_times.at[row.name + timedelta(minutes=k), 'cells']:
                k += 1

            t1 = row.name - timedelta(minutes=j)
            t2 = row.name + timedelta(minutes=k)
            # keep track of the coordinates of the point whose distance is stored
            temp1 = list(df_times.loc[t1, 'cells'])[0]
            temp2 = list(df_times.loc[t2, 'cells'])[0]

            return (temp1[0], temp1[1], j), (temp2[0], temp2[1], k)

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
                if (start, end, (n1 + n2 - 1)) in memo:
                    northing = memo[(start, end, (n1 + n2 - 1))][n1][1]
                    easting = memo[(start, end, (n1 + n2 - 1))][n1][0]
                    return {(int(round(northing) / res) * res, int(round(easting) / res) * res)}
                path = points_along_path(start, end, graph, (n1 + n2 - 1))
                memo[(start, end, (n1 + n2 - 1))] = path
                return {(int(round(path[n1][1]) / res) * res, int(round(path[n1][0]) / res) * res)}  # reverse E and N
                # return {(path[n1][1], path[n1][0])}
            else:
                return row['cells']

        # print(df)
        # print(type(df))
        # sys.exit()
        start = df.at[0, 'time'].strftime('%Y-%m-%d %H:%M:%S')

        # ASSUMPTION: no bus trip is longer than 90 minutes -- extra time will be trimmed off
        times = pd.date_range(start, periods=90, freq='1min')
        df_times = pd.DataFrame(times)
        df_times.columns = ['time']

        df_times['route_id'] = route_id
        df_times['trip_id'] = df.at[0, 'trip_id']
        df_times['service_id'] = df.at[0, 'service_id']
        df.set_index('time')

        # in_main column: True if at that time in df_times there is a corresponding entry in df
        main_times = df['time'].dt.time.to_list()
        df_times['in_main'] = df_times.apply(in_main, axis=1)

        # keep slice starting from first in_main timestamp to last in_main timestamp
        df_times = df_times.loc[: df_times[(df_times['in_main'] == True)].index[-1], :]
        df_times['cells'] = df_times.apply(insert_known, axis=1)

        # find how far a row with no easting/northing entries is from the nearest rows that DO contain
        #   easting and northing
        df_times = df_times.set_index('time')
        df_times['loc_tuples_e_n'] = df_times.apply(loc_tuples, axis=1)

        # compute interpolation
        memo = dict()
        df_times['cells'] = df_times.apply(fill_in, axis=1)
        memo = dict()
        df_times = df_times.drop(['in_main', 'loc_tuples_e_n'], axis=1)

        trip_id = df_times.reset_index().at[0, 'trip_id']

        main_dir = destination

        if not os.path.exists(main_dir + '/'+ day):
            os.makedirs(main_dir + '/'+ day)

        if not os.path.exists(main_dir + '/'+ day + '/' + route_id):
            os.makedirs(main_dir + '/'+ day + '/' + route_id)

        df_times.to_csv(main_dir + '/'+ day + '/' + route_id + '/' + trip_id)

        print('finished trip id', trip_id, 'for route', route_id)
        return df_times

    graph = nx.read_gpickle(road_graph_path)
    calendar = pd.read_csv(calendar_path)[['service_id', 'monday', 'tuesday', 'wednesday',
                                           'thursday', 'friday', 'saturday', 'sunday']]
    main = pd.read_csv(bus_main_path, parse_dates=['time'])
    main = main[main['route_id'] == route_id]

    # get working service_id on that day of the week
    working = calendar[calendar[day_of_week] == 1]['service_id']
    main = main[main['service_id'].isin(working)]

    # separate all trips in their own dataframes
    trip_groupby = main.groupby(['trip_id'])
    all_trips = [trip_groupby.get_group(x).sort_values(by=['stop_sequence'])
                     .drop('stop_sequence', axis=1).reset_index() for x in trip_groupby.groups]

    # execute function on each separate dataframe
    print('Starting interpolation for trips on', day, 'for route', route_id)
    [interpolate(x) for x in all_trips]


def create_routes_dict(bus_main_path, calendar_path, bus_dictionary_path):
    """
    Create a dictionary whose keys are (northing, easting) tuples and values are a [route_ids] list.
    These dictionaries are useful for the implementation of the filtering of bus routes which are irrelevant for
        a participant's trip, i.e. (for a participant trip) only compute jppa with bus routes that have bus stops
        within X meters from the start point of the trip and the end point of the trip
    :param bus_main_path: path to dataframe containing locations for all buses, all trips
    :param calendar_path: path to csv file containing information on what days bus routes operate on (from GTFS)
    :param bus_dictionary_path: path to directory where to pickle
    Post-conditions:
        7 pickled dictionaries stored in bus_dictionary_path -- one for each day of the week
    :return: None
    """

    def plug_in_routes(row):
        if not (row['northing'], row['easting']) in locations_routes:
            locations_routes[(row['northing'], row['easting'])] = [row['route_id']]
        else:
            if row['route_id'] not in locations_routes[(row['northing'], row['easting'])]:
                locations_routes[(row['northing'], row['easting'])].append(row['route_id'])

    main = pd.read_csv(bus_main_path, usecols=['route_id', 'easting', 'northing', 'service_id'])
    calendar = pd.read_csv(calendar_path)[['service_id', 'monday', 'tuesday', 'wednesday',
                                           'thursday', 'friday', 'saturday', 'sunday']]

    for day_of_week in days_of_week:
        locations_routes = dict()
        working = calendar[calendar[day_of_week] == 1]['service_id']
        main = main[main['service_id'].isin(working)]

        main.apply(plug_in_routes, axis=1)
        with open(bus_dictionary_path + '/dict_loc_buses' + '_' + day_of_week + '.pickle', 'wb') as handle:
            pickle.dump(locations_routes, handle, protocol=pickle.HIGHEST_PROTOCOL)


################################################################################################
# def wrap_func(route):
#     interpolate_bus_route(bus_path, calendar, graph_path, day, route, resolution)
#
#
# main_df = pd.read_csv(main_path, header=0, usecols=['route_id'])
#
# wed_fri_sat_sun = ['wednesday', 'friday', 'saturday', 'sunday']
#
# for day in wed_fri_sat_sun:
#     print('Starting work on', day)
#     list_routes = list(main_df['route_id'].unique())
#     with PoolExecutor(max_workers=2) as executor:
#         for _ in executor.map(wrap_func, list_routes):
#             pass
################################################################################################








