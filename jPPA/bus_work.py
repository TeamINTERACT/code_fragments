"""
Author: Antoniu Vadan
"""

import pandas as pd
from datetime import datetime, timedelta
from ppa_jppa import city_kdtree
import time
import sys


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
    :param vmax: maximum speed of person when travelling through the city in meters/second
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


days_of_week = {
        0 : 'monday',
        1 : 'tuesday',
        2 : 'wednesday',
        3 : 'thursday',
        4 : 'friday',
        5 : 'saturday',
        6 : 'sunday',
    }

main_path = 'bus_victoria_10jan_8april_18/main1.csv'
bus_calendar_path = 'bus_victoria_10jan_8april_18/calendar.txt'
date = '2018-01-12'
city_path = 'city_grids/victoria_df_250'

t1 = time.time()
result = ppa_bus_date(main_path, bus_calendar_path, city_path, date)
t2 = time.time()
# print(result.to_string())
print(str(timedelta(seconds=(t2 - t1))), '------- ppa bus 250m one day')
result.to_csv('bus_victoria_10jan_8april_18/ppa_250_' + date)
