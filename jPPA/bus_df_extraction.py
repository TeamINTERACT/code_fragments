import pandas as pd
import to_utm
import sys
import math
from datetime import datetime


def bus_df(directory, file_to, res):
    """
    Create dataframe from city bus routes and write it to a csv file
    :param directory: path to directory containing the stop_times, stops, trips text files
    :param file_to: dataframe pickle file destination
    :param res: space pixel resolution
    :return: None
    """
    df_stop_times = pd.read_csv(directory + '/stop_times.txt', parse_dates=['departure_time'])
    df_stop_times = df_stop_times[['trip_id', 'stop_id', 'departure_time', 'stop_sequence']]
    df_stop_times.columns = ['trip_id', 'stop_id', 'time', 'stop_sequence']
    df_stops = pd.read_csv(directory + '/stops.txt')
    df_stops = df_stops[['stop_id', 'stop_lat', 'stop_lon']]

    # a string was found in the 'stop_id' column (expected int)
    # convert to NaN and drop
    df_stops[['stop_id']] = df_stops[['stop_id']].apply(pd.to_numeric,errors='coerce')
    df_stops = df_stops[pd.notnull(df_stops['stop_id'])]

    df_trips = pd.read_csv(directory + '/trips.txt')
    # df_trips = df_trips[['route_id', 'trip_id', 'service_id']]
    routes_trips = df_trips[['route_id', 'trip_id']]
    service = df_trips[['service_id', 'trip_id']]  # service_id is added later -- after groupby
                                      # necessary to determine on which day of the week the bus
                                            # operates

    df_main = pd.merge(routes_trips, df_stop_times, how='inner', on='trip_id')
    df_main = pd.merge(df_main, df_stops, how='inner', on='stop_id')
    df_main = df_main.sort_values(['route_id', 'trip_id', 'stop_sequence'])
    df_main = df_main.reset_index(drop=True)

    # change lat lon coordinates to UTM
    df_stop_lat_lon = df_main[['stop_lat', 'stop_lon']]
    df_stop_lat_lon = df_stop_lat_lon.apply(lambda x: to_utm.ll_to_utm(x[0], x[1])[0:2], axis=1)
    df_stop_lat_lon = df_stop_lat_lon.apply(pd.Series)  # this line splits the resulting tuple
                                                        #   from the code above to two columns

    df_stop_lat_lon.columns = ['easting', 'northing']
    df_main = df_main.drop(['stop_lat', 'stop_lon'], axis=1)
    df_main = df_main.join(df_stop_lat_lon)

    # preprocess time measurements -- change hour 24 to hour 0 and convert to pandas timestamp
    df_main['time'] = df_main['time'].apply(lambda x: '00' + x[2:] if x[0:2] == '24' else x)
    df_main['time'] = df_main['time'].apply(lambda x: '01' + x[2:] if x[0:2] == '25' else x)
    df_main['time'] = df_main['time'].apply(lambda x: '02' + x[2:] if x[0:2] == '26' else x)
    df_main['time'] = df_main['time'].apply(lambda x: datetime.strptime(x, '%H:%M:%S'))
    df_main['time'] = df_main['time'].values.astype('<M8[m]')

    df_main = df_main.groupby(['route_id', 'trip_id', 'time']).mean().reset_index()
    df_main = pd.merge(df_main, service, how='inner', on='trip_id')

    # round to space resolution
    df_main.loc[:, 'easting'] = df_main.loc[:, 'easting'].apply(lambda x: int(round(x / res)) * res)
    df_main.loc[:, 'northing'] = df_main.loc[:, 'northing'].apply(lambda x: int(round(x / res)) * res)

    df_main = df_main[['route_id', 'trip_id', 'service_id', 'time', 'stop_sequence',
                       'easting', 'northing']]

    df_main.to_csv(file_to)


if __name__ == '__main__':
    direct = sys.argv[1]
    file = sys.argv[2]
    resolution = int(sys.argv[3])
    # direct = 'bus_victoria_10jan_8april_18'
    # file = 'bus_victoria_10jan_8april_18/main1.csv'
    # resolution = 250
    bus_df(direct, file, resolution)
