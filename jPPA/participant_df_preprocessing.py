"""
Author: Antoniu Vadan, summer 2019
Description: Process all participant data according to the steps defined in pre_processing()
"""


import pandas as pd
import to_utm
import sys
import networkx as nx
import osmnx as ox


def pre_processing(part_df, city_df, data='ethica', snap=False, graph_path=None):
    """
    Pickle (or to_csv, depending on future requirements) processed version of participant
        dataframe. This function does the following
            1. Rounds timestamp data to nearest minute
            2. Changes lat long coordinates to UTM
            3. Averages location of measurements taken within the same minute
            *4. Optional: snaps participant location to nearest road network node
            4. Rounds easting and northing to resolution
            5. Filters out data that does not lie within city polygon
    :param part_df: path to participant data in csv format
    :param city_df: path to city dataframe containing eastings as columns and northings
        as rows. Filled with 1s where points lie in polygon, np.Nans where they do not
    :param data: the name of the dataset from which the data is coming from. 'ethica' or 'sensedoc'
    :param snap: snap all entries to nearest point in city's road network. Boolean
    :param graph_path: path to undirected graph (in UTM coordinates) representation of road network of city

    Post-conditions:
        New pickle file created for participant data
    :return: None
    """
    if data not in ('ethica', 'sensedoc'):
        raise ValueError("Select data='ethica' or data='sensedoc'")

    if data == 'ethica':
        y_wgs = 'y_wgs_ph'
        x_wgs = 'x_wgs_ph'
    else:
        y_wgs = 'y_wgs_sd'
        x_wgs = 'x_wgs_sd'

    df_city = pd.read_pickle(city_df)
    df_part = pd.read_csv(part_df, parse_dates=['utc_date'])
    df_part = df_part[['interact_id', 'utc_date', y_wgs, x_wgs]]
    pd.to_numeric(df_part[y_wgs])
    pd.to_numeric(df_part[x_wgs])

    # get resolution from city_df
    cols = list(df_city.columns.values)
    res = cols[1] - cols[0]
    print('Rounding to nearest minute')
    # round to nearest minute
    df_part['utc_date'] = df_part['utc_date'].values.astype('<M8[m]')

    # change lat lon coordinates to UTM
    print('Converting to UTM')
    df_lat_lon = df_part[[y_wgs, x_wgs]]
    df_lat_lon = df_lat_lon.apply(lambda x: to_utm.ll_to_utm(x[0], x[1])[0:2], axis=1)
    df_lat_lon = df_lat_lon.apply(pd.Series)  # this line splits the resulting tuple

    df_lat_lon.columns = ['easting', 'northing']
    df_part = df_part.drop([y_wgs, x_wgs], axis=1)
    df_part = df_part.join(df_lat_lon)

    # average location measurements taken during the same minute
    # this step reduces the size of the dataframe by ~10x
    print('Averaging location of participant for each minute')
    df_part = df_part.groupby(['interact_id', 'utc_date']).mean().reset_index()

    def f(row):
        try:
            if df_city.at[row['northing'], row['easting']] == 1:
                return True
            else:
                return False
        except:
            return False

    def snap_to_network(row):
        easting = row['easting']
        northing = row['northing']
        point = (northing, easting)
        nearest_node = ox.get_nearest_node(graph, point, method='euclidean')
        nearest_point = nodes.at[nearest_node, 'geometry']
        new_point = nearest_point.bounds  # nearest_point is a Point object, thus use .bounds
        df_part.at[row.name, 'easting'] = new_point[0]
        df_part.at[row.name, 'northing'] = new_point[1]

    df_part = df_part.set_index('utc_date')
    if snap:
        graph = nx.read_gpickle(graph_path)
        nodes, _ = ox.graph_to_gdfs(graph, nodes=True, edges=True)
        # get nodes that are nearest to the entries in the dataframe
        print('Snapping to nearest road network node')
        df_part.apply(snap_to_network, axis=1)

    print('Rounding easting and northing to nearest grid point')
    # round easting, northing to closest grid point based on resolution
    df_part.loc[:, 'easting'] = df_part.loc[:, 'easting'].apply(lambda x: int(round(x / res)) * res)
    df_part.loc[:, 'northing'] = df_part.loc[:, 'northing'].apply(lambda x: int(round(x / res)) * res)

    # create column indicating if northing/easting lies inside city polygon
    print('Filtering points which do not lie in polygon')
    df_part['in_city'] = df_part.apply(f, axis=1)
    # remove entries not lying in polygon
    df_part = df_part[df_part['in_city'] == True].reset_index()

    df_part.drop(['in_city'], axis=1, inplace=True)

    return df_part



if __name__ == '__main__':
    # participant_data = '../jppa_participant_dfs/saskatoon/301223102/301223102_ethica_raw'
    # city_df = 'city_grids/saskatoon_df_250'
    # city_graph = 'graphs/saskatoon_statca_road_utm_undirected'

    # result = pre_processing(participant_data, city_df, 'ethica', snap=True, graph_path=city_graph)
    # result.to_csv('../jppa_participant_dfs/saskatoon/301223102/301223102_ethica_preprocessed_250_snapped')


