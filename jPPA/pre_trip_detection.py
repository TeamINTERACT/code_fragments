"""
Author: Antoniu Vadan, summer 2019
Description: Create histogram with city grid cell visit frequency. Can further process it to prepare for A* search.
"""

import pandas as pd
import numpy as np
import psycopg2
import os
import math
from astar import astar
from participant_df_preprocessing import pre_processing
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt  # for plotting -- testing
import sys
import dask.dataframe as dd
import time, datetime
from shapely.geometry import LineString, mapping


# To test correctness of path interpolation for participants, get dataframe of counts for filled in participant data
# and do entry-wise mean squared error with dataframe of counts containing data that is not filled in
# '../jppa_participant_dfs/victoria/histogram_all_sd_wave1_15_625m_snapped' -- HISTOGRAM TO COMPARE AGAINST


def global_histogram(city_df_path, graph_path, credentials, dataset):
    """
    Create frequency histogram of cells visited by all participants in a sensedoc gps table.
    Return a dataframe with frequency counts
    :param city_df_path: path to city_df pickle containing eastings as column names, northings as row names.
        Contains 1 for entries in the city, np.NaN for entries outside the city
    :param graph_path: path to gpickled graph of city. Undirected, not simplified, in UTM coordinates.
    :param credentials: path to text file with the following content
        Line 1: host name
        Line 2: database name
        Line 3: user name
        Line 4: password
    :param dataset: 'ethica' or 'sensedoc'
    :return: pandas Dataframe representing frequency histogram of visited cells in a city by all participants
        Contains np.NaN outside the city, -1 for cells that were not visited, and a positive integer for visited cells
            representing how many times that cell was visited
    """

    def fill_histogram(row):
        """
        Apply function to a single participant's dataframe. It takes the easting and northing, gets nearest node
            in road network graph, rounds that to resolution of histogram, and adds 1 to that location.
        """
        point = (row['northing'], row['easting'])
        nearest_node = ox.get_nearest_node(graph, point, method='euclidean')

        node_point = nodes.at[nearest_node, 'geometry']

        easting = int(round(node_point.bounds[0] / res)) * res
        northing = int(round(node_point.bounds[1] / res)) * res
        cell_value = df_histogram.at[northing, easting]

        if cell_value == np.NaN:
            return
        elif cell_value == -1:  # in the city, not visited
            df_histogram.at[northing, easting] = 1
        else:
            df_histogram.at[northing, easting] += 1

    if dataset not in ('ethica', 'sensedoc'):
        raise ValueError("Dataset can be 'ethica' or 'sensedoc'")

    if dataset == 'sensedoc':
        dataset = 'sd'

    city_df = pd.read_pickle(city_df_path)

    graph = nx.read_gpickle(graph_path)
    nodes, _ = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    df_histogram = city_df.to_dense().replace(1, -1)
    cols = list(df_histogram.columns.values)
    res = cols[1] - cols[0]

    # Set up connection with database where raw participant data is held
    f = open(credentials, 'r')
    content = f.readlines()
    host = content[0]
    dbname = content[1]
    user = content[2]
    password = content[3]
    conn = psycopg2.connect("host=" + host + " dbname=" + dbname + " user=" + user + " password=" + password)
    cur = conn.cursor()

    sql = "SELECT DISTINCT interact_id FROM ethica_gps_raw_test WHERE LEFT(interact_id::text, 3)='301';"
    ids = pd.read_sql_query(sql, conn)  # pandas Dataframe with a single column containing interact ids
    ids = ids['interact_id'].to_list()
    total_ids = len(ids)

    cur.close()
    conn.close()

    for index, id in enumerate(ids):
        conn = psycopg2.connect("host=" + host + " dbname=" + dbname + " user=" + user + " password=" + password)
        cur = conn.cursor()
        sql = "SELECT * FROM ethica_gps_raw_test WHERE interact_id=" + str(id) + ";"

        df = pd.read_sql_query(sql, conn)

        # Establishing connections and closing them after each query due to the assumption that it is better
        #   practice than leaving a connection open for a long period of time (whole process takes several hours)
        cur.close()
        conn.close()

        path = '../jppa_participant_dfs/saskatoon/temp/ethica/ethica_raw_' + str(id) + '.csv'
        df.to_csv(path)

        df_preprocessed = pre_processing(path, city_df_path, data='ethica', snap=True, graph_path=graph_path)
        os.remove(path)
        df_preprocessed.to_csv('../jppa_participant_dfs/saskatoon/preprocessed/ethica/' + str(id) + '_preprocessed_snapped_15_625m')

        print('Filling in histogram')
        df_preprocessed.apply(fill_histogram, axis=1)
        print('Done participant ' + str(index) + '/' + str(total_ids-1))
        print('+++++++++++++++++++++++++++++++')

    return df_histogram


def raw_histogram_processing(histogram_path, graph_path):
    """
    Take in a dataframe (histogram) with the frequency of visit for each cell and convert into a dataframe containing
        cost of passing through each cell, as to be used by the a* algorithm
    :param histogram_path: path to dataframe containing frequency of visits for each cell
        Note: contains  np.NaN if a cell is outside the city, and -1 for grids inside the city which
            have not been visited
    :param graph_path: path to graph of road network for the city (graph is not simplified, utm projected, undirected)
    :return: dataframe containing cost of passing through each cell, as to be used by the a* algorithm
        Note: visited nodes: cost  0 - 1
                 road nodes: cost    100
             not road nodes: cost 100000
           outside the city: cost 100000
    """
    # TODO: make sure the costs are set to desired values

    def redistribute_vertices(geom, distance):
        """
        Returns list of points `distance` meters apart between first and last points on the `geom`
        """
        if geom.geom_type == 'LineString':
            num_vert = int(round(geom.length / distance))
            if num_vert == 0:
                num_vert = 1
            return LineString(
                [geom.interpolate(float(n) / num_vert, normalized=True)
                 for n in range(num_vert + 1)])
        elif geom.geom_type == 'MultiLineString':
            parts = [redistribute_vertices(part, distance)
                     for part in geom]
            return type(geom)([p for p in parts if not p.is_empty])
        else:
            raise ValueError('unhandled geometry %s', (geom.geom_type,))

    def get_coords(row):
        """
        Extract coordinates
        """
        point = row['geometry']
        point = point.bounds
        nodes.at[row.name, 'easting'] = point[0]
        nodes.at[row.name, 'northing'] = point[1]

    def visit_cost(x):
        """
        Set cost of passing through a cell that was visited before
        """
        if math.isnan(x):
            return np.NaN
        elif 0 < x < 1:
            return 1 - x
        else:
            return 0

    def input_road_nodes(row):
        if df_histogram.at[row['northing'], row['easting']] == 0:
            df_histogram.at[row['northing'], row['easting']] = 100

    def input_complete_network(row):
        """
        For every graph edge, call redistribute_vertices() and input those points in the histogram
            -- provides a complete road network representation in the dataframe
        """
        multiline_r = redistribute_vertices(row['geometry'], res * 0.75)
        map_dict = mapping(multiline_r)  # dictionary. 'type':'LineString', 'coordinates':((.....))
        multiline_r = list(map_dict['coordinates'])  # list of coordinate tuples unrounded
        multiline_r = [((round(x[1] / res) * res), (round(x[0] / res) * res)) for x in
                       multiline_r]  # (northing, easting) tuples
        for pt in multiline_r:
            input_vertices(pt)

    def input_vertices(point):
        """
        :param point: (northing, easting) tuple
        """
        val = df_histogram.at[point[0], point[1]]
        if val == 0 or val > 1:
            df_histogram.at[point[0], point[1]] = 100

    def get_visited_cell_costs(x):
        """
        Keep track of all costs of visited cells. This is so that the costs can be modified -- reprojected on a
        new range of values: min value starting at 0 and max value ending at 1
        """
        if 0 < x < 1:
            visited_cell_costs.append(x)

    def modify_cost(x):
        """
        Change the cost of stepping into a visited cell. The range of values is projected onto a new range: from 0 to 1.
        The reason behind this computation is to spread out the costs.
        """
        if 0 < x < 1:
            new_val = (((x - old_min) * new_range) / old_range) + new_min
            return new_val
        else:
            return x

    df_histogram = pd.read_csv(histogram_path, dtype=np.float)
    df_histogram = df_histogram.set_index('Unnamed: 0')
    df_histogram.columns = df_histogram.columns.astype(float)
    df_histogram.index = df_histogram.index.astype(float)
    cols = list(df_histogram.columns.values)
    res = cols[1] - cols[0]

    graph = nx.read_gpickle(graph_path)
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    print('Create separate northing column and easting column for nodes')
    nodes.apply(get_coords, axis=1)  # creates separate northing column and easting column
    nodes['easting'] = nodes['easting'].apply(lambda x: int(round(x / res)) * res)
    nodes['northing'] = nodes['northing'].apply(lambda x: int(round(x / res)) * res)

    df_histogram = df_histogram.replace(-1, 0)

    print('Calculating total visits')
    df_histogram['sum_row'] = df_histogram.sum(axis=1)
    hist_total = df_histogram['sum_row'].sum(axis=0)
    df_histogram = df_histogram.drop('sum_row', axis=1)

    print('Changing histogram to each entry having cost of a* passing through')
    # make probability distribution
    df_histogram = df_histogram.applymap(lambda x: x / hist_total)
    df_histogram = df_histogram.applymap(visit_cost)  # cost function -- needed for A* algorithm -- CHANGE IT?

    # setting cost of stepping onto road, cost of stepping onto not road, and cost of stepping outside the city bounds
    df_histogram = df_histogram.replace(np.NaN, 100000)  # arbitrarily high -- outside city

    print('Plugging in nodes')
    nodes.apply(input_road_nodes, axis=1)
    print('Plugging in vertices along edges')
    edges.apply(input_complete_network, axis=1)

    df_histogram = df_histogram.replace(0, 100000)  # arbitrarily high -- not road

    visited_cell_costs = []

    # Spread visited cell costs over 0 - 1 range
    print('Modifying the range of costs for visited cells')
    df_histogram.applymap(get_visited_cell_costs)

    old_max = max(visited_cell_costs)
    old_min = min(visited_cell_costs)
    new_max = 1
    new_min = 0
    old_range = old_max - old_min
    new_range = new_max - new_min

    df_histogram = df_histogram.applymap(modify_cost)

    return df_histogram


# TODO: put underneath code in a function?
#  Purpose: after filling in all missing data of participants, fill in the histogram with the visit counts
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
directory = '../jppa_participant_dfs/victoria/preprocessed/all'
directory = os.fsdecode(directory)

graph_path = 'graphs/victoria_not_simplified_road_utm_undirected'
graph = nx.read_gpickle(graph_path)
nodes, _ = ox.graph_to_gdfs(graph, nodes=True, edges=True)

df_city_path = 'city_grids/victoria_15_625'
city_df = pd.read_pickle(df_city_path)
df_histogram = city_df.to_dense().replace(1, -1)
cols = list(df_histogram.columns.values)
res = cols[1] - cols[0]

i = 1

for file in os.listdir(directory):
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    filename = os.fsdecode(file)
    print(filename)
    print('Participant number:', i)
    part_path = directory + '/' + filename

    part_df = pd.read_csv(part_path, parse_dates=['utc_date'])
    part_df = part_df.iloc[::5, :]

    t1 = time.time()
    part_df.apply(fill_histogram, axis=1)  # filling takes place here
    t2 = time.time()
    print(str(datetime.timedelta(seconds=(t2 - t1))), '-------', file)
    i += 1
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

df_histogram.to_csv('../jppa_participant_dfs/victoria/histograms/histogram_ANALYSIS_every5_sd_wave1_15_625_counts')
"""
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

hist_path = '../jppa_participant_dfs/victoria/histograms/histogram_ANALYSIS_every5_sd_wave1_15_625_counts'
graph_path = 'graphs/victoria_not_simplified_road_utm_undirected'
processed_histogram = raw_histogram_processing(hist_path, graph_path)

processed_histogram.to_csv('../jppa_participant_dfs/victoria/histograms/histogram_ANALYSIS_every5_sd_wave1_15_625_cost')
