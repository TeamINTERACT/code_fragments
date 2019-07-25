"""
Author: Antoniu Vadan
"""


import pandas as pd
import numpy as np
import psycopg2
import os
from astar import astar
from participant_df_preprocessing import pre_processing
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt  # for plotting -- testing
import sys
import dask.dataframe as dd
import time, datetime


# for pre trip detection -- get all interact_ids, perform function on each section of main dataframe
#   by interact_id
#       select distinct interact_ids
#       on each interact_id, take the dataframe and perform preprocessing + function to insert in histogram
# this function is one that adds 1 to frequency histogram


"""
f = open('/home/anv309/Documents/yakitori_creds.txt', 'r')
content = f.readlines()
host = content[0]
dbname = content[1]
user = content[2]
password = content[3]

conn = psycopg2.connect("host=" + host + " dbname=" + dbname + " user=" + user + " password="+password)
cur = conn.cursor()

# gps table names: sd_raw_gps_test, ethica_raw_gps_test
sql = "SELECT DISTINCT interact_id FROM sd_gps_raw_test ;"
ser = pd.read_sql_query(sql, conn)  # pandas Dataframe with a single column containing interact ids

def fill_histogram(row):
    # Function is meant to be applied on the pandas Dataframe containing a column of distinct interact IDs (ser).

    pass

cur.close()
conn.close()

"""

def global_histogram(city_df_path, graph_path, credentials):
    """
    Create frequency histogram of cells visited by all participants in a sensedoc gps table.
    :param city_df_path: path to city_df pickle containing eastings as column names, northings as row names.
        Contains 1 for entries in the city, np.NaN for entries outside the city
    :param graph_path: path to gpickled graph of city. Undirected, in UTM coordinates.
    :param credentials: path to text file with the following content
        Line 1: host name
        Line 2: database name
        Line 3: user name
        Line 4: password
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

    city_df = pd.read_pickle(city_df_path)
    graph = nx.read_gpickle(graph_path)
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    df_histogram = city_df.to_dense().replace(1, -1)
    cols = list(df_histogram.columns.values)
    res = cols[1] - cols[0]

    f = open(credentials, 'r')
    content = f.readlines()
    host = content[0]
    dbname = content[1]
    user = content[2]
    password = content[3]
    conn = psycopg2.connect("host=" + host + " dbname=" + dbname + " user=" + user + " password=" + password)
    cur = conn.cursor()

    sql = "SELECT DISTINCT interact_id FROM sd_gps_raw_test WHERE LEFT(interact_id::text, 3)='101';"
    ids = pd.read_sql_query(sql, conn)  # pandas Dataframe with a single column containing interact ids
    ids = ids['interact_id'].to_list()
    total_ids = len(ids)

    cur.close()
    conn.close()

    for index, id in enumerate(ids):
        conn = psycopg2.connect("host=" + host + " dbname=" + dbname + " user=" + user + " password=" + password)
        cur = conn.cursor()
        sql = "SELECT * FROM sd_gps_raw_test WHERE interact_id=" + str(id) + ";"

        df = pd.read_sql_query(sql, conn)
        # TODO: COMMENT/UNCOMMENT FOLLOWING LINE
        # df = df.iloc[::5, :]

        # establishing connections and closing them after each query due to the assumption that it is better
        #   practice than leaving a connection open for a long period of time (whole process takes several hours)
        cur.close()
        conn.close()

        path = '../jppa_participant_dfs/victoria/temp/all/sd_raw_' + str(id) + '.csv'
        df.to_csv(path)

        df_preprocessed = pre_processing(path, city_df_path, data='sensedoc', snap=True, graph_path=graph_path)
        os.remove(path)
        df_preprocessed.to_csv('../jppa_participant_dfs/victoria/preprocessed/all/' + str(id) + '_preprocessed_snapped_15_625m')

        print('Filling in histogram')
        df_preprocessed.apply(fill_histogram, axis=1)
        print('Done participant ' + str(index) + '/' + str(total_ids-1))
        print('+++++++++++++++++++++++++++++++')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # For development stage
    # df_part = pd.read_csv('../jppa_participant_dfs/victoria/101065382_50/101065382_sd_preprocessed_50',
    #                       parse_dates=['utc_date'])
    # df_part = df_part.iloc[:, 1:]
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    return df_histogram


# def complete_histogram(histogram, road_graph, paths=True):
#     """
#
#     :param histogram:
#     :param road_graph:
#     :return:
#     """
#
#     if paths:
#         histogram = pd.read_csv(histogram)
#         graph = nx.read_gpickle(road_graph)





df_city = 'city_grids/victoria_15_625'
graph_p = 'graphs/victoria_statca_road_utm_undirected'
creds_path = '../../yakitori_creds.txt'

print('+++++++ all Victoria sd wave 1 onto 15.625m histogram +++++++')
t1 = time.time()
hist = global_histogram(df_city, graph_p, creds_path)
t2 = time.time()
print(str(datetime.timedelta(seconds=(t2 - t1))), '------- all Victoria sd wave 1 onto 15.625m histogram')
hist.to_csv('../jppa_participant_dfs/victoria/histogram_all_sd_wave1_15_625m_snapped')


# '../jppa_participant_dfs/victoria/histogram_all_sd_wave1_15_625m_snapped' -- HISTOGRAM TO COMPARE AGAINST


################################################################################################################
# TESTING A*


# df_histogram['sum_row'] = df_histogram.sum(axis=1)
# hist_total = df_histogram['sum_row'].sum(axis=0)
# df_histogram.drop('sum_row', axis=1)
#
# # make probability distribution
# df_histogram = df_histogram.applymap(lambda x: x/hist_total)
# df_histogram = df_histogram.applymap(lambda x: (1 - x)**4 if x < 1 else 1)  # g cost -- needed for A* algorithm
# # print(list(df_histogram.columns.values)[0], list(df_histogram.columns.values)[-2])  # limits of easting
# # print(list(df_histogram.index.values)[0], list(df_histogram.index.values)[-1])      # limits of northing




"""
start = (5776750, 390250)
end = (5776250, 386250)

print(df_part)
# print(df_histogram)
print(df_histogram.loc[5776000:5777000, range(386000, 390500, 250)].to_string())

path = astar(df_histogram, start, end)
print(path)

y = [y[0] for y in path]
x = [x[1] for x in path]

plt.gca().invert_yaxis()
plt.scatter(x, y)
plt.show()
"""