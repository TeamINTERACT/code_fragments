"""
Author: Antoniu Vadan, summer 2019
Description: this file contains function which gets n points between two points along a street network
"""

import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, mapping
import sys
import time
import datetime


def pickle_graph(shp_ll, path_out):
    """
    Given a path to a shapefile in (LONG LAT) format, get geometry and pickle the
        road network contained within the boundary. Road network is not simplified.
        i.e. roads that are not straight/linear are shaped by a series of nodes.
        (if simplify=True, then curved roads are represented by a few nodes, reducing the
        accuracy of the a* algorithm
    :param shp_ll: path to shapefile in long lat format
    :param path_out: path to where to pickle
    :return: None
    """

    geometry = gpd.read_file(shp_ll).at[0, 'geometry']
    G = ox.graph_from_polygon(geometry, network_type='drive', simplify=False)
    nx.write_gpickle(G, path_out)

def pickle_utm_graph(shp_ll, path_out):
    """
    Given a path to a shapefile in (LONG LAT) format, get geometry and pickle the
        road network contained within the boundary with data in UTM format.
        Road network is not simplified.
        i.e. roads that are not straight/linear are shaped by a series of nodes.
        (if simplify=True, then curved roads are represented by a few nodes, reducing the
        accuracy of the a* algorithm
    :param shp_ll: path to shapefile in long lat format
    :param path_out: path to where to pickle
    :return: None
    """
    geometry = gpd.read_file(shp_ll).at[0, 'geometry']
    G = ox.graph_from_polygon(geometry, network_type='drive', simplify=False)
    graph_proj = ox.project_graph(G)  # converts to UTM
    graph_proj = graph_proj.to_undirected()
    nx.write_gpickle(graph_proj, path_out)

def points_along_path(p1, p2, graph_proj, n):
    """
    Given 2 points, compute shortest path. Along that path, get n evenly spaced points
    There is a chance that the resultant points MAY NOT lie exactly on road network, as
        linear interpolation is used along segments of the linestring. Use non-simplified
        road graphs.
    :param p1: starting point (northing, easting)
    :param p2: destination (northing, easting)
    :param graph_proj: graph projection to UTM -- undirected
    :param n: number of evenly spaced points BETWEEN starting point and destination
    :return: list of tuples representing point coordinates in (EASTING, NORTHING) format
    """
    nodes_proj, edges_proj = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)

    # get nodes that are nearest to the points
    orig_node = ox.get_nearest_node(graph_proj, p1, method='euclidean')
    target_node = ox.get_nearest_node(graph_proj, p2, method='euclidean')
    e_n1 = nodes_proj.at[orig_node, 'geometry']
    e_n2 = nodes_proj.at[target_node, 'geometry']
    # print(e_n1)
    if e_n1 == e_n2:  # check if nodes are the same
        return [(e_n1.bounds[1], e_n1.bounds[0])] * (n+2)
    #################### TESTING ####################
    # cols = list(nodes_proj.columns.values)
    # print('orig node:', orig_node)
    # print(nodes_proj.loc[orig_node, cols])
    # print(edges_proj[edges_proj['v'] == 27571866].to_string())
    # print('target node:', target_node)
    # fig, ax = ox.plot_graph(graph_utm, show=False, close=False)
    # e_n1 = nodes_proj.at[orig_node, 'geometry']
    # e_n2 = nodes_proj.at[target_node, 'geometry']
    # print(e_n1)
    # print(e_n2)
    # plt.scatter([e_n1.bounds[0], e_n2.bounds[0]],
    #             [e_n1.bounds[1], e_n2.bounds[1]], c='red', s=30)
    # # plt.scatter(e_n1.bounds[0], e_n1.bounds[1], c='red', s=30)
    # plt.show()
    # sys.exit()
    #################################################
    # node entry in dataframe
    # o_closest = nodes_proj.loc[orig_node]
    # t_closest = nodes_proj.loc[target_node]
    route = nx.shortest_path(G=graph_proj, source=orig_node, target=target_node, weight='length')
    # route_length = nx.shortest_path_length(G=graph_proj, source=orig_node, target=target_node, weight='length')

    ##########################################################
    ### CODE FOR INTERPOLATING LOCATION ALONG LINESTRING/MULTILINESTRING ###
    node_list = list(map(lambda node: nodes_proj.at[node, 'geometry'], route))  # list of Point objects along route
    path_line = LineString(node_list)
    # Code from:
    # https://stackoverflow.com/questions/34906124/interpolating-every-x-distance-along-multiline-in-shapely
    # def redistribute_vertices(geom, distance):
    #     if geom.geom_type == 'LineString':
    #         num_vert = int(round(geom.length / distance))
    #         if num_vert == 0:
    #             num_vert = 1
    #         return LineString(
    #             [geom.interpolate(float(n) / num_vert, normalized=True)
    #              for n in range(num_vert + 1)])
    #     elif geom.geom_type == 'MultiLineString':
    #         parts = [redistribute_vertices(part, distance)
    #                  for part in geom]
    #         return type(geom)([p for p in parts if not p.is_empty])
    #     else:
    #         raise ValueError('unhandled geometry %s', (geom.geom_type,))
    def redistribute_vertices(geom, num_vert):
        if geom.geom_type == 'LineString':
            if num_vert == 0:
                num_vert = 1
            num_vert = num_vert + 1  # so that we get <num_vert> vertices IN BETWEEN p1 and p2
            return LineString(
                [geom.interpolate(float(i) / num_vert, normalized=True)
                 for i in range(num_vert + 1)])
        else:
            raise ValueError('unhandled geometry %s', (geom.geom_type,))

    # k = route_length/(n+1)
    # multiline_r = redistribute_vertices(path_line, k)
    multiline_r = redistribute_vertices(path_line, n)
    map_dict = mapping(multiline_r)  # dictionary. 'type':'LineString', 'coordinates':((.....))
    multiline_r = list(map_dict['coordinates'])  # list of coordinate tuples (easting, northing) unrounded

    return multiline_r


if __name__ == '__main__':
    print('reading graph')
    graph_utm = nx.read_gpickle('graphs/victoria_statca_road_utm_undirected')
    # undirected = graph_utm.to_undirected()
    # nx.write_gpickle(undirected, 'graphs/victoria_statca_road_utm_undirected')
    # the following point1 as source cannot be associated with a path (5777000, 387750)
    # point1 = (5777000, 387750)  # note: Point objects have (easting, northing) format
    # point2 = (5760000, 389000)
    # ((5772500, 395500, 1), (5772500, 395250, 3))
    # point1 = (5772500, 395500)
    # point2 = (5772500, 395250)
    point1 = (5369300, 470750)
    point2 = (5369250, 470750)
    # TODO: instead of 5 (chosen as example), this parameter should be given by amount of
    # TODO:     minute-wise timestamps in between the two points which do not have a
    # TODO:     location associated with them
    result = points_along_path(point1, point2, graph_utm, 29)
    print('Length:', len(result))
    print(result)
    # print(result[28])

    # sys.exit()

    # the following 2 lines are for plotting
    eastings = [x[0] for x in result]
    northings = [x[1] for x in result]

    print('plotting')

    # fig, ax = ox.plot_graph_route(graph_proj, route, origin_point=p1, destination_point=p2)
    fig, ax = ox.plot_graph(graph_utm, show=False, close=False)
    plt.scatter(eastings, northings, c='red', s=30)

    # plt.tight_layout()
    plt.show()

    # pickle_utm_graph('city_shapefiles/saskatoon_lat_long.shp',
    #                  'graphs/saskatoon_statca_road_utm_undirected', True)