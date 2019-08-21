"""
Author: Antoniu Vadan, summer 2019
Description: the following code sets up the pandas dataframe of
    data points representing grid for a given city (ASSUMPTION: city falls in one UTM
    zone number and zone letter)
"""

import sys
import numpy as np
import pandas as pd
import time
import geopandas as gpd
import to_utm
import fiona
from shapely.geometry import Point, MultiPolygon, mapping
from shapely.geometry.polygon import Polygon
import datetime


def city_matrix(filename, file_to, res):
    """
    Create pandas dataframe filled with zeros.
    Index values are northing values; column values are easting values
    :param filename: shapefile of single city -- data in Lambert Conformal Conic from StatsCanada
    :param file_to: name of file to pickle dataframe in
    :param res: resolution of grid (in meters)
    :return: None
    """

    multipolygon = get_polygon_points(filename, 'utm')

    if len(multipolygon) == 1:
        multipolygon = MultiPolygon([Polygon(multipolygon[0])])
    else:
        multi = [Polygon(poly) for poly in multipolygon]
        multipolygon = MultiPolygon(multi)

    xmin, ymin, xmax, ymax = multipolygon.bounds  # rectangle enclosing city
    s_w = (xmin, ymin)  # south-west coordinates
    n_e = (xmax, ymax)  # north-east coordinates

    # round boundary point coordinates to integers
    s_w = tuple(map(lambda x: int(round(x/res))*res if isinstance(x, float) else x, s_w))
    n_e = tuple(map(lambda x: int(round(x/res))*res if isinstance(x, float) else x, n_e))

    col_names = np.arange(s_w[0], n_e[0], res)
    row_names = np.arange(s_w[1], n_e[1], res)

    # set up empty city grid
    city_df = pd.DataFrame(np.NaN, index=row_names, columns=col_names)

    # algorithm: for each polygon create a box around it, then check each point in the box
    #   to see if it lies in the polygon
    for poly in multipolygon:
        xmin_s, ymin_s, xmax_s, ymax_s = poly.bounds
        for i in np.arange(int(round(ymin_s/res))*res, int(round(ymax_s/res))*res, res):
            for j in np.arange(int(round(xmin_s/res))*res, int(round(xmax_s/res))*res, res):
                point = Point(j, i)
                if poly.contains(point):
                    city_df.at[i, j] = 1

    # convert to sparse dataframe
    city_sparse = city_df.to_sparse()
    city_sparse.to_pickle(file_to)


def get_polygon_points(filename, coord_system):
    """
    Get polygon points in (UTM format) - ASSUMES POINTS LIE IN THE SAME ZONE
    :param filename: path of shapefile containing single city
    :return: list (multipolygon) of lists (polygons) containing tuples containing
        (easting, northing)
    """
    if coord_system not in ['utm', 'll']:
        print("Possible coord_system values are 'utm' and 'll'.")
        sys.exit()

    df = gpd.read_file(filename)
    shape = df.iloc[0]['geometry']
    shape_dict = mapping(shape)  # creates dictionary with entries 'type' and 'coordinates'

    # change data type to list instead of shapely multipolygon
    polygon_list = list()
    if shape_dict['type'] == 'Polygon':  # city boundary is defined by a single polygon
        polygon_list.append(list(shape_dict['coordinates'][0]))
    else:                                # city boundary is defined by multiple polygons
        temporary = list(shape_dict['coordinates'])
        for i in temporary:
            polygon_list.append(i[0])

    ### convert polygon vertices to UTM format ###
    if coord_system == 'utm':
        for i, polygon in enumerate(polygon_list):
            polygon_list[i] = [to_utm.lcc_to_utm(x[0], x[1])[0:2] for x in polygon]
    else:
        ### convert polygon vertices to lat-long format ###
        for i, polygon in enumerate(polygon_list):
            # polygon_list[i] = [to_utm.lcc_to_ll(x[0], x[1]) for x in polygon]
            polygon_list[i] = [(to_utm.lcc_to_ll(x[0], x[1])[1],
                                to_utm.lcc_to_ll(x[0], x[1])[0]) for x in polygon]

    return polygon_list


def shp_lcc_to_ll(path_shp_in, path_shp_out):
    """
    Create new shapefile by converting existing shapefile data from lcc to wgs84
    :param path_shp_in: path to shapefile containing a polygon/multipolygon in lcc coordinates
    :param path_shp_out: path and name of new shapefile containing lat/long data
        NOTE: must have .shp extension
    Post-conditions:
        new file created
    :return: None
    """
    polygon_list = get_polygon_points(path_shp_in, 'll')

    if len(polygon_list) == 1:
        multipolygon = Polygon(polygon_list[0])
        schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int'}  # not sure what the purpose of properties is
        }
    else:
        multi = [Polygon(poly) for poly in polygon_list]
        multipolygon = MultiPolygon(multi)
        schema = {
            'geometry': 'MultiPolygon',
            'properties': {'id': 'int'}  # not sure what the purpose of properties is
        }

    with fiona.open(path_shp_out, 'w', 'ESRI Shapefile', schema) as c:
        c.write({
            'geometry' : mapping(multipolygon),
            'properties' : {'id' : 0}
        })


if __name__ == '__main__':
    file = str(sys.argv[1])
    pickle_file = str(sys.argv[2])
    resolution = int(sys.argv[3])
    city_matrix(file, pickle_file, resolution)
    


    #### Extracts xmin, ymin, xmax, ymax when needed ####

    # multipolygon = get_polygon_points('city_shapefiles/victoria.shp', 'll')
    #
    # if len(multipolygon) == 1:
    #     multipolygon = MultiPolygon([Polygon(multipolygon[0])])
    # else:
    #     multi = [Polygon(poly) for poly in multipolygon]
    #     multipolygon = MultiPolygon(multi)
    #
    # xmin, ymin, xmax, ymax = multipolygon.bounds
    #
    # print(ymin, xmin, ymax, xmax)
