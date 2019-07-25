"""
Author: Antoniu Vadan
Description: the following code converts LCC (Lambert conformal conic) data into UTM coordinate data
    - also rounds the input data to voxel data (depends on resolution)
"""

from pyproj import Proj, transform
import utm
import geopandas as gpd


def lcc_to_utm(x, y):
    """
    Take in x and y coordinates in Lambert conformal conic format (configurations described below)
        and convert to UTM coordinates
    :param x: LCC x coordinate
    :param y: LCC y coordinate
    :return: location in UTM format -- easting, northing, zone number, zone letter
    """
    proj_lcc = Proj(
        proj='lcc',
        datum='NAD83',
        lat_1=49,
        lat_2=77,
        lat_0=63.390675,
        lon_0=-91.87,
        x_0= 6200000.000000,
        y_0= 3000000.000000
        )

    lon, lat = proj_lcc(x, y, inverse=True)

    # convert latitude-longitude to UTM
    return utm.from_latlon(lat, lon)


def lcc_to_ll(x, y):
    """
        Take in x and y coordinates in Lambert conformal conic format (configurations described below)
            and convert to lat-long coordinates
        :param x: LCC x coordinate
        :param y: LCC y coordinate
        :return: location in lat-long format
        """

    proj_lcc = Proj(
        proj='lcc',
        datum='NAD83',
        lat_1=49,
        lat_2=77,
        lat_0=63.390675,
        lon_0=-91.87,
        x_0=6200000.000000,
        y_0=3000000.000000
    )
    lon, lat = proj_lcc(x, y, inverse=True)
    return lat, lon



def ll_to_utm(lat, lon):
    """
    Take in x and y coordinates in latitude longitude format and convert to UTM coordinates
    :param lat: latitude
    :param lon: longitude
    :return: location in UTM format -- easting, northing, zone number, zone letter
    """
    return utm.from_latlon(lat, lon)


def lcc_to_utm_polygon(multipolygon):
    """
    Take in list (multipolygon) of lists (polygons) containing coordinate tuples expressed in LCC
        and return same list of lists in UTM format (only easting, northing) -- assumes same zone number
        and letter
    :param multipolygon: list of lists containing tuples of LCC coordinates
    :return: multipolygon in UTM coordinates
    """
    proj_lcc = Proj(
        proj='lcc',
        datum='NAD83',
        lat_1=49,
        lat_2=77,
        lat_0=63.390675,
        lon_0=-91.87,
        x_0= 6200000.000000,
        y_0= 3000000.000000
        )

    for poly in multipolygon:
        for i in range(len(poly)):
            lon, lat = proj_lcc(poly[i][0], poly[i][1], inverse=True)
            coords_utm = utm.from_latlon(lat, lon)
            poly[i] = (coords_utm[0], coords_utm[1])
    return multipolygon


def shp_lcc_to_ll(path_shp_in, path_shp_out):
    """
    Convert shapefile data from lcc to wgs84
    :param path_shp_in: path to shapefile containing a polygon/multipolygon in lcc coordinates
    :param path_shp_out: path and name of new shapefile containing lat/long data
    Post-conditions:
        new file created
    :return: None
    """
    gdf = gpd.read_file(path_shp_in)
    polygon_list = get_polygon_points(path_shp_in, 'll')
    return polygon_list


