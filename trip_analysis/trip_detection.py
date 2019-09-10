"""
trip_detection.py
Author: Janelle Berscheid, July 2018
Port of original ArcPy trip detection code by Benoit Thierry.

detect_trips() method takes in a path to a .csv file containing GPS data with at least the following headers:
interact_id, utc_date, lat and lon (will generate utm_e and utm_n if not already included)
and returns the detected paths and hotspots.
A parameter dictionary may be passed to detect_trips(). If not, the default parameters will be used.

Parameters:

CELL_SIZE: The square cell size of the location grid, in meters.
KERNEL_BANDWIDTH: The uncertainty of the GPS reading, in meters. Used for KDE calculations.
INTERPOLATE_MAX_DELAY: The time threshold for starting interpolation, in seconds.
INTERPOLATE_MAX_DROP_TIME: The maximum time between points after which not to interpolate, in seconds.
INTERPOLATE_MAX_DISTANCE: The maximum distance between points, in meters, to interpolate over.
MIN_VISIT_DURATION: Minimum duration of a dwell, in seconds, for it to be considered a valid visit
MIN_VISIT_TIMESPAN: Minimum time, in seconds, that must elapse between dwells; otherwise they will be merged into a single visit.
MIN_LOOP_DURATION: Minimum duration of a looping path, in second, to keep it and consider it a valid path.
SIMPLIFY_LINE_TOLERANCE: # TODO: currently unused, parameter for unimplemented line simplification method
SMOOTH_LINE_TOLERANCE: # TODO: currently unused, parameter for unimplemented line smoothing method
DOWNSAMPLE_FREQUENCY: The frequency to downsample GPS signals to: one point every ?? seconds.
MIN_HOTSPOT_KEEP_DURATION: Minimum duration of a hotspot, in seconds, to keep it and consider it a hotspot.
BUFFER_WEIGHT: Array for convolving the GPS points in temporal order to weigh the kernel values to identify hotspots.
BUFFER_SNAP: Array for convolving the GPS points in temporal order to determine which should be snapped to hotspots.
MIN_PATH_KEEP_DURATION: (new parameter) The minimum duration of a path to keep it, in seconds.
KERNEL_HOTSPOT_THRESHOLD: (new parameter) A threshold above which kernel values may be considered on hotspots.
                        The original code set it as 1.0 (standard deviations above the mean zone kernel value), but this
                        port uses a different kernel and performs better with lower values.
                        Warning: this value may be tuned to fit the sample data too closely. Test with other participants
                        to determine the best default value for this parameter.
SIGNIFICANT_GAP_LENGTH: (new parameter) Minimum length, in seconds, of a GPS data gap for it to be labeled as a period
                        of missing data. Gaps with missing data cannot be labeled as either trips or visits.
                        In the final results, some gaps may be folded into temporally adjacent visits at the same location
                        to create a single, longer visit if the length of the gap is less than MIN_VISIT_TIMESPAN.

Remaining work:
- Test to see if TinyExtentsError will ever actually trigger; if so, port code which handles it, otherwise remove it
- Test to see if the maximum raster size will ever be a problem; if not, remove code which limits it
- Port path smoothing and simplifying functions
- Add option to smooth out the detected missing data gaps, combining long adjacent gaps separated by a few observations

Quality of life improvements:
- Verbose mode to save more intermediate steps to disk
- Clean up temporary files when finished, if desired
- Write log file of each run
"""

from utilities import *
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# Constants
# TODO: remove if not strictly necessary. Requires further testing
MAX_RASTER_SIZE = 30000  # greatest possible number of cells per row or column

# Default Parameters (see explanation above)
DEFAULT_PARAMS = {
    "CELL_SIZE": 10,  # meters, for grid
    "KERNEL_BANDWIDTH": 100,  # basically uncertainty of GPS in meters
    "INTERPOLATE_MAX_DELAY": 120,  # seconds
    "INTERPOLATE_MAX_DROP_TIME": 3600,  # seconds
    "INTERPOLATE_MAX_DISTANCE": 100,  # meters
    "MIN_VISIT_DURATION": 300,  # seconds
    "MIN_VISIT_TIMESPAN": 3600,
    "MIN_LOOP_DURATION": 300,
    "SIMPLIFY_LINE_TOLERANCE": 100,  # TODO: currently unused, for implementing ArcPy path simplification in future
    "SMOOTH_LINE_TOLERANCE": 50,  # TODO: currently unused, for implementing ArcPy path smoothing in future
    "DOWNSAMPLE_FREQUENCY": 5,
    "MIN_HOTSPOT_KEEP_DURATION": 150,  # seconds
    "MIN_PATH_KEEP_DURATION": 30,  # NEW parameter! seconds
    "KERNEL_HOTSPOT_THRESHOLD": 0.2,  # NEW parameter! May need to be tuned
    "BUFFER_WEIGHT": np.array([0.1, 0.25, 0.3, 0.25, 0.1]),
    "BUFFER_SNAP": np.array([0.5, 1.0, 1.5, 1.0, 0.5]),
    "SIGNIFICANT_GAP_LENGTH": 300  # NEW parameter! Minimum length of a data gap to be considered a missing data period
}

SENSEDOC_COLUMNS = {
    "lat": "y_wgs_sd",
    "lon": "x_wgs_sd",
    "utm_n": "y_utm_sd",
    "utm_e": "x_utm_sd"
}

ETHICA_COLUMNS = {
    "lat": "y_wgs_ph",
    "lon": "x_wgs_ph",
    "utm_n": "y_utm_ph",
    "utm_e": "x_utm_ph"
}

##################################
# Utility methods
##################################


def format_gps_data(df, columns=SENSEDOC_COLUMNS, code=None):
    """
    Method for removing unneeded columns and invalid GPS points prior to beginning trip detection.
    :param df: pandas dataframe containing GPS data to format
    :param columns: a dictionary containing the names of lat, lon, utm_n and utm_e columns in the dataframe.
                    May differ between Ethica and Sensedoc data.
    :param code: the ESPG code of the data to be projected, as a string. Leave as None if no conversion is needed.
    :return: formatted pandas dataframe
    """
    df.columns = df.columns.str.lower()

    # Add UTM projections to DF if it doesn't exist already
    if columns["utm_e"] not in list(df.columns) or columns["utm_n"] not in list(df.columns):
        utm_e, utm_n = get_projection(df[columns["lon"]].values, df[columns["lat"]].values, code=code)
        df[columns["utm_e"]] = utm_e
        df[columns["utm_n"]] = utm_n

    # Remove all columns but the ones we need.
    try:
        df = df[list(["utc_date"] + list(columns.values()))]
        # Remove GPS points |X| > 180.0, |Y| > 90.0
        df = df[df[columns["lat"]].abs() <= 90.0]
        df = df[df[columns["lon"]].abs() <= 180.0]
    except KeyError:
        print("Error: incorrectly formatted GPS dataframe. Missing expected columns: ",
              list(["utc_date"] + list(columns.values())))
        exit()

    # Sort df by utc_date
    df = df.sort_values("utc_date")
    # TODO: wrangle datetime format if we have to

    # TODO: is there other cleaning that needs to be done we can assume for later steps?

    df = df[["utc_date", columns["lon"], columns["lat"], columns["utm_e"], columns["utm_n"]]]
    """
    # BT: one very rough filtering could be on DOP values, more sophisticated filtering
    #     could involve mean/median filtering or Kalman filters.
    #     See Zheng paper [DOI: http://dx.doi.org/10.1145/2743025]
    """

    return df


def get_data_gaps(df, gap_length=DEFAULT_PARAMS["SIGNIFICANT_GAP_LENGTH"], smooth_gaps=False):
    """
    Utility function for characterizing gaps existing in GPS data prior to trip detection.
    Creates a dataframe containing gaps of non-trivial length, marking their start and stop time, and saves as a temporary
    file to disk.
    :param df: pandas dataframe containing the GPS observations
    :param gap_length: inimum length of a data gap, in seconds, to be considered as a missing data period
    :param smooth_gaps: If True, gaps will be merged if separated by a small number of observations
    :return: None. Writes the gaps to a temporary file, to be loaded during later processing.
    """
    df = df.sort_values("utc_date")
    df["gap_time"] = ((df["utc_date"] - df.shift(1)["utc_date"]) / np.timedelta64(1, 's')).fillna(0)
    gaps = df[df["gap_time"] > 1][["utc_date", "gap_time"]]
    print("Writing temp gap file.")
    gaps.to_csv(os.getcwd() + "/temp/temp_gaps.csv", index=False)
    sig_gaps = gaps[gaps["gap_time"] > gap_length]
    sig_gaps["stop_time"] = sig_gaps["utc_date"] - np.timedelta64(1, 's')
    sig_gaps["start_time"] = sig_gaps["stop_time"] - pd.to_timedelta(sig_gaps["gap_time"], unit="s")
    sig_gaps = sig_gaps[["start_time", "stop_time"]]

    if smooth_gaps:
        # TODO: If a very long gap is broken up by just a couple of observations, discard the observations and fold into the gap
        pass

    sig_gaps.to_csv(os.getcwd() + "/temp/sig_temp_gaps.csv", index=False)
    df = df.drop(["gap_time"], axis=1)


def resize_cells(w, h, cell_size, kernel_bandwidth, utm=True):
    """
    Method for resizing the grid cells if the area covered is too large to compute properly with the default cell size.
    :param w: width of the extents, in decimal degrees
    :param h: height of the extents, in decimal degrees
    :param cell_size: current width/height of square grid cells, in meters
    :param kernel_bandwidth: width/height of kernel, in meters
    :param utm: True if the GPS readings are in UTM format, false otherwise.
    :return: integer value of the new grid cell size, in meters
    """
    # get the longest dimension of the GPS data (width or height) from data extents
    max_dim = max(w, h)
    if not utm:
        max_dim = decimal_degrees_to_meters(max_dim)
    # compare with max raster size
    # TODO: Do we even need to worry about max raster size? Is this an ArcPy limitation?
    """
    BT: this is indeed a limitation of ArcPy, yet we may still want to keep a max raster
        size to keep a reasonnable computation time || the other option would be to implement
        a multiscale/sparse raster processing approach and focus on the areas where GPS 
        fixes exist
    """
    if max_dim / cell_size > MAX_RASTER_SIZE:
        # redefine cellsize to stay within limits
        cell_size = int(np.ceil(max_dim / MAX_RASTER_SIZE))

        # check that new cell size is still compatible with the kernel bandwidth
        if kernel_bandwidth / 2.0 > cell_size:
            print("Changing raster resolution to ", cell_size, ", to keep raster size below threshold.")
        else:
            print("ABORTING DUE TO CRITICAL RASTER SIZE (max dimension: {:.1f}m)".format(max_dim))
            sys.exit()
    return cell_size


def downsample_trace(df, frequency=5):
    """
    Takes the original GPS trace, and downsamples it to one reading every [frequency] seconds.
    :param df: GPS points as a pandas dataframe
    :param frequency: integer describing the number of seconds to elapse between readings in the downsampled data
    :return: a new dataframe containing the downsampled data
    """
    dates = df["utc_date"]
    # Calculate difference between two adjacent samples
    differences = ((dates - dates.shift(1)) / np.timedelta64(1, 's')).fillna(0)

    # Extract those that have difference over frequency
    mask = differences > frequency
    keep = df[mask]
    lower_diff = df[~mask]
    lower_diff = lower_diff.iloc[::frequency]
    downsampled = pd.concat([keep, lower_diff], axis=0)
    downsampled = downsampled.sort_values("utc_date").reset_index(drop=True)
    return downsampled


def interpolate_over_period(df, frequency, noise=3.0, columns=SENSEDOC_COLUMNS):
    """
    Helper method for interpolate_over_dropped_periods() (see below), which performs actual interpolation
    Here we are assuming UTM
    TODO: Refine to allow for WGS as well?
    :param df: pandas dataframe (group??) containing the range of points to iterpolate over
    :param frequency: how frequently to add interpolated points, in seconds
    :param noise: sigma parameter for the random noise distribution, in meters
                    TODO: if allowing for WGS, convert this parameter to decimal degrees before using
    :param columns: a dictionary containing the names of lat, lon, utm_n and utm_e columns in the dataframe.
                    May differ between Ethica and Sensedoc data.
    :return: pandas dataframe containing only the new rows
                concatenate these new rows onto your original dataframe
    """

    f_lat, f_lon, f_n, f_e = tuple(df[[columns["lat"], columns["lon"], columns["utm_n"], columns["utm_e"]]].iloc[0])
    new_rows = df.copy().set_index("utc_date").resample("%dS" % (int(frequency),)).asfreq().replace([0.0], np.nan)
    new_rows.iloc[0][columns["lat"]] = f_lat
    new_rows.iloc[0][columns["lon"]] = f_lon
    new_rows.iloc[0][columns["utm_e"]] = f_e
    new_rows.iloc[0][columns["utm_n"]] = f_n
    new_rows = new_rows.interpolate()
    new_rows = new_rows.reset_index()
    new_rows.columns = ["utc_date", columns["lat"], columns["lon"], columns["utm_n"], columns["utm_e"]]

    noise = pd.DataFrame(np.random.normal(0, noise, (len(new_rows), 2)), columns=["xnoise", "ynoise"])
    new_rows[columns["utm_n"]] = new_rows[columns["utm_n"]] + noise["ynoise"]
    new_rows[columns["utm_e"]] = new_rows[columns["utm_e"]] + noise["xnoise"]

    # TODO: if target dataframe holds both WGS and UTM, convert the columns UTM to WGS and add those (or vice versa)

    return new_rows


def interpolate_over_dropped_periods(df, frequency, max_delay=120, max_drop_time=3600, max_distance=100,
                                     columns=SENSEDOC_COLUMNS, utm=True):
    """
    Interpolate GPS points for periods with large gaps
    Criteria:
        - if delay with previous point over max_delay: start interpolation
        - if dropped period is over max_drop_time: do not interpolate
            (unless the distance from last valid point is below max_distance)
    :param df: pandas dataframe of GPS points
    :param frequency: how frequently to add interpolated points, in seconds
    :param max_delay: time threshold for starting interpolation, in seconds
    :param max_drop_time: maximum time after which to not interpolate, in seconds
    :param max_distance: maximum distance for interpolation between points, in meters
    :param columns: a dictionary containing the names of lat, lon, utm_n and utm_e columns in the dataframe.
                May differ between Ethica and Sensedoc data.
    :param utm: True if the GPS readings are in UTM format, false otherwise.
    :return: new dataframe containing the interpolated points
    """
    # get time delta between all adjacent pairs of points (in seconds)
    differences = ((df["utc_date"].shift(-1) - df["utc_date"]).shift(1) / np.timedelta64(1, 's')).fillna(0.0)

    # get Euclidean distance delta between all adjacent pairs of points
    if utm:
        df["prev_e"] = df.shift(1)[columns["utm_e"]]
        df["prev_n"] = df.shift(1)[columns["utm_n"]]
        distance_delta = df[[columns["utm_e"], columns["utm_n"],
                             "prev_e", "prev_n"]].apply(lambda x: get_euclidean_distance((x["prev_e"], x["prev_n"]),
                                                         (x[columns["utm_e"]], x[columns["utm_n"]])), axis=1)
    else:
        distance_delta = [get_great_circle_distance((df.iloc[i - 1][columns["lon"]], df.iloc[i - 1][columns["lat"]]),
                                                    (df.iloc[i][columns["lon"]], df.iloc[i][columns["lat"]])) for i in range(1, len(df))]

    distance_delta = pd.Series([0.0]).append(distance_delta, ignore_index=True)
    distance_delta = pd.Series(distance_delta)

    # find all rows where time delta is greater than the max_delay parameter AND
    # (the delay is less than max_drop_time OR distance delta is less than max_distance)

    delay_mask = differences > max_delay
    drop_distance_mask = ((differences < max_drop_time) | (distance_delta < max_distance))
    df = df.drop(["prev_e", "prev_n"], axis=1)
    selected = df[delay_mask & drop_distance_mask]

    # interpolate over those matching rows with interpolate_over_period()
    print("Previous trace length: ", len(df))
    new_points = interpolate_over_period(selected, frequency)
    interpolated_df = pd.concat([df, new_points], axis=0)
    interpolated_df = interpolated_df.drop_duplicates(["utc_date"])
    interpolated_df = interpolated_df.sort_values("utc_date")
    interpolated_df.reset_index()
    print("New trace length after interpolation: ", len(interpolated_df))
    return interpolated_df


def grid_loc_to_utm(coord, axis):
    """
    Utility method to convert a grid location to a UTM value
    :param coord: the index value along a given axis, as a floating-point number
    :param axis: numpy array containing the axis values for each index
    :return: a single value representing the UTM coordinate at that index
    """
    edge1 = int(np.floor(coord))
    edge2 = int(np.ceil(coord))
    if edge1 == edge2:
        return axis[edge1]

    difference = axis[edge2] - axis[edge1]
    return axis[edge1] + np.modf(coord)[0] * difference

##################################
# Algorithm stages
##################################


# Step 1: Extract hotspots from GPS data (may have been resampled, interpolated etc.)

def extract_hotspots(points, x_bounds, y_bounds, kernel_bandwidth, qualifiers="5", width=50, height=50, cell_size=10):
    """
    Extract hotspots from GPS track by identifying the local peaks
        on a kernel density surface built from the GPS fixes.
    :param points: a numpy array containing the GPS coordinates
    :param x_bounds: tuple of the minimum/maximum x value
    :param y_bounds: tuple of the minimum/maximum y value
    :param kernel_bandwidth: the kernel uncertainty in meters
    :param width: the number of cells along the grid width
    :param height: the number of cells along the grid height
    :param cell_size: size of a grid cell in meters
    :return: a numpy array containing the kernel values for each GPS point,
             a numpy array containing the zone values for each GPS point,
             and a pandas dataframe containing the discovered hotspots.
    """

    print("Initializing the grid")
    # Step 0a: Initialize everything you need

    x_grid = np.linspace(x_bounds[0], x_bounds[1], width)
    y_grid = np.linspace(y_bounds[0], y_bounds[1], height)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    grid = np.vstack((y_mesh.ravel(), x_mesh.ravel())).T

    # Step 1: Run Kernel Density Estimation
    # ArcPy assumes planar distance and quartic kernel; we can only manage Gaussian/Epanechnikov

    # Note that fitting the kernel takes a long time; 3-4 hours may be normal
    kde = KernelDensity(bandwidth=kernel_bandwidth, kernel="epanechnikov")
    print("Fitting the kernel")
    kde.fit(points)
    print("Scoring the kernel")
    result_kernel = np.exp(kde.score_samples(grid))

    result_k_2d = result_kernel.reshape((height, width))
    # Make the plot
    levels = np.linspace(0, result_kernel.max(), 255)
    plt.contourf(x_mesh, y_mesh, result_k_2d, levels=levels, cmap=plt.cm.Reds)
    plt.show()

    # Locating kernel values for GPS points
    print("Snapping kernel values.")
    y_indices = np.searchsorted(y_grid, points.T[0])
    x_indices = np.searchsorted(x_grid, points.T[1])
    indexes = [y_indices[i] * width + x_indices[i] for i in range(len(y_indices))]
    kernel_snapped_values = [result_kernel[x] for x in indexes]

    print("Masking to area of interest")
    # redefining area of interest
    aoi = np.where(result_kernel == 0.0, np.nan, result_kernel)

    # get the mean of the non-zero values (from area of interest)
    mean = np.nanmean(aoi)

    # Step 3: Retain values above average or 1 std, set others to null
    aoi = np.where(result_kernel < mean, 0, result_kernel)
    k_mask = np.where(aoi != 0, 1.0, 0).reshape((height, width))
    k_bool = k_mask.astype(bool)

    # Watershed algorithm using skimage:
    # See http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    # (removed the line about distance since the kernel function operates as distance)
    # Increasing the footprint tends to decrease the number of local maxima

    print("Running watershed.")
    local_maxi = peak_local_max(result_k_2d, indices=False, footprint=np.ones((3, 3)),
                                labels=k_bool)
    markers = ndi.label(local_maxi, structure=np.ones((3, 3)))[0]

    sink_mask = np.where(markers > 0, 1, 0)
    temp_markers = ndi.label(sink_mask, structure=np.ones((3, 3)))[0]
    maximum = np.max(temp_markers)
    centroids = ndi.measurements.center_of_mass(sink_mask, temp_markers, index=range(1, maximum + 1))

    print("Calculating hotspots.")
    np.set_printoptions(threshold=np.nan)
    if len(np.unique(markers)) == 2:
        hotspots = pd.DataFrame([[1, grid_loc_to_utm(centroids[0][1], x_grid),
                                  grid_loc_to_utm(centroids[0][0], y_grid)]],
                                columns=["hotspot_id", "centroid_x", "centroid_y"])
    else:
        centroids_x = [grid_loc_to_utm(x[1], x_grid) for x in centroids]
        centroids_y = [grid_loc_to_utm(x[0], y_grid) for x in centroids]
        hotspots = pd.DataFrame([], columns=["hotspot_id", "centroid_x", "centroid_y"])
        hotspots["hotspot_id"] = range(1, np.max(markers) + 1)
        hotspots["centroid_x"] = centroids_x
        hotspots["centroid_y"] = centroids_y

    # Perform watershed using sinks as markers
    labels = watershed(-result_k_2d, markers, mask=k_bool)
    zone_labels = [labels.reshape((height * width,))[x] for x in indexes]

    pd.set_option('display.max_rows', len(hotspots))
    print("Located hotspots:")
    print(hotspots)

    return kernel_snapped_values, zone_labels, hotspots


# Step 2: Calculate normal and modified kernel

def get_norm_modified_kernel(df, buffer_weight, kernel_bandwidth=100, qualifiers="5",):
    """
    Adds the time-convolved modified kernel values to the GPS data, as well as the normalized values for both.
    :param df: a pandas dataframe containing the GPS data (plus zone and kernel values from extract_hotspots())
    :param buffer_weight: a 1D numpy array containing the weight window for convolution
    :return: a pandas datafram containing the input data plus modified kernel and normalized kernel values.
    """
    # calculate zone mean, std kernel values
    zone_groups = df.groupby("zone")
    aggregate_stats = zone_groups.agg({"kernel": [np.mean, np.std]})
    aggregate_stats.columns = ["mean", "std"]
    kernel = df["kernel"].values
    zones = df["zone"].values

    # calculate convolution of kernel values with weight (=modified kernel) --uses weight matrix and buffer values above
    modified_kernel = np.convolve(kernel, buffer_weight, "same")

    # calculate normalized values of kernels based on the zone/basin
    zone_mean = np.array([-9999 * len(zones)], np.float)
    zone_std = np.array([-9999 * len(zones)], np.float)
    norm_kernel = np.array([0 * len(kernel)], np.float)
    for index, row in aggregate_stats.iterrows():
        mean = row["mean"]
        std = row["std"]
        zone_mean = np.where(zones == index, mean, zone_mean)
        zone_std = np.where(zones == index, std, zone_std)

    try:
        norm_kernel = np.where(zone_mean == -9999, -9999, (kernel - zone_mean) / zone_std)
    except RuntimeWarning:
        print("Divide by zero error.")
        norm_kernel = np.where(zone_std == 0, zone_mean, norm_kernel)
    norm_modified_kernel = np.where(zone_mean == -9999, -9999, (modified_kernel - zone_mean) / zone_std)

    # Good line to place a breakpoint to debug Kernel

    df["norm_kernel"] = norm_kernel
    df["modfied_kernel"] = modified_kernel
    df["norm_modified_kernel"] = norm_modified_kernel
    if len(df[df["norm_kernel"] == -9999]) > 0 or len(df[df["norm_modified_kernel"] == -9999]) > 0:
        print("WARNING: Some kernel values are still set to null values.")

    return df


# Step 3: Link GPS data to discovered hotspots

def link_gps_to_hotspots(df, buffer_snap, kernel_hotspot_threshold):
    """
    Method for finding GPS points closest to hotspots, to determine which should be "snapped" to those hotspots.
    :param df: a pandas dataframe containing the GPS data, zone, and normalized kernel values
    :param buffer_snap: a 1D numpy array containing the weight window for convolution
    :param kernel_hotspot_threshold: a positive float value, greater than 0, used to find significant kernel values
    :return: a pandas dataframe with the GPS data, plus a binary "snap" value indicating a hotspot match
    """
    # removing 0 ("no zone") from the list of valid zones)
    zones = np.setdiff1d(df["zone"].value_counts().index.values, np.zeros((1,)))

    snap_dict = dict.fromkeys(np.unique(zones))
    for z in zones:
        snap_dict[z] = np.zeros(len(df), dtype=int)

    # Find points with kernel values higher than the threshold and assume they belong to hotspots.
    for i in range(len(df)):
        if df.iloc[i]["zone"] != 0:
            if ((df.iloc[i]["norm_kernel"] >= kernel_hotspot_threshold) or
                    (df.iloc[i]["norm_modified_kernel"] >= kernel_hotspot_threshold)):
                snap_dict[df.iloc[i]["zone"]][i] = 1

    # Convolve the snap condition temporally for each zone
    for z in zones:
        snap_dict[z] = np.convolve(snap_dict[z], buffer_snap, "same")

    zone_array = df["zone"].values
    snap_val = np.zeros(len(df))
    for k, snap in snap_dict.items():
        snap_val = np.where(zone_array == k, snap, snap_val)

    # give a mask of the points where the smoothed kernel values are above 0
    snap_val = np.where(snap_val > 0, 1, 0)

    df["snap_to_hs"] = snap_val
    return df


# Step 4: Refine the detected hotspots by extracting dwell times

def refine_hotspots(df, hotspots, data_gaps, min_keep_duration, gap_length=DEFAULT_PARAMS["SIGNIFICANT_GAP_LENGTH"]):
    """
    Create a list of dwells to find potential visits, find the time spent at each hotspot, and refine list of hotspots
    to include only those meeting the minimum duration threshold.
    :param df: a pandas dataframe containing the GPS data with the "snap" flag from link_gps_to_hotspots()
    :param hotspots: a pandas dataframe containing the hotspots found in extract_hotspots()
                    Contains a hotspot ID and the centroid coordinates.
    :param data_gaps: a pandas dataframe containing the start and stop periods for missing data gaps.
                    Used to create "dummy" points for breaking up visits
    :param min_keep_duration: the number of seconds above which a hotspot must be visited to be considered valid
    :param gap_length: minimum length of a missing data gap, in seconds, to be considered significant
    :return: the filtered pandas dataframe containing hotspot data, with the total duration added, and
                a pandas dataframe containing dwells, with the hotspot ID, start and end times and duration in seconds
    """

    # filter df to only snapped values
    filtered = df[df["snap_to_hs"] > 0].copy()

    # Add fake GPS points to the filtered visits to signify data gaps.
    # These are used to break up visits with long gaps, and are removed later.
    fake_points = data_gaps.copy()
    fake_points.loc[:, "utc_date"] = fake_points["start_time"] + np.timedelta64(1, 's')
    fake_points.loc[:, "zone"] = -999

    filtered = pd.concat([filtered, fake_points], axis=0)
    filtered = filtered.sort_values("utc_date").reset_index(drop=True)

    # Identify points as beginnings of new visits by their difference in zone
    # Fill appropriate values for first row to account for shifts
    filtered["diff_visit"] = (filtered["zone"] != filtered["zone"].shift(-1)).shift(1)
    filtered.loc[0, "diff_visit"] = True

    # A visit must be broken into two if it has a period of missing data in the middle.
    filtered["sig_gap"] = (((filtered["utc_date"].shift(-1) - filtered["utc_date"]) /
                            np.timedelta64(1, 's')).fillna(0) > gap_length).shift(1)
    filtered.loc[0, "sig_gap"] = True
    filtered["diff_visit"] = (filtered["diff_visit"] | filtered["sig_gap"])

    # Propagate the start time "downwards" through the results; all points from the same visit will have the same start
    filtered["start_time"] = filtered.shift(1)["utc_date"]
    filtered.loc[0, "start_time"] = filtered.iloc[0]["utc_date"]
    filtered.loc[filtered["diff_visit"], "start_time"] = filtered[filtered["diff_visit"]]["utc_date"]
    last_result = pd.Series([0])
    while not filtered["start_time"].equals(last_result):
        last_result = filtered["start_time"].copy()
        filtered.loc[~(filtered["diff_visit"]), "start_time"] = filtered["start_time"].shift(1)

    # Remove the fake data points representing gaps
    filtered = filtered[filtered["zone"] != -999]

    # Get the end time for each dwell by finding the last timestamp for each dwell
    visit_agg = filtered.groupby("start_time")
    stop_times = []
    for a, b in visit_agg:
        b = b.sort_values("utc_date")
        stop_times.append(b.iloc[-1]["utc_date"])

    # select rows where df["diff_visit"] = True to get a summary of the visit
    dwell_rows = filtered[filtered["diff_visit"]].copy()
    dwell_rows["stop_time"] = stop_times

    # Create dataframe containing dwells
    dwell_rows["duration"] = (dwell_rows["stop_time"] - dwell_rows["start_time"]) / np.timedelta64(1, 's')
    dwell_rows = dwell_rows.reset_index(drop=True)
    dwell_rows = dwell_rows[["zone", "start_time", "stop_time", "duration"]]

    # Add the total visit duration (in seconds) to a location in the hotspot data and filter by duration
    # Accounts for shift in hotspot ID and index
    duration_agg = dwell_rows[["zone", "duration"]].groupby("zone").agg(sum)
    last = duration_agg.iloc[-1]["duration"]
    hotspots["total_duration"] = duration_agg["duration"]
    hotspots["total_duration"] = hotspots["total_duration"].shift(-1)
    hotspots.loc[max(duration_agg.index) - 1, "total_duration"] = last
    hotspots = hotspots[~hotspots["total_duration"].isnull()]
    hotspots = hotspots[hotspots["total_duration"] > min_keep_duration]
    return hotspots, dwell_rows


# Step 5: Build visits from the refined hotspot data

def filter_visits(dwell_table, min_visit_timespan, min_visit_duration):
    """
    Method for finding visits from the dwell data.
    Group visits to the same place that too close to the previous visit, remove too-short visits
    :param dwell_table: pandas dataframe containing dwell data (start, stop, locationID, duration)
    :param min_visit_timespan: the minimum number of seconds that must elapse between visits to be considered different
    :param min_visit_duration: the minimum number of seconds a dwell must be to be considered a visit
    :return: dataframe containing the visits (columns: start, stop, locationID, duration)
    """
    df = dwell_table.copy().reset_index(drop=True)
    df = df.sort_values("start_time")

    # Columns used to make other calculations; add extra values on edges to account for shift
    df["diff_location"] = (df["zone"] != df["zone"].shift(-1)).shift(1)
    df.loc[0, "diff_location"] = True

    df["time_since_last_visit"] = ((df["start_time"].shift(-1) - df["stop_time"]).shift(1)) / np.timedelta64(1, 's')
    df.loc[0, "time_since_last_visit"] = 0.0

    # Do gnarly pandas calculations!
    # Propagate start times "downwards" for visits meeting criteria:
    #   Propagate if the location hasn't changed and if the time falls below the threshold
    last_result = pd.Series([0])
    while not df["start_time"].equals(last_result):
        last_result = df["start_time"].copy()
        df.loc[~((df["time_since_last_visit"] > min_visit_timespan) |
                 (df["diff_location"])), "start_time"] = df["start_time"].shift(1)

    # Aggregate result by start time to get records grouped by visit
    agg_dwells = df.groupby("start_time")

    # Build the list of visits from the aggregated results
    filtered_visits = pd.DataFrame()
    for start, frame in agg_dwells:
        stop = frame["stop_time"].max()
        dur = (stop - start) / np.timedelta64(1, 's')
        loc = frame.iloc[0]["zone"]
        row = pd.DataFrame([[start, stop, loc, dur]], columns=["start_time", "stop_time", "zone", "duration"])
        filtered_visits = pd.concat([filtered_visits, row], axis=0, ignore_index=True)

    # Remove visits with a duration too short
    filtered_visits = filtered_visits[filtered_visits["duration"] > min_visit_duration]
    filtered_visits.reset_index()
    return filtered_visits


def build_visits(dwell_table, min_visit_timespan, min_visit_duration):
    """
    Takes the identified dwells and turns them into visits.
    :param dwell_table: a pandas dataframe containing the dwells from refine_hotspots()
    :param min_visit_timespan: the minimum number of seconds that must elapse between visits to be considered different
    :param min_visit_duration: the minimum number of seconds a dwell must be to be considered a visit
    :return: a dataframe of visits containing the columns (zone, start_time, end_time, duration)
    """
    # filter visits (only need one call unlike original code)
    visit_table = filter_visits(dwell_table, min_visit_timespan, min_visit_duration)

    # TODO figure out timestamp conversion if needed
    # Convert local start and end times for each visit to UTC (or vice versa?)

    visit_table = visit_table.sort_values("start_time")
    return visit_table


# Step 6: Update coordinates for snapped points

def update_snapped_points(gps_data, visits, hotspots, columns=SENSEDOC_COLUMNS):
    """
    Update the GPS points to have the same coordinates as the hotspot where they fall within a visit.
    For all filtered visits, finds the gps points within start and end times for that visit and sets that point's
    location to be the hotspot centroid coordinates instead of the point coordinates
    :param gps_data: a pandas dataframe containing the GPS data
    :param visits: a pandas dataframe containing the visits found in build_visits()
    :param hotspots: a pandas dataframe containing the hotspots refined from refine_hotspots()
    :param columns: a dictionary containing the names of lat, lon, utm_n and utm_e columns in the dataframe.
                May differ between Ethica and Sensedoc data.
    :return: the updated GPS dataframe with coordinates snapped to hotspot coordinates where applicable.
    """
    for i in range(len(visits)):
        gps_data.loc[((gps_data["utc_date"] >= visits.iloc[i]["start_time"]) &
                      (gps_data["utc_date"] <= visits.iloc[i]["stop_time"])), columns["utm_e"]] = \
            hotspots.loc[(hotspots["hotspot_id"] == visits.iloc[i]["zone"])].reset_index(drop=True).iloc[0]["centroid_x"]

        gps_data.loc[((gps_data["utc_date"] >= visits.iloc[i]["start_time"]) &
                      (gps_data["utc_date"] <= visits.iloc[i]["stop_time"])), columns["utm_n"]] = \
            hotspots.loc[(hotspots["hotspot_id"] == visits.iloc[i]["zone"])].reset_index(drop=True).iloc[0]["centroid_y"]

    return gps_data


# Step 7: Build path bouts from GPS, hotspot and visit data

def detect_path(gps_data, from_time, to_time, max_drop_time, min_keep_time, path_df, columns=SENSEDOC_COLUMNS, utm=True):
    """
    Helper method for build_paths() below. Given a set of boundary times, detect GPS points which fall into paths and
    build a path from their coordinates and times.
    :param gps_data: pandas dataframe containing GPS data
    :param from_time: lower filter bounds for GPS data, as a timestamp
    :param to_time: upper filter bounds for GPS data, as a timestamp
    :param max_drop_time: the maximum time in seconds between points, after which to not consider them consecutive
    :param min_keep_time: the minimum duration of a path, in seconds, for it to be valid
    :param path_df: dataframe containing the path data so far
    :param columns: a dictionary containing the names of lat, lon, utm_n and utm_e columns in the dataframe.
                May differ between Ethica and Sensedoc data.
    :param utm: True if the GPS readings are in UTM format, false otherwise.
    :return: the modified path dataframe with any newly discovered paths added
            The path contains its start and end times, duration, its start and end point coordinates, its start and end
            zones, the number of line segments in the path, and the points in the path.
    """

    if utm:
        x_col = columns["utm_e"]
        y_col = columns["utm_n"]
    else:
        x_col = columns["lon"]
        y_col = columns["lat"]

    selected = gps_data.copy()[((gps_data["utc_date"] >= from_time) &
                                (gps_data["utc_date"] < to_time))]

    if len(selected) == 0:
        # Don't give warning for very short intervals, because obviously nothing was selected.
        if (to_time - from_time) /  np.timedelta64(1, 's') > 10:
            print("No points selected within interval:", from_time, " to ", to_time)
        return path_df

    selected = selected.sort_values("utc_date")

    # Split the path as needed by temporal gap
    selected["between_point_time"] = ((selected.shift(-1)["utc_date"] - selected["utc_date"]).shift(1)) \
                                     / np.timedelta64(1, 's')
    selected = selected.reset_index(drop=True)
    selected.loc[0, "between_point_time"] = 0

    # Split into multiple dataframes based on long temporal gaps
    selected["path_id"] = (selected["between_point_time"] > max_drop_time).cumsum()
    split_groups = selected.groupby("path_id")

    # For each collection of points, build a unique sequence of points for the path
    for _, g in split_groups:
        points = selected[[x_col, y_col]]
        unique_sequence = points[~((points[x_col] == points[x_col].shift(-1)) &
                                   (points[y_col] == points[y_col].shift(-1)))]
        selected = selected[~((points[x_col] == points[x_col].shift(-1)) &
                              (points[y_col] == points[y_col].shift(-1)))].reset_index(drop=True)
        segments = [tuple(x) for x in unique_sequence.values]
        if len(segments) > 1:
            # TODO: Format the dates for the found paths (local -> UTM or vice versa)
            duration = (selected["utc_date"].max() - selected["utc_date"].min()) / np.timedelta64(1, 's')
            if duration > min_keep_time:
                new_path = pd.DataFrame([[selected["utc_date"].min(), selected["utc_date"].max(),
                                          duration, segments[0][0], segments[0][1], selected.iloc[0]["zone"],
                                          segments[-1][0], segments[-1][1], selected.iloc[-1]["zone"],
                                          len(segments) - 1, segments]],
                                        columns=list(path_df.columns))
                path_df = pd.concat([path_df, new_path], axis=0)

    return path_df


def build_paths(gps_data, visits, data_gaps, max_drop_time, min_keep_time):
    """
    Build paths from the detected visits and the GPS data.
    :param gps_data: pandas dataframe containing the GPS data
    :param visits: pandas dataframe containing the visit data
    :param data_gaps: pandas dataframe containing the start and stop times of each missing data gap
    :param max_drop_time: the maximum time in seconds between points, after which to not consider them consecutive
    :param min_keep_time: the minimum duration of a path, in seconds, for it to be valid
    :return: a dataframe containing paths with the columns
                ["start_time", "end_time", "duration", "path_start_x", "path_start_y",
                 "path_end_x", "path_end_y", "num_segments", "segments"]
                The segments are a list of (x, y) coordinate tuples as a string, which can be further parsed when the
                dataframe is returned
    """
    # Initialize empty path structure
    path_cols = ["start_time", "stop_time", "duration", "start_x", "start_y", "start_location_id", "stop_x",
                 "stop_y", "stop_location_id", "num_segments", "segments"]

    paths = pd.DataFrame([], columns=path_cols)

    visits.loc[:, "type"] = "visit"

    # Consider significant gaps in GPS data to break trips, as well as visits
    data_gaps.columns = ["start_time", "stop_time"]
    data_gaps.loc[:, "type"] = "gap"
    data_gaps.loc[:, "contained"] = False

    # Fold gaps into visits if they are fully contained within a visit
    # Visits were previously merged if they were less than MIN_VISIT_TIMESPAN apart, so gaps detected as significant may
    #  be contained withing these merged visits, if the SIGNIFICANT_GAP_LENGTH is less than MIN_VISIT_TIMESPAN
    for vindex, visit in visits.iterrows():
        data_gaps["contained"] = (((data_gaps["start_time"] > visit["start_time"]) &
                                   (data_gaps["stop_time"] < visit["stop_time"])) |
                                  (data_gaps["contained"]))

    data_gaps = data_gaps[~data_gaps["contained"]]
    data_gaps = data_gaps[["start_time", "stop_time"]]

    stop_events = pd.concat([data_gaps[["start_time", "stop_time"]], visits[["start_time", "stop_time"]]], axis=0)
    stop_events = stop_events.sort_values("start_time")

    # Loop through stops and find the points in-between stops to be candidate paths.
    from_time = gps_data.iloc[0]["utc_date"]
    for i in range(len(stop_events)):
        to_time = stop_events.iloc[i]["start_time"]
        paths = detect_path(gps_data, from_time, to_time, max_drop_time, min_keep_time, paths)
        # Update reference time
        from_time = stop_events.iloc[i]["stop_time"]

    # After iterating through stops, add the last piece of track which was not previously processed.
    to_time = gps_data.iloc[-1]["utc_date"] + pd.Timedelta(1, unit="s")
    paths = detect_path(gps_data, from_time, to_time, max_drop_time, min_keep_time, paths)

    # Drop duplicated paths (if any).
    paths = paths.drop_duplicates(["start_time", "stop_time"]).reset_index(drop=True)
    return paths


# Optional Step 8: Smooth the paths found in step 7. (Not yet implemented.)

def smooth_paths(paths, min_loop_duration):
    """
    Takes in: raw GPS paths (raw in what way? source?)
    Returns: filtered GPS paths (filtered in what way? source?)

    Filter paths to remove loops
    A / R of a duration less than a certain threshold
    Smoothing the resulting trace (Douglas Peucker type)
    :param paths: dataframe containing the paths from build_paths
    :param min_loop_duration: minimum duration of a loop (identical endpoints), in seconds, for it to be a valid path
    """
    paths = paths[~((paths["duration"] < min_loop_duration) & (paths["start_location_id"] == paths["stop_location_id"]))]
    paths = paths.reset_index(drop=True)

    # filter to check that the path end points are not the same
    # TODO: why?

    # filter the given data according to the where criteria:
    # where filter: the duration is larger than the min duration and the start and end locations are not the same

    # First smoothing: uses ArcPy's simplify cartography. For reference:
    # TODO: how?
    # http://pro.arcgis.com/en/pro-app/tool-reference/cartography/simplify-line.htm

    # arcGIS simplify_cartography
    # arcpy.SimplifyLine_cartography(tmpFiltPath, tmpSmplPath, "BEND_SIMPLIFY",
    #                            "{} Meters".format(self.simplifyTolerance),
    #                            "FLAG_ERRORS", "NO_KEEP", "NO_CHECK")

    # Second smoothing: using ArcPy's smoothline cartography. For reference:
    # TODO: how?
    # http://pro.arcgis.com/en/pro-app/tool-reference/cartography/smooth-line.htm

    # arcGIS smoothline_cartography
    # arcpy.SmoothLine_cartography(tmpSmplPath, cleanPath, "PAEK",
    #                            "{} Meters".format(self.smoothTolerance),
    #                            "FIXED_CLOSED_ENDPOINT","NO_CHECK")

    # return smoothed paths
    return True


#############################################
# Main method for putting it all together:
#############################################

def detect_trips(raw_gps_path, params=DEFAULT_PARAMS, interpolation=False, smoothing=False, columns=SENSEDOC_COLUMNS,
                 code=None):
    """
    Method for detecting trips from GPS data.
    Intended for use with only a single participant's data at a time.
    :param raw_gps_path: path to the GPS file to detect trips on.
    :param params: dictionary containing the parameters for the trip detection, each as described in the documentation.
    :param interpolation: True if gaps in the GPS data are to be interpolated prior to trip detection, False otherwise.
    :param smoothing: True if the detected paths are to be smoothed, False otherwise.
                        Smoothing functions are not yet implemented.
    :param columns: a dictionary containing the names of lat, lon, utm_n and utm_e columns in the dataframe.
                    May differ between Ethica and Sensedoc data.
    :param code: the ESPG code of the data to be projected, as a string. Leave as None if no conversion is needed.
    :return: a pandas dataframe containing the final detected trips,
             a pandas dataframe containing the detected hotspots, and
             a pandas dataframe containing the detected visits.
    """

    # Make temporary scratch folder to store intermediate values and prevent thrashing.
    os.makedirs(os.getcwd() + "/temp", exist_ok=True)

    # Load parameters and initialize needed values
    cell_size = params["CELL_SIZE"]  # meters
    kernel_bandwidth = params["KERNEL_BANDWIDTH"]  # meters
    downsample_frequency = params["DOWNSAMPLE_FREQUENCY"]  # seconds
    max_delay = params["INTERPOLATE_MAX_DELAY"]  # seconds
    max_drop_time = params["INTERPOLATE_MAX_DROP_TIME"]  # seconds
    max_distance = params["INTERPOLATE_MAX_DISTANCE"]  # meters
    frequency = downsample_frequency if downsample_frequency else 1
    interpolate = interpolation
    i_string = ""
    if interpolate:
        i_string = "i"
    i_string += str(downsample_frequency)
    smooth = smoothing
    min_visit_duration = params["MIN_VISIT_DURATION"]
    min_visit_timespan = params["MIN_VISIT_TIMESPAN"]
    buffer_weight = params["BUFFER_WEIGHT"]
    buffer_snap = params["BUFFER_SNAP"]
    min_hotspot_keep = params["MIN_HOTSPOT_KEEP_DURATION"]
    min_path_keep = params["MIN_PATH_KEEP_DURATION"]
    min_loop_duration = params["MIN_LOOP_DURATION"]
    kernel_hotspot_threshold = params["KERNEL_HOTSPOT_THRESHOLD"]
    utm = True

    if utm:
        x_col = columns["utm_e"]
        y_col = columns["utm_n"]
    else:
        x_col = columns["lon"]
        y_col = columns["lat"]

    gps_data = pd.read_csv(raw_gps_path, parse_dates=["utc_date"])

    # Currently assumes dataframe contains the following columns: (interact_id, utc_date, lat, lon, utm_e, utm_n)
    gps_data = format_gps_data(gps_data, columns=columns, code=code)

    print("Characterizing existing gaps in GPS data.")
    get_data_gaps(gps_data)

    # get grid width/height
    # TODO: Make sure the grid/width/height calculations still work with WGS figures
    width = np.ceil(gps_data[x_col].max() - gps_data[x_col].min())
    height = np.ceil(gps_data[y_col].max() - gps_data[y_col].min())
    print("Width: ", width)
    print("Height: ", height)
    print("Cell size: ", cell_size)
    print("Kernel_bandwidth", kernel_bandwidth)

    # Optional stuff before we begin (refining cells, downsampling, interpolating:
    cell_size = resize_cells(width, height, cell_size, kernel_bandwidth)
    print("Cell size after resizing: ", cell_size)

    if downsample_frequency != 0:
        print("Downsampling gps data.")
        gps_data = downsample_trace(gps_data, downsample_frequency)

    if interpolate:
        print("Interpolating over GPS data")
        gps_data = interpolate_over_dropped_periods(gps_data, frequency, max_delay, max_drop_time, max_distance)

    # Temporarily saving kernel to disk to free up memory.
    gps_data.to_csv(os.getcwd() + "/temp/temp_gps_ds" + i_string + ".csv", index=False)

    points = gps_data[[y_col, x_col]].fillna(method="ffill").values
    x_bounds = (gps_data[x_col].min(), gps_data[x_col].max())
    y_bounds = (gps_data[y_col].min(), gps_data[y_col].max())

    print("Extracting hotspots.")
    kernel, zones, hotspots = extract_hotspots(points, x_bounds, y_bounds, kernel_bandwidth=kernel_bandwidth,
                                               qualifiers=i_string, width=int(width / cell_size),
                                               height=int(height / cell_size))

    gps_data = pd.read_csv(os.getcwd() + "/temp/temp_gps_ds" + i_string + ".csv", parse_dates=["utc_date"])
    gps_data["kernel"] = kernel
    gps_data["zone"] = zones

    print("Normalizing the kernel.")
    gps_data = get_norm_modified_kernel(gps_data, buffer_weight)
    print("Linking the GPS points to hotspots.")
    gps_data = link_gps_to_hotspots(gps_data, buffer_snap, kernel_hotspot_threshold)
    print("Refining hotspots and finding dwells.")
    gaps = pd.read_csv(os.getcwd() + "/temp/sig_temp_gaps.csv", parse_dates=["start_time", "stop_time"])
    hotspots, dwells = refine_hotspots(gps_data, hotspots, gaps, min_hotspot_keep)
    print("Building visits from dwells.")
    visit_table = build_visits(dwells, min_visit_timespan, min_visit_duration)
    print("Updating GPS values for snapped points.")
    gps_data = update_snapped_points(gps_data, visit_table, hotspots)
    print("Detecting the final bouts.")

    final_bouts = build_paths(gps_data, visit_table, gaps, max_drop_time, min_path_keep)

    # Refining final bouts if desired
    if smooth:
        print("Simplifying and smoothing bouts.")
        final_bouts = smooth_paths(final_bouts)

    return final_bouts, hotspots, visit_table, gaps


def build_incident_table(paths, hotspots, visits, gaps, pid):
    incident_columns = ["start_time", "stop_time", "start_location_id", "stop_location_id",
                        "start_x", "start_y", "stop_x", "stop_y", "type"]

    # Format gaps to fit in the table
    paths["type"] = "trip"
    visits["type"] = "visit"
    gaps = gaps[~gaps["contained"]]
    gaps = gaps[["start_time", "stop_time"]]
    gaps["type"] = "gap"
    gaps["start_location_id"] = np.NaN
    gaps["stop_location_id"] = np.NaN
    gaps["start_x"] = np.NaN
    gaps["start_y"] = np.NaN
    gaps["stop_x"] = np.NaN
    gaps["stop_y"] = np.NaN

    visits["start_location_id"] = visits["zone"]
    visits["stop_location_id"] = visits["zone"]
    visits["start_x"] = visits["start_location_id"].map(lambda x: hotspots.loc[hotspots["hotspot_id"] == x,
                                                                               "centroid_x"].iloc[0])
    visits["start_y"] = visits["start_location_id"].map(lambda y: hotspots.loc[hotspots["hotspot_id"] == y,
                                                                               "centroid_y"].iloc[0])
    visits["stop_x"] = visits["start_x"]
    visits["stop_y"] = visits["start_y"]

    incidents = pd.concat([paths[incident_columns], visits[incident_columns], gaps[incident_columns]], axis=0)
    incidents.loc[:, "interact_id"] = pid
    incidents = incidents[["interact_id"] + incident_columns]
    incidents = incidents.sort_values("start_time")
    return incidents


if __name__ == "__main__":
    # testing the code above

    TEST_FILE_DIR = "../../../interact_data/trip_detection_test/101001870/gps_test.csv"
    result_paths, result_hotspots, result_visits, missing = detect_trips(TEST_FILE_DIR)

    print("Building incidents table.")
    incidents_table = build_incident_table(result_paths, result_hotspots, result_visits, missing, 101001870)
    if os.path.isfile("../../../interact_data/trip_detection_test/101001870/incidents_table.csv"):
        print("Removing old incidents table.")
        os.remove("../../../interact_data/trip_detection_test/101001870/incidents_table.csv")
    incidents_table.to_csv("../../../interact_data/trip_detection_test/101001870/incidents_table.csv", index=False)
