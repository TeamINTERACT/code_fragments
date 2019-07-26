"""
Author: Kole Phillips

Connect to the INTERACT database, process the data, and generate a csv file containing per-second data for accelerometer
data, activity counts, gps coordinates in both WGS and UTM, and health data related to the participant.

Usage: python top_generation.py CITY WAVE
  CITY: The name of the city the table is to represent
  WAVE: The wave number of the study to represent in the table

Options:
  -o: Ignore and overwrite the existing table of power csv
"""
import geopandas

import interact_tools as it
from generate_activity_counts import get_activity_counts
import psycopg2 as psy
import pandas as pd
from pyproj import Proj
import geopandas as gpd
from os.path import isfile, isdir
from os import remove
from sys import argv
from wear_time import wear_marking
from datetime import timedelta, datetime
import os

city_surveys = {
    'victoria': 'vic_data',
    'vancouver': 'van_data',
    'montreal': 'mtl_data',
    'saskatoon': 'skt_data'
}


city_timezones = {
    'victoria': 'Canada/Pacific',
    'vancouver': 'Canada/Pacific',
    'montreal': 'Canada/Eastern',
    'saskatoon': 'Canada/Central'
}


gender_key = {
    'victoria': {
        1: "Male",
        2: "Female",
        3: "Transgender",
        4: "Other"
    }
}


city_sf = {
    'victoria': "CMA/INTERACT_CMA_EPSG4326.shp"
}


city_zones = {
    'victoria': '10U',
    'vancouver': '10U',
    'montreal': '18T',
    'saskatoon': '13U'
}


def in_city(gps_data, sf):
    """
    Determines whether the gps point in the data frame is within the polygon provided by the .shp file
    :param gps_data: the data frame containing rows of latitude and longitude coordinates
    :param sf: the shape file which for the city we are attempting to process
    :return: New df column with the following values: 1 if row is in the city, 0 otherwise
    """
    # point = Point(row.lon, row.lat)
    # # for poly in shpfile.geometry:
    # if point.within(poly):
    #     return 1
    # return 0
    gps_data['in_city'] = 0
    points = geopandas.points_from_xy(gps_data.lon, gps_data.lat)
    in_points = geopandas.GeoDataFrame(geometry=points, index=gps_data.reset_index()['utcdate'].to_list())
    in_points = in_points.assign(**{str(key): in_points.within(geom) for key, geom in sf.geometry.items()}).drop(
        ['geometry'], axis=1)
    gps_data.loc[in_points.any(axis=1), 'in_city'] = 1
    return gps_data['in_city']


if __name__ == "__main__":
    city, wave = it.get_command_args("top_generation.py")
    participants = it.get_id_list(city, wave)
    if len(argv) < 4:
        print("Usage: python top_generation.py SITE_NAME WAVE_NUMBER OUTPUT_DIR")
        exit()
    output_dir = argv[3]
    if not isdir(output_dir):
        print("Could not locate directory: " + output_dir)
    if len(argv) > 5:
        high = int(argv[5])
        low = int(argv[4])
        participants = participants.loc[(participants['interact_id'] >= low) & (participants['interact_id'] < high)]

    if wave < 10:
        out_fname = output_dir + '/' + city + '_0' + str(wave) + '_table_of_power_' \
                + str(datetime.fromtimestamp(os.stat('.git/FETCH_HEAD').st_mtime))[0:10] + '.csv'
    else:
        out_fname = output_dir + '/' + city + '_' + str(wave) + '_table_of_power_' \
                + str(datetime.fromtimestamp(os.stat('.git/FETCH_HEAD').st_mtime))[0:10] + '.csv'

    participants = participants[['interact_id', 'treksoft_id_uid']].dropna()
    print(participants['interact_id'].tolist())

    if wave < 10:
        start_end = pd.read_csv(city + "_0" + str(wave) + "_start_end.csv")
    else:
        start_end = pd.read_csv(city + "_" + str(wave) + "_start_end.csv")

    kwargs = it.get_connection_kwargs()
    connection = psy.connect(**kwargs)
    cursor = connection.cursor()

    # loading the .shp file for the specified city (only Victoria for now)
    sf = gpd.GeoDataFrame.from_file(city_sf[city])

    # find the link between the two disjoint INTERACT IDs each participant has (situation unique to Victoria Wave 1)
    if city == 'victoria' and wave == 1:
        lut = pd.read_csv('xvic_lut.csv', delimiter=';')
        sdid = pd.read_csv('pairings_with_sdid.csv')
        pairs = lut.set_index('sensedoc_id').join(sdid.set_index('sensedoc_id'), lsuffix='_lut', rsuffix='_sdid')
        pairs = pairs[pairs.index.notnull()]
        pairs = pairs[['interact_id_lut', 'interact_id_sdid', 'treksoft_id_lut']].dropna().astype(int).set_index('interact_id_lut')
    else:
        pairs = pd.DataFrame

    if isfile(out_fname) and not '-o' in argv:
        processed = pd.read_csv(out_fname)['interact_id'].drop_duplicates().to_list()
        header = False
    else:
        if isfile(out_fname):
            remove(out_fname)
        processed = []
        header = True

    for p in participants.itertuples():
        print("Participant " + str(p.interact_id) + ".")
        if city == 'victoria' and wave == 1 and p.interact_id not in pairs.index:
            print("No SD data found for participant.\n")
            continue
        if p.interact_id in processed:
            print("Participant has already been processed.\n")
            continue

        # Get the start and end times for the participant so we can exclude data that is not a part of the study
        if start_end[start_end.interact_id == p.interact_id].empty:
            start_date = '2000-01-01 00:00:00-00:00'
            end_date = '2100-01-01 00:00:00-00:00'
        else:
            start_date = pd.Timestamp(start_end[start_end.interact_id == p.interact_id]['start'].item()).tz_localize(city_timezones[city])
            end_date = pd.Timestamp(start_end[start_end.interact_id == p.interact_id]['end'].item()).tz_localize(city_timezones[city]) + timedelta(days=1)

        # Gather health data. Constant for participant's entire entry
        health_data = {'interact_id': p.interact_id,
                       'age': -1,
                       'gender': [],
                       'city_id': city,
                       'wave_id': wave}
        if city == 'victoria' and wave == 1:
            querystr = """
            select * from survey.vic_data where pid = %s;
            """ % pairs.loc[p.interact_id].treksoft_id_lut
        else:
            querystr = """
            select * from survey.%s where uid = %s;
            """ % (city_surveys[city], p.interact_id)
        survey_data = pd.read_sql(querystr, connection)
        if not survey_data.empty:
            survey_year = survey_data.updated_at[0].year
            survey = survey_data.rawdata[0]
            if survey is not None and survey != []:
                if 'Eligibility_Q1_C3' in survey:
                    health_data['age'] = survey_year - int(survey['Eligibility_Q1_C3'])
                if 'Eligibility_Q2' in survey:
                    gender_raw = survey['Eligibility_Q2']
                    gender_str = []
                    for g in gender_raw:
                        gender_str.append(gender_key[city][g])
                    health_data['gender'] = gender_str
            print("Health data collected.")

        # Get SenseDoc accelerometer data
        if city == 'victoria' and wave == 1:
            querystr = """
            select utc_date, x_acc_sd, y_acc_sd, z_acc_sd from sd_accel_raw_test 
            where (interact_id = %s) and (utc_date >= '%s') and (utc_date < '%s');
            """ % (pairs.loc[p.interact_id].interact_id_sdid, start_date, end_date)
        else:
            querystr = """
            select utc_date, x_acc_sd, y_acc_sd, z_acc_sd from sd_accel_raw_test 
            where (interact_id = %s) and (utc_date >= '%s') and (utc_date < '%s');
            """ % (p.interact_id, start_date, end_date)
        print("Collecting accel data.")

        # Pull from cache if it exists, otherwise create the cache for the participant
        if isfile("cache/" + str(p.interact_id) + "_accel.csv") and '-r' not in argv:
            accel_data = pd.read_csv("cache/" + str(p.interact_id) + "_accel.csv")
        else:
            accel_data = pd.read_sql(querystr, connection)
            accel_data.columns = ['utcdate', 'x', 'y', 'z']
            # accel_data.to_csv("cache/" + str(p.interact_id) + "_accel.csv")
        counts = pd.DataFrame(index=['utcdate'], columns=['summary_count'])
        print("Processing accel data.")
        if not accel_data.empty:
            accel_data['utcdate'] = pd.to_datetime(accel_data.utcdate)
            # Compute activity counts
            counts = get_activity_counts(accel_data)
            counts.set_index(['utcdate'], inplace=True)
            counts = wear_marking(counts, epoch='1S')
            accel_data.set_index(['utcdate'], inplace=True, drop=True)
            # Get average accel data for each second
            accel_data = accel_data.resample('s').mean().dropna()
            accel_data.index.round('s')
            accel_data.drop(['x', 'y', 'z'], axis=1, inplace=True)
        print("Accel processing complete.")

        # Get SenseDoc GPS data
        if city == 'victoria' and wave == 1:
            querystr = """
            select utc_date, x_wgs_sd, y_wgs_sd, speed_sd, alt_sd from sd_gps_raw_test 
            where interact_id = %s and utc_date >= '%s' and utc_date < '%s';
            """ % (pairs.loc[p.interact_id].interact_id_sdid, start_date, end_date)
        else:
            querystr = """
            select utc_date, x_wgs_sd, y_wgs_sd, speed_sd, alt_sd from sd_gps_raw_test 
            where interact_id = %s and utc_date >= '%s' and utc_date < '%s';
            """ % (p.interact_id, start_date, end_date)
        print("Collecting gps data.")

        # Pull from cache if it exists, otherwise create the cache for the participant
        if isfile("cache/" + str(p.interact_id) + "_gps.csv") and '-r' not in argv:
            gps_data = pd.read_csv("cache/" + str(p.interact_id) + "_gps.csv")
        else:
            gps_data = pd.read_sql(querystr, connection)
            gps_data.columns = ['utcdate', 'lon', 'lat', 'speed', 'alt']
            # gps_data.to_csv("cache/" + str(p.interact_id) + "_gps.csv")
        print("Processing gps data.")
        if not gps_data.empty:
            gps_data['utcdate'] = pd.to_datetime(gps_data.utcdate)
            gps_data.set_index(['utcdate'], inplace=True, drop=True)
            # Get average gps data for each second
            gps_data = gps_data.resample('1S').mean().dropna(subset=['lat', 'lon'])
            gps_data.index.round('1S')
            # Convert coordinates to UTM
            proj = Proj("+proj=utm +zone=" + city_zones[city] + ", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
            gps_data['northing'], gps_data['easting'] = proj(gps_data['lon'].values, gps_data['lat'].values)
            gps_data['zone'] = city_zones[city]
            # Generate column which tells when the participant was within the city limits
            gps_data['in_city'] = in_city(gps_data, sf)
        print("GPS processing complete.")

        # Combine each piece of processed data into one table
        print("Merging accel and gps data.")
        table = accel_data.join(gps_data).join(counts)
        table.dropna(subset=['summary_count'], inplace=True)
        table[['x_count', 'y_count', 'z_count', 'wearing', 'in_city']] = \
            table[['x_count', 'y_count', 'z_count', 'wearing', 'in_city']].astype('Int64')
        print("Merging health data.")
        table['interact_id'] = health_data['interact_id']
        table['age'] = health_data['age']
        table['gender'] = str(health_data['gender'])
        table['city_id'] = health_data['city_id']
        table['wave_id'] = health_data['wave_id']
        table = table.reset_index().set_index(['interact_id', 'utcdate'])
        print("Merge complete.\n")

        if header and (not isfile(out_fname)):
            # Create a new csv file if this is the first participant processed
            table.to_csv(out_fname)
            header = False
        else:
            # Otherwise, append this participant's data to the end of the file
            table.to_csv(out_fname, header=False, mode='a')

    print("Table constructed.")
    cursor.close()
    connection.close()
    exit()
