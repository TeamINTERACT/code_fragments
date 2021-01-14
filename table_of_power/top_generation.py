"""
Usage: python top_generation.py CITY WAVE OUTPUT_DIR [-osme]

Author: Kole Phillips

Connect to the INTERACT database, process the data, and generate a csv file containing per-second data for accelerometer
data, activity counts, gps coordinates in both WGS and UTM, and health data related to the participant.

Arguments:
  CITY: The name of the city the table is to represent
  WAVE: The wave number of the study to represent in the table
  OUTPUT_DIR: Location the output files should be stored in

Options:
  -o: Ignore and overwrite the existing table of power csv
  -s: Split output into individual table of power files
  -m: Merge output from '-s' option into a single file
  -e: Use Ethica data instead of Sensedoc data to create the table of power
"""

import interact_tools as it
from generate_activity_counts import get_activity_counts
import psycopg2 as psy
import pandas as pd
from pyproj import Proj
import geopandas as gpd
from os.path import isfile, isdir
from sys import argv
from wear_time import wear_marking
from datetime import timedelta
from survey_parse import answers

city_surveys = {
    'victoria': 'vic_data',
    'vancouver': 'van_data',
    'montreal': 'mtl_data',
    'saskatoon': 'skt_data'
}

city_letters = {
    'victoria': 'vic',
    'vancouver': 'van',
    'montreal': 'mtl',
    'saskatoon': 'ssk'
}

city_timezones = {
    'victoria': 'Canada/Pacific',
    'vancouver': 'Canada/Pacific',
    'montreal': 'Canada/Eastern',
    'saskatoon': 'Canada/Central'
}

gender_key = {
    1:  "Male",
    2:  "Female",
    3:  "TransgenderMale",
    4:  "TransgenderFemale",
    5:  "Genderqueer/non-conforming",
    6:  "Other",
    77: "Prefer not to answer"
}

city_zones = {
    'victoria': '10U',
    'vancouver': '10U',
    'montreal': '18T',
    'saskatoon': '13U'
}


def in_city(gps_data, sf, city):
    """
    Determines whether the gps point in the data frame is within the polygon provided by the .shp file
    :param gps_data: the data frame containing rows of latitude and longitude coordinates
    :param sf: the shape file which for the city we are attempting to process
    :return: New df column with the following values: 1 if row is in the city, 0 otherwise
    """
    gps_data['in_city'] = 0
    points = gpd.points_from_xy(gps_data.lon, gps_data.lat)
    multipoly = sf.geometry[it.city_sf[city]]
    in_points = gpd.GeoDataFrame(geometry=points, index=gps_data.reset_index()['utcdate'].to_list())
    in_points = in_points.assign(**{'1': in_points.within(multipoly)}).drop(['geometry'], axis=1)
    gps_data.loc[in_points.any(axis=1), 'in_city'] = 1
    return gps_data['in_city']


def psql_get_data(query):
    """
    Creates a connection to the PSQL database and runs a given query, returning the result in a pandas dataframe
    :param query: The SQL query to be run
    :return: A new pandas dataframe containing the results of the query
    """
    conn = psy.connect(**it.get_connection_kwargs())
    to_ret = pd.read_sql(query, conn)
    conn.close()
    return to_ret


if __name__ == "__main__":
    city, wave = it.get_command_args("top_generation.py")
    ver = it.get_last_commit_date()
    if len(argv) < 4:
        print("Usage: python top_generation.py SITE_NAME WAVE_NUMBER OUTPUT_DIR")
        exit()
    output_dir = argv[3]
    if not isdir(output_dir):
        print("Could not locate directory: " + output_dir)
    if wave < 10:
        out_fname = output_dir + '/' + city + '_0' + str(wave) + '_table_of_power_' + ver
    else:
        out_fname = output_dir + '/' + city + '_' + str(wave) + '_table_of_power_' + ver
    if '-e' in argv:
        out_fname = out_fname + '_ethica' + '.csv'
        ethica = True
    else:
        out_fname = out_fname + '_sd' + '.csv'
        ethica = False

    participants = it.iid_list(city, wave, ethica=ethica)
    if 'iid' in participants.columns:
        participants['interact_id'] = participants['iid']
    if len(argv) > 5 and argv[4].isdigit() and argv[5].isdigit():
        high = int(argv[5])
        low = int(argv[4])
        participants = participants.loc[(participants['interact_id'] >= low) & (participants['interact_id'] < high)]
    f = open('debug.txt', 'w+')

    participants.dropna(inplace=True)
    print(participants['interact_id'].tolist())

    # loading the .shp file for the specified city
    sf = gpd.GeoDataFrame.from_file('CMA/INTERACT_CMA_EPSG4326.shp')

    if isfile(out_fname) and '-o' not in argv:
        data_chunks = pd.read_csv(out_fname, chunksize=10000)
        processed = []
        for chunk in data_chunks:
            processed = processed + chunk['interact_id'].drop_duplicates().to_list()
        processed = list(set(processed))
        header = False
    else:
        processed = []
        header = True

    linkage = pd.DataFrame()
    if city == 'victoria' and wave == 1:
        linkage = pd.read_csv('~/projects/def-dfuller/interact/permanent_archive/Victoria/Wave1/linkage.csv')

    # Merging existing files generated by the '-s' option
    if '-m' in argv:
        for p in participants.itertuples():
            print("Participant " + str(p.interact_id) + ".")
            if p.interact_id in processed:
                print("Participant has already been processed.\n")
                continue
            if ethica:                
                in_fname = output_dir + '/' + str(p.interact_id) + '_table_of_power_' + it.get_last_commit_date() + '_ethica.csv'
            else:
                in_fname = output_dir + '/' + str(p.interact_id) + '_table_of_power_' + it.get_last_commit_date() + '_sd.csv'
            if not isfile(in_fname):
                print("File does not exist.")
                continue
            p_top = pd.read_csv(in_fname)
            if isfile(out_fname):
                p_top.to_csv(out_fname, header=False, mode='a')
            else:
                p_top.to_csv(out_fname)
        print("Merge complete.")
        exit()
    aborted = []
    for p in participants.itertuples():
        print(p.interact_id)
        if '-s' in argv:
            out_fname = output_dir + '/' + str(p.interact_id) + '_table_of_power_' + it.get_last_commit_date()
            if ethica:
                out_fname = out_fname + '_ethica' + '.csv'
            else:
                out_fname = out_fname + '_sd' + '.csv'

            if '-o' not in argv and isfile(out_fname):
                print("Participant has already been processed.\n")
                continue

        if p.interact_id in processed:
            print("Participant has already been processed.\n")
            continue

        # Get the start and end times for the participant so we can exclude data that is not a part of the study
        try:
            if ethica:
                querystr = """
                select start_date, end_date from portal_dev.ethica_assignments where interact_id = %s;
                """ % (p.interact_id,)
            else:
                querystr = """
                select started_wearing, stopped_wearing from portal_dev.sensedoc_assignments where interact_id = %s;
                """ % (p.interact_id,)
            start_date = pd.Timestamp(psql_get_data(querystr).started_wearing[0]).tz_localize(city_timezones[city])
            end_date = pd.Timestamp(psql_get_data(querystr).stopped_wearing[0]).tz_localize(city_timezones[city]) + timedelta(days=1)
        except:
            start_date = '2015-01-01 00:00:00-00:00'
            end_date = '2030-01-01 00:00:00-00:00'

        # Gather health data. Constant for participant's entire entry
        # NOTE: This section the section most likely to require adjustments between cities and waves due to the
        # different survey structures. Make note of these differences and edit the script accordingly.
        health_data = {'interact_id': p.interact_id,
                       'age': -1,
                       'gender': [],
                       'city_id': city,
                       'wave_id': wave,
                       'income': '',
                       'education': '',
                       'ethnicity': ''}
        if wave == 1:
            if city == 'saskatoon':
                querystr = """
                select income, education, gender, group_id_skt from lut.health_1skt_main where interact_id::varchar = %s::varchar;
                """ % (str(p.interact_id), )
            elif city == 'victoria':
                querystr = """
                select income, group_id from lut.health_1%s_main where interact_id::varchar = %s::varchar;
                """ % (city_letters[city], str(p.interact_id), )

            else:
                querystr = """
                select income, education, gender, group_id_%s from lut.health_1%s_main where interact_id::varchar = %s::varchar;
                """ % (city_letters[city], city_letters[city], str(p.interact_id), )
            survey_data = psql_get_data(querystr)
            if survey_data.empty:
                continue
            survey_data = survey_data.iloc[0]
            health_data['income'] = answers['income'][survey_data.income]
            if city == 'victoria':
                trek_id = linkage[linkage.interact_id == p.interact_id].treksoft_id.tolist()[0]
                gen_query = """select data from survey.vic_data where pid = %s""" % (int(trek_id), )
                for d in list(psql_get_data(gen_query).data):
                    if 'Eligibility_Q2' in d:
                        gk = ['', 'Male', 'Female', 'Trans', 'Other']
                        health_data['gender'] = ''
                        for g in d['Eligibility_Q2']:
                            health_data['gender'] = health_data['gender'] + gk[g] + ', '
                        health_data['gender'] = health_data['gender'][:-2]
                    if 'Eligibility_Q1_C3' in d:
                        health_data['age'] = 2017 - d['Eligibility_Q1_C3']
            else:
                health_data['education'] = answers['education'][survey_data.education]
                health_data['gender'] = gender_key[survey_data.gender]
            if city == 'victoria':
                eth = list(survey_data['group_id'])
            else:
                eth = list(survey_data['group_id_' + city_letters[city]])
            for i in eth:
                if i.isdigit():
                    health_data['ethnicity'] = health_data['ethnicity'] + answers['ethnicity'][int(i)] + ', '
            health_data['ethnicity'] = health_data['ethnicity'][:-2]
            #if type(s['gender']) is int:
            #    health_data['gender'] = gender_key[s['gender']]
            #else:
            #    try:
            #        health_data['gender'] = str([gender_key[x] for x in s['gender']])[1:-1]
            #    except:
            #        health_data['gender'] = []
            if city == 'saskatoon':
                querystr = """
                select added, birthdate from survey.skt_eligibility where participant_id = %s;
                """ % (str(p.interact_id),)
                age_data = psql_get_data(querystr)
                try:
                    health_data['age'] = age_data.added[0].year - age_data.birthdate[0].year
                except:
                    health_data['age'] = -1

            #querystr2 = """
            #select * from survey.health_%s%s_main where treksoft_id = %s;
            #""" % (str(wave), city_letters[city], trek_id)
            #health_survey = psql_get_data(querystr2)
            #querystr3 = """
            #select session_created_at from survey.%s_data where uid = %s;
            #""" % (city_letters[city], trek_id)
            #try:
            #    survey_year = psql_get_data(querystr3).session_created_at[0].year
            #    health_data['age'] = survey_year - survey_data.birth_date[0].year
            #except:
            #    health_data['age'] = -1

        else:
            querystr = """
            select * from survey%s.%s_data where iid = %s
            """ % (str(wave), city_letters[city], str(p.interact_id))
            survey_data = psql_get_data(querystr)
            if not survey_data.empty:
                for s in survey_data.itertuples():
                    if s.data is None:
                        continue
                    responses = dict(s.data)
                    if 'gender' in responses:
                        health_data['gender'] = gender_key[responses['gender']]
                    if 'income' in responses:
                        health_data['income'] = answers['income'][responses['income']]
                    if 'birth_date' in responses:
                        health_data['age'] = int(str(s.date_completed)[0:4]) - int(str(responses['birth_date'])[0:4])
                    else:
                        try:
                            querystr2 = """
                            select age from level_1second.table_of_power where interact_id = %s limit 1
                            """ % (p.interact_id,)
                            health_data['age'] = int(psql_get_data(querystr2).age[0])
                        except:
                            health_data['age'] = -1
                    if 'education' in responses:
                        health_data['education'] = answers['education'][responses['education']]

        # Get SenseDoc accelerometer data
        if ethica:
            # TODO: Replace with appropriate table when available
            querystr = """
            select record_time, x, y, z from level_0.%s_w1_eth_xls_delconflrec
            where iid = %s and record_time >= '%s' and record_time < '%s';
            """ % (city_letters[city], p.interact_id, start_date, end_date)
        else:
            querystr = """
            select ts, x, y, z from level_0.sd_accel 
            where (iid = %s) and (ts > '%s') and (ts < '%s');
            """ % (p.interact_id, start_date, end_date)
        print("Collecting accel data.")
        accel_data = psql_get_data(querystr).drop_duplicates()
        accel_data.columns = ['utcdate', 'x', 'y', 'z']
        counts = pd.DataFrame(index=['utcdate'], columns=['summary_count'])
        print("Processing accel data.")
        if not accel_data.empty:
            accel_data['utcdate'] = pd.to_datetime(accel_data.utcdate, utc=True)
            if city == 'victoria':
                accel_data = accel_data.set_index('utcdate').resample('20L', how='median').reset_index().dropna(subset=['x'])
            # Compute activity counts
            counts = get_activity_counts(accel_data)
            counts.set_index(['utcdate'], inplace=True)
            counts = wear_marking(counts, epoch='1S')
            accel_data.set_index(['utcdate'], inplace=True, drop=True)
            # Get average accel data for each second
            accel_data = accel_data.resample('s').mean().dropna()
            accel_data.index.round('s')
            accel_data.drop(['x', 'y', 'z'], axis=1, inplace=True)
        else:
            print("No accel data found.")
            aborted.append(p.interact_id)
            continue
        print("Accel processing complete.")

        # Get GPS data
        if ethica:
            # TODO: Replace with appropriate table when available
            querystr = """
            select record_time, lat, lon from level_0.%s_w1_eth_gps_delconflrec
            where iid = %s and record_time >= '%s' and record_time < '%s';
            """ % (city_letters[city], p.interact_id, start_date, end_date)
        else:
            querystr = """
            select ts, lat, lon from level_0.sd_gps 
            where iid = %s and ts > '%s' and ts < '%s';
            """ % (p.interact_id, start_date, end_date)
        print("Collecting gps data.")
        gps_data = psql_get_data(querystr).drop_duplicates()
        gps_data.columns = ['utcdate', 'lat', 'lon']
        print("Processing gps data.")
        if not gps_data.empty:
            gps_data['utcdate'] = pd.to_datetime(gps_data.utcdate, utc=True)
            gps_data.set_index(['utcdate'], inplace=True, drop=True)
            # Get average gps data for each second
            try:
                gps_data = gps_data.resample('1S').mean()
                gps_data = gps_data.dropna(subset=['lat', 'lon'])
                gps_data.index.round('1S')
                # Convert coordinates to UTM
                proj = Proj("+proj=utm +zone=" + city_zones[city] + ", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
                gps_data['northing'], gps_data['easting'] = proj(gps_data['lon'].values, gps_data['lat'].values)
                gps_data['zone'] = city_zones[city]
                # Generate column which tells when the participant was within the city limits
                gps_data['in_city'] = in_city(gps_data, sf, city)
            except:
                print("ABORTED\n")
                aborted.append(p.interact_id)
                continue
        else:
            print("No GPS data found.")
            gps_data = pd.DataFrame()

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
        table['income'] = health_data['income']
        table['education'] = health_data['education']
        table['ethnicity'] = health_data['ethnicity']
        table['city_id'] = health_data['city_id']
        table['wave_id'] = health_data['wave_id']
        table = table.reset_index().set_index(['interact_id', 'utcdate'])
        print("Merge complete.\n")

        if '-s' in argv:
            table.to_csv(out_fname)
        elif header and (not isfile(out_fname)):
            # Create a new csv file if this is the first participant processed
            table.to_csv(out_fname)
            header = False
        else:
            # Otherwise, append this participant's data to the end of the file
            table.to_csv(out_fname, header=False, mode='a')

    print("Table constructed.")
    print("Aborted:")
    print(aborted)
    exit()
