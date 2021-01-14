"""
A set of functions and routines that are frequently used from
within other Interact project scripts.
"""
import os
import psycopg2 as psy
from sys import argv
import pandas as pd
import git
from datetime import datetime


def get_connection_kwargs():
    return {
        'database': os.environ.get('DB_NAME'),
        'user': os.environ.get('DB_USER'),
        'password': os.environ.get('DB_PASSWORD'),
        'host': os.environ.get('DB_HOST'),
        'connect_timeout': int(os.environ.get('DB_TIMEOUT')),
    }


def get_connection_kwargs_legacy():
    return {
        'database': os.environ.get('YAKI_DATABASE'),
        'user': os.environ.get('YAKI_USER'),
        'password': os.environ.get('YAKI_PASSWORD'),
        'host': os.environ.get('YAKI_HOST'),
        'connect_timeout': int(os.environ.get('YAKI_TIMEOUT')),
    }

city_zones = {
    'victoria': [10, 'U'],
    'vancouver': [10, 'U'],
    'montreal': [18, 'T'],
    'saskatoon': [13, 'U']
}

city_names = {
    1: "victoria",
    2: "vancouver",
    3: "saskatoon",
    4: "montreal",
    5: "unknown"
}

city_sf = {
    'montreal': 0,
    'saskatoon': 1,
    'vancouver': 2,
    'victoria': 3
}

cities = {
    "victoria": 1,
    "vancouver": 2,
    "saskatoon": 3,
    "montreal": 4
}


def get_city_from_id(interact_id):
    """Helper function to decode an id and extract city name."""
    city_num = int(interact_id / 100000000)
    if city_num in city_names:
        return city_names[city_num]
    return "really unknown"


def get_id_list(city, wave, minbound=0, maxbound=10 ** 10, legacy=False):
    """
    Returns a list of every participant ID in the database corresponding to the provided city and wave
    """
    id_range_low = cities[city] * 100000000 + int(wave) * 1000000
    if legacy:
        kwargs = get_connection_kwargs_legacy()
    else:
        kwargs = get_connection_kwargs()
    connection = psy.connect(**kwargs)
    cursor = connection.cursor()
    # query_str = """select interact_id, treksoft_id_uid, treksoft_id_pid
    #                from master_id
    #                where interact_id = 101549282"""
    query_str = """select interact_id, treksoft_id_uid, treksoft_id_pid
                   from master_id
                   where interact_id > %s and interact_id < %s""" % (max(id_range_low, minbound),
                                                                     min(id_range_low + 999999, maxbound))

    ids = pd.read_sql(query_str, connection)
    cursor.close()
    connection.close()

    return ids


ethica_studies = {
    'victoria': {1: [212]},
    'saskatoon': {1: [448], },
    'vancouver': {1: [376], },
    'montreal': {1: [395, 396]},
}


def iid_list(city, wave, minbound=0, maxbound=10 ** 10, ethica=False):
    """
    Returns a list of every participant ID in the database corresponding to the provided city and wave
    """

    kwargs = get_connection_kwargs()
    connection = psy.connect(**kwargs)
    cursor = connection.cursor()
    if ethica:
        ids = pd.DataFrame()
        for study in ethica_studies[city][wave]:
            query_str = """select distinct interact_id from portal_dev.ethica_assignments where interact_id > %s and interact_id < %s and study_id = %s
            """ % (minbound, maxbound, study)
            ids = ids.append(pd.read_sql(query_str, connection))
    else:
        query_str = """select distinct interact_id from portal_dev.sensedoc_assignments where interact_id > %s and interact_id < %s and city_id = %s and wave_id = %s
        """ % (minbound, maxbound, cities[city], wave)
        ids = pd.read_sql(query_str, connection)
    cursor.close()
    connection.close()
    return ids


def get_command_args(scriptname=''):
    """
    Standard function which ensures the function is provided with a city name and wave number
    :param scriptname: The name of the script being run
    :return:
    """
    if len(argv) < 3:
        print("Usage: python " + argv[0] + " SITE_NAME WAVE_NUMBER")
        exit()
    city = argv[1]
    wave = argv[2]
    # Ensure given arguments are valid
    if not city.lower() in cities:
        print("City " + city + " is not a participating city")
        exit()
    try:
        if int(wave) > 99 or int(wave) < 1:
            print("Wave number must be a positive two-digit integer")
            exit()
    except:
        print("Wave number must be a positive two-digit integer")
        exit()
    return city, int(wave)


def get_last_commit_date():
    """
    Gets the date of the current repo's last commit
    :return: the date of the last commit
    """
    repo = git.Repo('.')
    tree = repo.tree()
    date = None
    for blob in tree:
        commit = next(repo.iter_commits(paths=blob.path, max_count=1))
        if date is None or date < datetime.fromtimestamp(commit.committed_date):
            date = datetime.fromtimestamp(commit.committed_date)
    return str(date)[0:10]


def psql_get_data(query):
    conn = psy.connect(**get_connection_kwargs())
    to_ret = pd.read_sql(query, conn)
    conn.close()
    return to_ret
