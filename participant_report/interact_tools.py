"""
A set of functions and routines that are frequently used from
within other Interact project scripts.
"""
import os
import psycopg2 as psy
from sys import argv
import pandas as pd


def get_connection_kwargs():
    return {
        'database': os.environ.get('YAKI_DATABASE'),
        'user': os.environ.get('YAKI_USER'),
        'password': os.environ.get('YAKI_PASSWORD'),
        'host': os.environ.get('YAKI_HOST'),
        'connect_timeout': int(os.environ.get('YAKI_TIMEOUT')),
    }


city_names = { 1: "victoria",
               2: "vancouver",
               3: "saskatoon",
               4: "montreal",
               5: "unknown",}

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


def get_id_list(city, wave):
    """
    Returns a list of every participant ID in the database corresponding to the provided city and wave
    """
    id_range_low = cities[city] * 100000000 + int(wave) * 1000000

    kwargs = get_connection_kwargs()
    connection = psy.connect(**kwargs)
    cursor = connection.cursor()
    # query_str = """select interact_id, treksoft_id_uid, treksoft_id_pid
    #                from master_id
    #                where interact_id = 101549282"""
    query_str = """select interact_id, treksoft_id_uid, treksoft_id_pid
                   from master_id
                   where interact_id > %s and interact_id < %s""" % (id_range_low, id_range_low + 999999)

    ids = pd.read_sql(query_str, connection)
    cursor.close()
    connection.close()

    return ids


def get_command_args(scriptname):
    """
    Standard function which validates
    :param scriptname:
    :return:
    """
    if len(argv) < 3:
        print("Usage: python " + scriptname + " SITE_NAME WAVE_NUMBER")
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
