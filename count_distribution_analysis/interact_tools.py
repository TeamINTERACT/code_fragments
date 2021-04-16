"""
A set of functions and routines that are frequently used from
within other Interact project scripts.
"""
import os
import psycopg2 as psy
import pandas as pd
from sys import argv
import numpy as np


def trim_10k(arr):
    """
    Remove all items from a list with a value greater than 10,000
    :param arr: The initial list
    :return: The input list without any values greater than 10,000
    """
    return list(np.array(arr)[np.array(arr) < 10000])


def get_connection_kwargs():
    return {
        'database': os.environ.get('DB_NAME'),
        'user': os.environ.get('DB_USER'),
        'password': os.environ.get('DB_PASSWORD'),
        'host': os.environ.get('DB_HOST'),
        'connect_timeout': int(os.environ.get('DB_TIMEOUT')),
    }


def psql_get_data(query):
    """
    Opens a connection to the PSQL server, sends a query, and returns the result as a pandas dataframe after closing
    the connection
    :param query: The query to be sent, as a string
    :return: a pandas dataframe with the results of the query
    """
    conn = psy.connect(**get_connection_kwargs())
    to_ret = pd.read_sql(query, conn)
    conn.close()
    return to_ret


def get_command_args(scriptname=argv[0]):
    """
    Standard function which ensures the function is provided with a city name and wave number
    :param scriptname: The name of the script being run
    :return: The city and wave given as a command zone argument
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


def top_counts(iid, table='1minute', utcas='ts', countas='counts'):
    """
    Retrieves the x-axis activity counts and their corresponding timestamps for a given participant
    :param iid: The ID of the participant in question
    :param table: Which version of the ToP to use, 1second or 1minute
    :param utcas: Column name assigned to the UTC timestamp
    :param countas: Column name assigned to the x_axis activithy counts
    :return: A pandas dataframe containing the specified information
    """
    query = """
    select utcdate as %s, x_count as %s, interact_id as iid from level_%s.table_of_power where interact_id = %s;
    """ % (utcas, countas, table, iid)
    return psql_get_data(query)


cities = {
    "victoria": 1,
    "vancouver": 2,
    "saskatoon": 3,
    "montreal": 4
}


def top_iid_list(city, wave):
    """
    Returns a list of every participant ID in the database corresponding to the provided city and wave
    :param city: The name of the city the study took place in
    :param wave: The number corresponding to the study in question
    :return: a list with the ID of every participant in the requested study
    """
    query_str = """
    select distinct interact_id from level_1second.table_of_power where city_id = '%s' and wave_id = %s
    """ % (city, wave)
    return psql_get_data(query_str).interact_id.tolist()
