"""
Author: Kole Phillips

Usage: python top_generation.py CITY WAVE
  CITY: The name of the city the table is to represent
  WAVE: The wave number of the study to represent in the table
"""

from Art_activity_count_algos import counts
import interact_tools as it
import psycopg2 as psy
import pandas as pd
import numpy as np


cities = {
    "victoria": 1,
    "vancouver": 2,
    "saskatoon": 3,
    "montreal": 4
}


def get_activity_counts(accel_df):
    """
    Calculates the activity counts per second from the provided accelerometer readings
    :param accel_df: A pandas dataframe containing the x, y, and z accelerometer data with a readings frequency of 50 Hz
    :return:
    """
    accel_df['utc'] = accel_df.utcdate
    print("Calculating counts.")
    x_activity_count = list(counts(accel_df.x, 50))
    y_activity_count = list(counts(accel_df.y, 50))
    z_activity_count = list(counts(accel_df.z, 50))
    times = accel_df.set_index('utcdate').resample('1s').mean().dropna().reset_index()['utcdate'].astype(str).to_list()

    if len(times) > len(x_activity_count):
        times = times[0:len(x_activity_count)]

    count_sum = np.power(np.power(x_activity_count, 2) + np.power(y_activity_count, 2) + np.power(z_activity_count, 2),
                         0.5)

    to_ret = pd.DataFrame(data={'utcdate': times,
                                'x_count': x_activity_count,
                                'y_count': y_activity_count,
                                'z_count': z_activity_count,
                                'summary_count': count_sum})
    to_ret.x_count = to_ret.x_count.astype(int)
    to_ret.y_count = to_ret.y_count.astype(int)
    to_ret.z_count = to_ret.z_count.astype(int)
    to_ret.utcdate = pd.to_datetime(to_ret.utcdate)

    return to_ret
    

if __name__ == "__main__":
    city, wave = it.get_command_args("generate_activity_counts.py")
    inter_ids = it.get_id_list(city, wave)
    inter_ids = inter_ids.interact_id.values

    kwargs = it.get_connection_kwargs()
    connection = psy.connect(**kwargs)
    cursor = connection.cursor()

    for participants in inter_ids:
        participant = participants
        out_fname = "output/" + str(participant) + "_activity_counts.csv"
        querystr = """
            select utc_date, x_acc_sd, y_acc_sd, z_acc_sd from sd_accel_raw_test where interact_id = %s;
            """ % participant
        print("Collecting data for participant " + str(participant) + ".")
        data = pd.read_sql(querystr, connection)
        data.columns = ['utcdate', 'x', 'y', 'z']

        if data.empty:
            print("No data.\n")
            continue

        out_counts = get_activity_counts(data)

        out_counts.to_csv(out_fname, sep=',', index=False)
        print("Counts saved.\n")

    cursor.close()
    connection.close()
