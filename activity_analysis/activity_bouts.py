import os
from datetime import datetime
from os.path import isfile, isdir
from sys import argv
import pandas as pd
import interact_tools as it
import numpy as np


def get_bouts(df, epoch='H'):
    """
    Get the number of minutes that fall into each activity level for each hour or day
    :param df: the dataframe for one participant's table of power data on the minute level
    :param epoch: the time interval we are aggregating to
    :return: a dataframe containing the number of minutes for each hour or day that fall in to each activity level
    """
    max_min = 0
    if epoch == 'H':
        max_min = 60
    if epoch == 'D':
        max_min = 60*24
    bouts_df = pd.DataFrame(columns=['utcdate']).set_index('utcdate')
    bouts_df['non-wear'] = max_min - df.resample(epoch).sum()['wearing']
    for level in ['sedentary', 'light', 'moderate', 'vigorous']:
        bouts_df[level] = df[df['activity_levels'] == level].resample(epoch).sum()['wearing']
    return bouts_df.fillna(0).astype(int)


def get_min_bouts(df):
    """
    Get the number of seconds that fall into each activity level for each minute
    :param df: the dataframe for one participant's table of power data on the second level
    :return: a dataframe containing the number of seconds for each minute that fall in to each activity level
    """
    max_sec = 60
    thresh_low = {'sedentary': 0,
                  'light': 100 / 60,
                  'moderate': 2020 / 60,
                  'vigorous': 5999 / 60}
    thresh_high = {'sedentary': 100 / 60,
                   'light': 2020 / 60,
                   'moderate': 5999 / 60,
                   'vigorous': 10 ** 6}
    bouts_df = pd.DataFrame(columns=['utcdate']).set_index('utcdate')
    bouts_df['non-wear'] = max_sec - df.resample('min').sum()['wearing']
    for level in ['sedentary', 'light', 'moderate', 'vigorous']:
        bouts_df[level] = df[np.logical_and(df['summary_count'] < thresh_high[level],
                                            df['summary_count'] >= thresh_low[level])].resample(epoch).sum()['wearing']
    bouts_df = bouts_df.fillna(0).astype(int)
    bouts_df['dominant'] = bouts_df.idxmax(axis=1)
    return bouts_df


if __name__ == "__main__":
    city, wave = it.get_command_args("activity_bouts.py")
    if len(argv) < 4:
        print("Usage: python activity_bouts.py SITE_NAME WAVE_NUMBER DIRECTORY")
        exit()
    output_dir = argv[3]
    if not isdir(output_dir):
        print("Could not locate provided directory.")
        exit()

    input_type = '_top_1min_'
    if '-D' in argv:
        epoch = 'D'
    elif '-m' in argv:
        epoch = 'min'
        input_type = '_table_of_power_'
    else:
        epoch = 'H'

    if wave < 10:
        top_fname = output_dir + '/' + city + '_0' + str(wave) + input_type \
                + str(datetime.fromtimestamp(os.stat('.git/FETCH_HEAD').st_mtime))[0:10] + '.csv'
        out_fname = output_dir + '/' + city + '_0' + str(wave) + '_' + epoch + '_bouts_' \
                + str(datetime.fromtimestamp(os.stat('.git/FETCH_HEAD').st_mtime))[0:10] + '.csv'
    else:
        top_fname = output_dir + '/' + city + '_' + str(wave) + input_type \
                + str(datetime.fromtimestamp(os.stat('.git/FETCH_HEAD').st_mtime))[0:10] + '.csv'
        out_fname = output_dir + '/' + city + '_' + str(wave) + '_' + epoch + '_bouts_' \
                + str(datetime.fromtimestamp(os.stat('.git/FETCH_HEAD').st_mtime))[0:10] + '.csv'

    if not isfile(top_fname):
        print("Provided path for file name is not a file.")
        exit()

    top_df = pd.read_csv(top_fname)
    top_df['utcdate'] = pd.to_datetime(top_df['utcdate'])
    top_df.set_index('utcdate', inplace=True)
    header = True
    for interact_id in top_df['interact_id'].drop_duplicates().to_list():
        print("Processing participant " + str(interact_id) + ".")
        participant = top_df[top_df['interact_id'] == interact_id]
        if epoch == 'min':
            bouts = get_min_bouts(participant)
        else:
            bouts = get_bouts(participant, epoch)
        participant = participant.resample(epoch).first()[['interact_id', 'age', 'gender', 'city_id', 'wave_id']]
        for col in ['interact_id', 'age', 'gender', 'city_id', 'wave_id']:
            participant[col] = participant.iloc[0][col]
        participant[['interact_id', 'age', 'wave_id']] = participant[['interact_id', 'age', 'wave_id']].astype(int)
        participant = participant.join(bouts)
        participant = participant.reset_index().set_index(['interact_id', 'utcdate'])
        if header:
            participant.to_csv(out_fname)
            header = False
        else:
            participant.to_csv(out_fname, mode='a', header=False)
