"""
Author: Kole Phillips
"""

import pandas as pd
from datetime import timedelta


def filter_days(raw_df):
    """
    Determines whether a day has records in at least 600 minutes, and removes it if it does not
    :param raw_df: A single participant's data
    :return: The filtered data
    """
    dates = raw_df.utcdate.dt.round('D').unique()
    kept_days = []
    for day in dates:
        day_data = raw_df[raw_df.utcdate.dt.round('D') == day]
        minutes = day_data.utcdate.dt.round('min').unique()
        if len(minutes) > 600:
            kept_days.append(day)
    return kept_days


def filter_data(p_data):
    """
    Determines whether a participant has sufficient data, ie more than 3 days that satisfy the conditions of
    filter_days()
    :param p_data: A single participant's data
    :return: The participant's filtered data if their data was sufficient, or an empty dataframe otherwise
    """
    kept_data = pd.DataFrame()
    kept_days = filter_days(p_data)
    if len(kept_days) > 3:
        kept_data = kept_data.append(p_data[p_data.utcdate.dt.round('D').isin(kept_days)], ignore_index=True)
    return kept_data


def skip_data(data_df, mins_skip=1):
    """
    Deletes a set number of minutes for every one minute kept, then duplicates each kept minute for every minute that
    was deleted in its cycle
    :param data_df: The participant's dataframe
    :param mins_skip: The number of minutes to skip each cycle
    :return:
    """
    return_data = pd.DataFrame()
    p_data = data_df
    all_minutes = p_data.utcdate.dt.round('min').unique()
    minutes_used = all_minutes[list(range(0, len(all_minutes), 1 + mins_skip))]
    kept_data = p_data[p_data.utcdate.dt.round('min').isin(minutes_used)]
    return_data = return_data.append(kept_data, ignore_index=True)
    for i in range(mins_skip):
        copied_data = kept_data.copy(deep=True)
        copied_data['utcdate'] = copied_data.utcdate + timedelta(minutes=1 + i)
        return_data = return_data.append(copied_data, ignore_index=True)
    return return_data.drop_duplicates('utcdate')
