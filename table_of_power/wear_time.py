"""
Ported wear time detection function used at
https://cran.r-project.org/web/packages/PhysicalActivity/PhysicalActivity.pdf
"""

import pandas as pd
import numpy as np


def add_seq(start_list, stop_list, counts_bool):
    """
    Function which retrieves the total entries with counts > 0 in all of the provided intervals
    :param start_list: A list of all of the start points of the intervals
    :param stop_list: A list of all of the end points of the intervals
    :param counts_bool: Whether or not the corresponding row has a nonzero value in the counts column
    :return: The total number nonzero entries within the provided intervals
    """
    to_ret = 0
    for i in range(0, len(start_list)):
        start = start_list[i]
        stop = stop_list[i]
        if start != stop:
            to_ret = to_ret + counts_bool.iloc[range(start, stop)].sum()
    return to_ret


def marking(counts_df, frame, stream_frame=None, allowance_frame=2):
    """
    Determines the times in which the device was worn and not worn by checking for extended periods of time with no
    activity counts
    :param counts_df: A dataframe containing the counts for each time unit
    :param frame: The size of time interval to be considered
    :param stream_frame: The size of time interval that the program will look back or forward if activity is detected.
    The default is the half of frame
    :param allowance_frame: The size of time interval that zero counts allowed. The default is 2
    :return: A dataframe with an extra wearing/non-wearing marking column
    """
    if stream_frame is None:
        stream_frame = round(frame / 2)
    size = counts_df.shape[0]
    wearing = pd.DataFrame([0] * size)
    counts_bool = counts_df.summary_count > 0
    start_pos = np.where(np.diff(np.concatenate(([0, ], counts_bool.to_list()))) == 1)[0]
    end_pos = np.where(np.diff(np.concatenate((counts_bool.to_list(), [0, ]))) == -1)[0]
    index = np.where(end_pos - start_pos < allowance_frame)
    us_start = np.maximum(start_pos[index] - stream_frame, 1)
    us_end = np.maximum(start_pos[index] - 1, 1)
    ds_end = np.minimum(end_pos[index] + stream_frame, size)
    ds_start = np.minimum(end_pos[index] + 1, size)
    up_sum = add_seq(us_start, us_end, counts_bool)
    down_sum = add_seq(ds_start, ds_end, counts_bool)
    bad_index = np.where(np.logical_and(up_sum == 0, down_sum == 0))[0]
    if len(bad_index) > 0:
        start_pos = np.delete(start_pos, bad_index)
        end_pos = np.delete(end_pos, bad_index)

    gaplen = len(end_pos)
    if gaplen > 0:
        end_gap = np.delete(end_pos, gaplen - 1)
        start_gap = np.delete(start_pos, 0)
        gap = np.where(start_gap - end_gap > frame)[0]
        new_start_pos = np.concatenate(([start_pos[1], ], start_gap[gap]))
        new_end_pos = np.concatenate((end_gap[gap], [end_pos[gaplen - 1], ]))
        tog_wear = []
        for i in range(0, len(new_start_pos)):
            tog_wear = tog_wear + list(range(new_start_pos[i], new_end_pos[i]))
        i = 0
        tog_wear = tog_wear + list(range(new_start_pos[i], new_end_pos[i]))
        wearing.iloc[tog_wear] = 1
        wearing.iloc[size - 1] = wearing.iloc[size - 2]
    counts_df.reset_index(inplace=True)
    counts_df['wearing'] = wearing
    return counts_df.set_index(['utcdate'])


def wear_marking(counts_df, frame=90, epoch='1min', stream_frame=None, allowance_frame=2):
    """
    Resamples the dataframe with a specified time interval, then determines when the device was worn and not worn
    :param counts_df: A dataframe containing the counts for each time unit
    :param frame: The size of time interval to be considered
    :param epoch: The time interval the dataframe has been resampled to
    :param stream_frame: The size of time interval that the program will look back or forward if activity is detected.
    The default is the half of frame
    :param allowance_frame: The size of time interval that zero counts allowed. The default is 2
    :return: A dataframe with an extra wearing/non-wearing marking column
    """
    counts_df = counts_df.resample(epoch).sum()
    counts_df = marking(counts_df, frame, stream_frame, allowance_frame)
    return counts_df
