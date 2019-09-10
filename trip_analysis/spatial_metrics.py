import numpy as np
import matplotlib.pyplot as plt
import os
from collections import OrderedDict


def plot_dist(data, xName, graphName, graphSavePath):
    '''
    Plot the distribution in a log-log scale
    :param data: x and y data in a dataframe
    :param xName: name of the x-axis
    :param graphName: title of the graph
    :param graphSavePath: string of where the graph should be saved
    :return: none, writes a file
    '''
    #Makes sure folder to save graphs exists. If not, create it
    if not os.path.exists(graphSavePath):
        os.makedirs(graphSavePath)

    data.cumsum()
    plt.figure()
    data.plot(x=xName, y='count')
    plt.title(graphName)
    plt.yscale('log', basey=2)
    plt.xscale('log', basex=2)
    plt.xlim(0, 2**12)
    plt.ylim(0, 2**12)
    plt.savefig(graphSavePath + graphName, bbox_inches='tight', format='eps', dpi=1000)
    plt.close()


def get_grid(x, y, grid_size):
    '''
    Stratifies the location points into a grid of cell size m
    :param x: column with the east-west location values
    :param y: column with the south-north location values
    :param size: integer representing the cell size in meters
    :return: Series with the grid values
    '''

    start_x = min(x)
    start_y = min(y)

    # Find the grids that each position point belongs to
    cell_x = np.ceil((x - start_x) / grid_size).astype(int).astype(str)
    cell_y = np.ceil((y - start_y) / grid_size).astype(int).astype(str)

    return cell_x + cell_y


def visit_frequency(df):
    '''
    Operationalize visit frequency. "This metric indicates overall place popularity." (Tuhin, 2017)
    :param df: data frame with the location cells
    :return: data frame with "the distribution of the count of participant samples in a given location."
    (Tuhin, 2017)
    '''
    return df.groupby(['cell']).size().reset_index(name='frequency')\
            .groupby('frequency').size().reset_index(name='count')


def movement_detection(df, n):
    '''
    Operationalize trips and dwells. Movement is defined as changing locations for at least three consecutive dwells
    time steps.

    :param df: data frame with the location cells
    :param n: integer for N-times duty cycle
    :return: dataframe with the length and duration of the trips
    '''

    def rename_columns():
        df.columns = df.columns.droplevel()
        df.columns.values[0] = 'inter_id'
        df.columns.values[1] = 'start_time'
        df.columns.values[2] = 'end_time'
        df.columns.values[3] = 'grid_cell'
        df.columns.values[4] = 'duration'

    df = df.sort_values(["utc_date", "inter_id"])

    df['duration'] = ((df['grid_cell'] != df['grid_cell'].shift()) | (df['inter_id'] != df['inter_id'].shift())).cumsum()
    df = df.groupby(df['duration'], as_index=False).agg(OrderedDict([('inter_id', 'first'), ('utc_date', ['first', 'last']),
                                                                     ('grid_cell', 'first'), ('duration', 'count')]))
    rename_columns()

    df['trip_length'] = df['duration']
    df['trip_length'] = ((df['duration'] > n) | (df['duration'].shift() > n)).cumsum()
    df = df.groupby(df['trip_length'], as_index=False).agg(OrderedDict([('inter_id', 'first'), ('start_time', 'first'),
                                                                        ('end_time', 'last'), ('grid_cell', 'first'),
                                                                        ('duration', 'sum'), ('trip_length', 'count')]))

    return df
