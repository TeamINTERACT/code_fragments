"""
Source code
https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2

Adapted for travel on histogram by Antoniu Vadan in the summer of 2019.
"""
import math


class Node(object):
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent      # tuple (northing, easting)
        self.position = position  # tuple (northing, easting)

        self.g = 0  # sum of (1-p) values of path up to node
        self.h = 0  # heuristic = dist(current_node, end) / 2 * dist(start, end)
        self.f = 0  # total cost of node

    def __eq__(self, other):
        return self.position == other.position


def astar(df_histogram, start, end):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param df_histogram: dataframe containing frequency of visits
    :param start: tuple (northing, easting) where trip begins
    :param end: tuple (northing, easting) where trip ends
    :return:
    """

    ###################           ###################
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(start)
    print(end)
    #################################################


    # establish resolution
    columns_list = list(df_histogram.columns.values)  # eastings
    index_list = list(df_histogram.index.values)      # northings
    res = index_list[1] - index_list[0]

    # Create start and end node
    start_node = Node(None, start)
    end_node = Node(None, end)

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    def found(node):
        path = []
        current = node
        while current is not None:
            path.append(current.position)
            current = current.parent
        # return path[::-1]  # Return reversed path
        ### CODE FOR INTERPOLATING LOCATION ALONG LINESTRING/MULTILINESTRING ###
        path = path[::-1]
        return path


    # Loop until you find the end
    while len(open_list) > 0:
        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return found(current_node)

        # Generate children
        children = []
        for new_position in [(0, -1*res), (0, 1*res), (-1*res, 0), (1*res, 0), (-1*res, -1*res),
                             (-1*res, 1*res), (1*res, -1*res), (1*res, 1*res)]: # Adjacent squares
            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if (node_position[0] > index_list[-2]) or \
                (node_position[0] < index_list[0]) or \
                (node_position[1] > columns_list[-2]) or \
                (node_position[1] < columns_list[0]):
                continue

            # NOTE: if filtering for certain entries in df_histogram, filter HERE
            # if df_histogram.at[node_position[0], node_position[1]] == 1:
            #     continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            if child in closed_list:
                continue

            # Create the f, g, and h values
            child.g = df_histogram.at[child.position[0], child.position[1]] + \
                      child.parent.g

            # TODO: LOOK INTO THE MEANING OF DIAGONAL_DISTANCE BELOW
            # use diagonal distance for heuristic
            dx = abs(child.position[1] - end_node.position[1]) / res  # actual grid steps (1, 2, 3 ...)
            dy = abs(child.position[0] - end_node.position[0]) / res
            minimum = min(dx, dy)
            maximum = max(dx, dy)
            diagonal_steps = minimum
            straight_steps = maximum - minimum
            diagonal_distance = math.sqrt(2) * diagonal_steps + straight_steps
            child.h = diagonal_distance

            child.f = child.g + child.h
            # print('child.g:', child.g, '      child.h:', child.h, '      child.f:', child.f)

            # Child is already in the open list
            to_append = True
            tracker = None
            for i, open_node in enumerate(open_list):
                if child == open_node and child.g >= open_node.g:
                    to_append = False
                elif child == open_node and child.g < open_node.g:
                    tracker = i

            ########## trying something ###########
            # if the child is in the open list already and the cost found now is lower than the cost found
            #   in open_list, add this node to the open list and remove the old node
            if tracker is not None:
                del open_list[tracker]
            #######################################

            # Add the child to the open list
            # open_list.append(child)
            ##################################
            if to_append:
                open_list.append(child)
            ##################################


