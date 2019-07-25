"""
Author: Antoniu Vadan
Description: contains function which, given any two points in a grid (with timestamp), the
    grid's resolution, an upper bound on speed, and stationary activity time, returns
    points inside the ellipse as described by Miller, 2005, A Measurement Theory for Time Geography
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def points_in_ellipse(p1, p2, res, vmax, stat_act):
    """
    Description:
        Given two points, function returns grid points that lie inside an ellipse as described
            by Miller, 2005, A Measurement Theory for Time Geography
    :param p1: point 1 -- northing, easting, pandas_time_object
    :param p2: point 2 -- northing, easting, pandas_time_object
    :param res: resolution of city grid
    :param vmax: maximum speed of agent
    :param stat_act: stationary activity time
    :return: grid points that lie in the ellipse
    ********NOT FILTERING OUT POINTS THAT ARE NOT IN CITY POLYGON********
    """
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    y_0 = (p1[0] + p2[0])/2  # ellipse center northing
    x_0 = (p1[1] + p2[1])/2  # ellipse center easting
    g_ell_center = (y_0, x_0)

    dy = abs(p2[0] - p1[0])  # difference in row numbers -- y axis
    dx = abs(p2[1] - p1[1])  # difference in column numbers -- x axis
    dist_squared = dy**2 + dx**2  # distance is squared because distance alone would require
                                  #     taking the square root of this equation

    t2_s = (p2[2].datetime(1970,1,1)).total_seconds()
    t1_s = (p1[2].datetime(1970,1,1)).total_seconds()

    g_ell_width = (t2_s - t1_s - stat_act) * vmax  # TODO: VERIFY THAT TIME DIFFERENCE IS IN SECONDS
    g_ell_height = ((((t2_s - t1_s - stat_act) * vmax) ** 2) - dist_squared) ** 0.5

    # create bounding box for the ellipse for increased performance
    y_upper = y_0 + g_ell_width/2
    y_lower = y_0 - g_ell_width/2
    x_upper = x_0 + g_ell_width/2
    x_lower = x_0 - g_ell_width/2

    y, x = np.mgrid[np.arange(y_lower, y_upper, res),
                    np.arange(x_lower, x_upper, res)]
    y, x = y.ravel(), x.ravel()

    angle = np.arctan(dy/dx)
    angle = np.rad2deg(angle)

    g_ellipse = patches.Ellipse(g_ell_center, g_ell_width, g_ell_height, angle=angle)
    ax.add_patch(g_ellipse)

    cos_angle = np.cos(np.radians(180. - angle))
    sin_angle = np.sin(np.radians(180. - angle))

    xc = x - g_ell_center[0]
    yc = y - g_ell_center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle
    rad_cc = (xct ** 2 / (g_ell_width / 2.) ** 2) + (yct ** 2 / (g_ell_height / 2.) ** 2)

    in_ellipse = []

    for i in range(len(rad_cc)):
        if rad_cc[i] <= 1.:
            # point in ellipse
            in_ellipse.append((y[i], x[i]))

    return in_ellipse


#######################################################################
#################### D E M O N S T R A T I O N ########################
####################### I N   E L L I P S E ###########################
#######################################################################
# # Some test points
# y, x = np.mgrid[30:1080:50, 50:1100:50]
# y, x = y.ravel(), x.ravel()
#
# fig, ax = plt.subplots(1)
# ax.set_aspect('equal')
#
# # The ellipse
# g_ell_center = (500, 500)
# g_ell_width = 400  # major axis
# g_ell_height = 200  # minor axis
# angle = 30.
#
# g_ellipse = patches.Ellipse(g_ell_center, g_ell_width, g_ell_height, angle=angle,
#                             fill=False, edgecolor='green', linewidth=2)  # for display
# ax.add_patch(g_ellipse)
#
# cos_angle = np.cos(np.radians(180.-angle))
# sin_angle = np.sin(np.radians(180.-angle))
#
# xc = x - g_ell_center[0]
# yc = y - g_ell_center[1]
#
# xct = xc * cos_angle - yc * sin_angle
# yct = xc * sin_angle + yc * cos_angle
# rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)
#
# colors_array = []
# in_ellipse = []
#
# for i in range(len(rad_cc)):
#     if rad_cc[i] <= 1.:
#         # point in ellipse
#         colors_array.append('green')
#         in_ellipse.append((y[i], x[i]))
#     else:
#         # point not in ellipse
#         colors_array.append('black')
#
# ax.scatter(x,y,c=colors_array,linewidths=0.1, s=5)
# print(in_ellipse)
# plt.show()
