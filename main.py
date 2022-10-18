from generation import Track as Track
from generation import Path as Path
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def toExcel():
    # General track info for chart
    df1_setup = {"Name": ["Country", "City", "Type", "Configuration", "Direction", "Mirror"],
                 "Basic Track": ["Dynamistan", "Your mom's house", "Permanent", "Closed", "Forward", "Off"]}

    df1 = pd.DataFrame(df1_setup)

    # Saves csv of track approximated by circular arcs
    df2_setup = {"Type": [], "Section Length": [], "Corner Radius": []}

    segment_size = .1
    for i in np.arange(0, final_t, segment_size):
        k = new_track.optimal.curvature_from_dist(i)[0]
        sgn = new_track.optimal.curvature_from_dist(i)[2]
        r = 1 / k
        df2_setup["Type"].append("Left" if sgn == 1 else ("Right" if sgn == -1 else "Straight"))
        df2_setup["Section Length"].append(segment_size)
        df2_setup["Corner Radius"].append(r)

    df2 = pd.DataFrame(df2_setup)

    # Write elevations to excel
    df3_setup = {"Point [m]": [0], "Elevation [m]": [0]}

    df3 = pd.DataFrame(df3_setup)

    # Write banking to excel
    df4_setup = {"Point [m]": [0], "Banking [deg]": [0]}

    df4 = pd.DataFrame(df4_setup)

    # Write grip factors to excel
    df5_setup = {"Start Point [m]": [0], "Grip Factor [-]": [1]}

    df5 = pd.DataFrame(df5_setup)

    # Write sectors to excel
    df6_setup = {"Start Point [m]": [0], "Sector": [1]}

    df6 = pd.DataFrame(df6_setup)

    writer = pd.ExcelWriter('basic_track.xlsx', engine='xlsxwriter')

    df1.to_excel(writer, sheet_name='Info', header=True, startrow=0, startcol=0, index=False)
    df2.to_excel(writer, sheet_name='Shape', header=True, startrow=0, startcol=0, index=False)
    df3.to_excel(writer, sheet_name='Elevation', header=True, startrow=0, startcol=0, index=False)
    df4.to_excel(writer, sheet_name='Banking', header=True, startrow=0, startcol=0, index=False)
    df5.to_excel(writer, sheet_name='Grip Factors', header=True, startrow=0, startcol=0, index=False)
    df6.to_excel(writer, sheet_name='Sectors', header=True, startrow=0, startcol=0, index=False)
    writer.save()



if __name__ == "__main__":

    track_points, track_width = [[1, 3], [2, 6], [5, 6], [4, 2], [6, -2], [3, 0], [1, 3]], 1

    # track_points, track_width = [[1, 3], [0.5, 5], [1, 7], [3, 8], [5.5, 10], [5.5, 30], [7, 35], [8.5, 30], [8.5, 10], [11, 8], [13, 7], [13.5, 5], [13, 3], [10, 2], [7, 3], [4, 2], [1, 3]], 1

    ##################################################################################
    new_track = Track(track_points, track_width)
    eq_t = new_track.eq_t
    ts = np.arange(1, len(track_points), 0.01) # originally 0.01

    disc_x = new_track.spline_x(ts)
    disc_y = new_track.spline_y(ts)

    x = new_track.spline_x(eq_t)
    y = new_track.spline_y(eq_t)

    outer_bound = [gate.bounds(track_width)[0] for gate in new_track.gates]
    inner_bound = [gate.bounds(track_width)[1] for gate in new_track.gates]

    final_t = max([fsolve(lambda t: new_track.optimal.final_spline_x(t) - new_track.optimal.optimal_points[-1][0], len(eq_t)), 
    fsolve(lambda x: new_track.optimal.final_spline_x(x) - new_track.optimal.optimal_points[-1][0], len(eq_t))])

    print(final_t)

    optimal_ts = np.arange(1, final_t, 0.1)
    optimal_x = new_track.optimal.final_spline_x(optimal_ts)
    optimal_y = new_track.optimal.final_spline_y(optimal_ts)

    # Plots spline representation of track midline
    plt.plot(disc_x, disc_y)
    plt.plot(x, y, 'ro')
    
    # Outer and inner bounds
    plt.plot([point[0] for point in outer_bound], [point[1] for point in outer_bound], 'ro')
    plt.plot([point[0] for point in inner_bound], [point[1] for point in inner_bound], 'ro')

    # Optimal points
    plt.plot(optimal_x, optimal_y)

    # Optional formatting
    plt.gca().set_aspect('equal')
    # plt.axis([-1, 15, 0, 40])


    plt.show()

    toExcel()

    ##################################################################################