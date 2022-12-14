from generation import Track as Track
from generation import Path as Path
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == "__main__":

    track_points, track_width = [[1, 3], [2, 6], [5, 6], [4, 2], [6, -2], [3, 0], [1, 3]], 1

    # track_points, track_width = [[1, 3], [0.5, 5], [1, 7], [3, 8], [5.5, 10], [5.5, 30], [7, 35], [8.5, 30], [8.5, 10], [11, 8], [13, 7], [13.5, 5], [13, 3], [10, 2], [7, 3], [4, 2], [1, 3]], 1

    ##################################################################################
    new_track = Track(track_points, track_width)
    eq_t = new_track.eq_t
    ts = np.arange(1, len(track_points), 0.01)

    disc_x = new_track.spline_x(ts)
    disc_y = new_track.spline_y(ts)

    x = new_track.spline_x(eq_t)
    y = new_track.spline_y(eq_t)

    outer_bound = [gate.bounds(track_width)[0] for gate in new_track.gates]
    inner_bound = [gate.bounds(track_width)[1] for gate in new_track.gates]

    final_t = max([fsolve(lambda t: new_track.optimal.final_spline_x(t) - new_track.optimal.optimal_points[-1][0], len(eq_t)), 
    fsolve(lambda x: new_track.optimal.final_spline_x(x) - new_track.optimal.optimal_points[-1][0], len(eq_t))])

    optimal_ts = np.arange(1, final_t, 0.01)
    optimal_x = new_track.optimal.final_spline_x(optimal_ts)
    optimal_y = new_track.optimal.final_spline_y(optimal_ts)

    # Saves csv of curvature as a function of distance
    # df_setup = {"dist": [], "curvature": []}

    # for i in np.arange(0, 23, 0.01):
    #     df_setup["dist"] += [i]
    #     df_setup["curvature"] += [new_track.optimal.curvature_from_dist(i)[0]]
    
    # df = pd.DataFrame(df_setup)

    # df.to_csv("urmomv2.csv")
    
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
    ##################################################################################