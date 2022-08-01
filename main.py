from generation import Track as Track
from generation import Path as Path
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    track_points, track_width = [[1, 3], [2, 6], [5, 6], [4, 2], [6, -2], [3, 0], [1, 3]], 1

    # holy shit this is messy asf 

    ##################################################################################
    new_track = Track(track_points, track_width)
    eq_t = new_track.eq_t
    ts = np.arange(1, len(track_points), 0.01)

    disc_x = new_track.spline_x(ts)
    disc_y = new_track.spline_y(ts)

    x = new_track.spline_x(eq_t)
    y = new_track.spline_y(eq_t)

    outer_bound = [gate.endpoints[0] for gate in new_track.gates]
    inner_bound = [gate.endpoints[1] for gate in new_track.gates]

    final_t_lst = [fsolve(lambda t: new_track.optimal.final_spline_x(t) - new_track.optimal.optimal_path[-1][0], len(eq_t)), 
    fsolve(lambda x: new_track.optimal.final_spline_x(x) - new_track.optimal.optimal_path[-1][0], len(eq_t))]

    optimal_ts = np.arange(1, max(final_t_lst), 0.01)
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
    plt.axis([-2, 7, -2, 7])

    plt.show()
    ##################################################################################