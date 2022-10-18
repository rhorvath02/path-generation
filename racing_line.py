import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

class Racing_Line:
    def __init__(self, cone_pts_csv, track_type, track_name):
        self.type = track_type
        self.name = track_name

        self.cone_pts = pd.read_excel(io=("../tracks/%s" % cone_pts_csv))  # in feet, assumes following form: out_x | out_y | in_x | in_y, pd will auto add gate num

        for x in ["out_x", "out_y", "in_x", "in_y"]:
            self.cone_pts[x] *= .3048  # convert cone coords into meters

        tw = 1.2192  # car.front_trackwidth in meters
        self.bound_pts = pd.DataFrame(columns=['out_x_bound', 'out_y_bound', 'in_x_bound', 'in_y_bound'])
        self.center_pts = pd.DataFrame(columns=['center_x', 'center_y'])

        for i in range(len(self.cone_pts.index)):
            out_vec = np.array([self.cone_pts["out_x"][i], self.cone_pts["out_y"][i]])  # vectorize gate points
            in_vec = np.array([self.cone_pts["in_x"][i], self.cone_pts["in_y"][i]])
            gate_vec = in_vec - out_vec  # find relative displacement of gate cones
            gate_width = np.linalg.norm(gate_vec)
            out_bound_vec = out_vec + tw/(2*gate_width)*gate_vec  # set boundaries for car accounting for trackwidth
            in_bound_vec = in_vec - tw/(2*gate_width)*gate_vec
            center_vec = out_vec+.5*gate_vec  # find central point between gates
            self.bound_pts.loc[len(self.bound_pts.index)] = [out_bound_vec[0], out_bound_vec[1], in_bound_vec[0], in_bound_vec[1]]  # append boundary points to dataframe
            self.center_pts.loc[len(self.center_pts.index)] = [center_vec[0], center_vec[1]]  # append center points to dataframe

        # Creates a local list of n interpolating splines
        self.n = [x for x in range(1, len(self.center_pts.index)+1)]
        # print(self.n)

        # Defines parameterized splines
        self.spline_x = CubicSpline(self.n, [point[0] for point in self.center_pts.to_numpy()], bc_type="natural")
        self.spline_y = CubicSpline(self.n, [point[1] for point in self.center_pts.to_numpy()], bc_type="natural")

        self.optimize()

    def curvature_calc(self, t, x, y):
        # let r(t) = (x(t), y(t))

        r_dot_x, r_dot_y = x.derivative(1)(t), y.derivative(1)(t)

        r_double_dot_x, r_double_dot_y = x.derivative(2)(t), y.derivative(2)(t)

        # k = ||r'(t) x r''(t)|| / ||r'(t)||**3

        cross = np.cross((r_dot_x, r_dot_y), (r_double_dot_x, r_double_dot_y))

        return abs(cross) / ((r_dot_x) ** 2 + (r_dot_y) ** 2) ** (3 / 2)

    def cost(self, gate_fracs):
        self.update_path(gate_fracs)

        dt = .01  # interval to sample curvature
        curv_sum = 0  # total curvature weighted by arc length
        ts = np.arange(1, len(self.center_pts.to_numpy()), dt)
        for t in ts:
            ds = np.sqrt(self.spline_x.derivative(1)(t)**2+self.spline_y.derivative(1)(t)**2)*dt  # arc length differential from dt
            curv_sum += self.curvature_calc(t,self.spline_x,self.spline_y)*ds
        print("eval")
        return curv_sum


    def optimize(self):
        offsets = scipy.optimize.minimize(self.cost, [.5 for x in range(len(self.cone_pts.index))], bounds=[[0, 1] for x in range(len(self.cone_pts.index))], options={"maxiter":5}).x
        self.write_path()
        print("done")

    def update_path(self, gate_fracs):
        self.inter_pts = pd.DataFrame(columns=['inter_x', 'inter_y'])
        for i in range(len(self.cone_pts.index)):
            x_inter = (gate_fracs[i]) * (self.bound_pts.loc[i][0]) + (1 - gate_fracs[i]) * (self.bound_pts.loc[i][2])
            y_inter = (gate_fracs[i]) * (self.bound_pts.loc[i][1]) + (1 - gate_fracs[i]) * (self.bound_pts.loc[i][3])
            self.inter_pts.loc[i] = [x_inter, y_inter]

        if self.type != "ax":
            self.inter_pts.loc[0] = self.inter_pts.loc[len(self.cone_pts.index)-1]

        self.spline_x = CubicSpline(self.n, [point[0] for point in self.inter_pts.to_numpy()], bc_type="natural")# bc_type=("natural" if self.type=="ax" else "periodic"))
        self.spline_y = CubicSpline(self.n, [point[1] for point in self.inter_pts.to_numpy()], bc_type="natural")# bc_type=("natural" if self.type=="ax" else "periodic"))

    def write_path(self):
        inc = .01
        s = np.arange(1, len(self.n), inc)
        x = self.spline_x(s)
        y = self.spline_y(s)
        z = [0]*len(s)

        df = pd.DataFrame({"x": x, "y": y, "z": z})
        df.to_csv(path_or_buf=("../racing_lines/%s.csv" % self.name), index=False)




# r = Racing_Line("Autocross_Michigan_2019_Sanitized.xlsx", "ax", "ax_mi_2019")
r = Racing_Line("Endurance_Michigan_2019_Sanitized.xlsx", "en", "en_mi_2019")

print("init done")

ts = np.arange(1, len(r.center_pts.index), 0.01)
disc_x = r.spline_x(ts)
disc_y = r.spline_y(ts)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(disc_x,disc_y, c='orange')

ax1.scatter(r.cone_pts["out_x"].to_numpy(), r.cone_pts["out_y"].to_numpy(), c='b')
ax1.scatter(r.cone_pts["in_x"].to_numpy(), r.cone_pts["in_y"].to_numpy(), c='b')
ax1.scatter(r.bound_pts["out_x_bound"].to_numpy(), r.bound_pts["out_y_bound"].to_numpy(), c='r')
ax1.scatter(r.bound_pts["in_x_bound"].to_numpy(), r.bound_pts["in_y_bound"].to_numpy(), c='r')
ax1.scatter(r.center_pts["center_x"].to_numpy(), r.center_pts["center_y"].to_numpy(), c='g')

plt.show()