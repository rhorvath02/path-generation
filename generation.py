from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.optimize import minimize
import numpy as np

class Track():
    def __init__(self, track_pts, width):
        self.track_pts = track_pts

        # Creates a local list of t to discretize splines
        self.t = [x for x in range(1, len(self.track_pts) + 1)]

        # Defines parameterized splines
        self.spline_x = CubicSpline(self.t, [point[0] for point in self.track_pts])
        self.spline_y = CubicSpline(self.t, [point[1] for point in self.track_pts])

        # Integrand of arc length calculation
        self.arc_length_integrand = self.arc_length(self.spline_x, self.spline_y)

        # t such that distance along spline remains constant (track width)
        self.eq_t = self.equal_spacing(width)

        # Gates at each equally spaced t
        self.gates = [Gate(width, eq_t, self.spline_y.derivative(1)(eq_t) / self.spline_x.derivative(1)(eq_t), self.spline_x, self.spline_y) for eq_t in self.eq_t]

        self.optimal = Path(self.gates, self.track_pts)

    # arrives at a new function f(x) = sqrt((dx/dt)**2 + (dy/dt)**2)
    def arc_length(self, x_t, y_t):
        # Defines a list of t for calculations with discrete points
        discrete_t = np.arange(1, self.t[-1], 0.001)
        
        # Takes derivative of parameterized input splines
        dxdt = x_t.derivative(1)
        dydt = y_t.derivative(1)

        # Discretizes given splines
        discretized_dxdt = dxdt(discrete_t)
        discretized_dydt = dydt(discrete_t)

        # Performs the square operations in the arc length formula
        discretized_dxdt_squared = [dxdt**2 for dxdt in discretized_dxdt]
        discretized_dydt_squared = [dydt**2 for dydt in discretized_dydt]

        # Performs the sum in the arc length formula
        discretized_summation = [dxdt + dydt for (dxdt, dydt) in zip(discretized_dxdt_squared, discretized_dydt_squared)]

        # Performs the sqrt operation in the arc length formula
        discretized_integrand = [x**(1/2) for x in discretized_summation]

        return CubicSpline(discrete_t, discretized_integrand).antiderivative(1)

    # Finds t such that distance along spline remains constant (track width)
    def equal_spacing(self, distance):
        t_vals = [1]
        end_point = 1

        while end_point < self.t[-1]:
            t_vals.append(fsolve(lambda x: self.arc_length_integrand(x) - self.arc_length_integrand(t_vals[-1]) - distance, t_vals[-1]))
            np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            end_point = t_vals[-1]
        
        adjusted_t = [t for t in t_vals if t <= len(self.track_pts)]

        return adjusted_t

class Path(Track):
    def __init__(self, gates, init_points):
        self.width = gates[0].width
        self.init_points = init_points
        self.t = [t for t in range(1, len(init_points) + 1)]
        self.spline_x = CubicSpline(self.t, [point[0] for point in self.init_points])
        self.spline_y = CubicSpline(self.t, [point[1] for point in self.init_points])
        
        self.gates = gates
        
        self.optimal_path, self.final_spline_x, self.final_spline_y = self.optimal_gate_pos(gates)


    def cost(self, params):
        disc_t = [x for x in range(1, len(self.gates) + 1)]

        points = [gate.gate_traverse(params[self.gates.index(gate)]) for gate in self.gates]

        spline_x = CubicSpline(disc_t, [point[0] for point in points])
        spline_y = CubicSpline(disc_t, [point[1] for point in points])

        curvature = 0
        # Change 0.5 below to change accuracy of cost function (0.1 takes 4 mins to run, so might need a better method)
        for t in np.arange(1, disc_t[-1] + 1, 0.1):
            curvature += self.curvature_calc(t, spline_x, spline_y)

        return curvature


    def optimal_gate_pos(self, gates):
        offsets = minimize(self.cost, [0 for x in range(len(gates))], bounds=[[-1 * self.width / 2, self.width / 2]
        for x in range(len(gates))]).x

        optimal_points = []

        for gate in gates:
            # print(offsets)
            optimal_points.append(gate.gate_traverse(offsets[gates.index(gate)]))

        optimal_points.append(optimal_points[0])

        # print(optimal_points)

        spline_x = CubicSpline([x for x in range(1, len(optimal_points) + 1)], [point[0] for point in optimal_points])
        spline_y = CubicSpline([x for x in range(1, len(optimal_points) + 1)], [point[1] for point in optimal_points])

        return optimal_points, spline_x, spline_y


    def curvature_calc(self, t, spline_x, spline_y):
        # let r(t) = (x(t), y(t))

        r_dot_x, r_dot_y = spline_x.derivative(1)(t), spline_y.derivative(1)(t)

        r_double_dot_x, r_double_dot_y = spline_x.derivative(2)(t), spline_y.derivative(2)(t)

        # k = ||r'(t) x r''(t)|| / ||r'(t)||**3

        cross = abs(np.cross((r_dot_x, r_dot_y), (r_double_dot_x, r_double_dot_y)))

        return (cross / ((r_dot_x)**2 + (r_dot_y)**2)**(3/2))
    
    def curvature_from_dist(self, dist):
        t = fsolve(lambda x: self.arc_length(self.final_spline_x, self.final_spline_y)(x) - dist, 1)[0]

        return self.curvature_calc(t, self.final_spline_x, self.final_spline_y), t


class Gate():
    def __init__(self, width, t, slope, spline_x, spline_y):
        self.width = width
        self.spline_x = spline_x
        self.spline_y = spline_y
        self.t = float(t)
        self.direction = self.unit_vector(-1 / slope)
        self.endpoints = self.bounds(self.width)
        self.position = 0
        
    def unit_vector(self, slope):
        magnitude = (1 + slope**2)**(1/2)
        x = 1 / magnitude
        y = slope / magnitude

        return (x, y)
    
    # This method defines the bounds of the track
    def bounds(self, distance):
        start_x, start_y = self.spline_x(self.t), self.spline_y(self.t)

        x_1, y_1 = start_x + self.direction[0] * distance / 2, start_y + self.direction[1] * distance / 2

        x_2, y_2 = start_x - self.direction[0] * distance / 2, start_y - self.direction[1] * distance / 2

        return (x_1, y_1), (x_2, y_2)
    
    def gate_traverse(self, distance):
        start_x, start_y = self.spline_x(self.t), self.spline_y(self.t)
        x_1, y_1 = start_x + self.direction[0] * distance, start_y + self.direction[1] * distance

        return (x_1, y_1)