from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from scipy.optimize import minimize
import numpy as np

class Track():
    def __init__(self, track_pts, width):
        self.track_pts = track_pts
        self.width = width

        # Creates a local list of t to discretize splines
        self.t = [x for x in range(1, len(self.track_pts) + 1)]

        # Defines parameterized splines
        self.spline_x = CubicSpline(self.t, [point[0] for point in self.track_pts], bc_type = "periodic")
        self.spline_y = CubicSpline(self.t, [point[1] for point in self.track_pts], bc_type = "periodic")

        # Integrand of arc length calculation
        self.arc_length_integrand = self.arc_length(self.spline_x, self.spline_y, self.t[-1])

        # t such that distance along spline remains constant (track width)
        self.eq_t = self.equal_spacing(width)

        # Gates at each equally spaced t
        self.gates = [Gate(eq_t, self.spline_y.derivative(1)(eq_t) / self.spline_x.derivative(1)(eq_t), self.spline_x, self.spline_y) for eq_t in self.eq_t]

        # Optimal path generation
        self.optimal = Path(self.gates, self.eq_t, width)

    # arrives at a new function f(x) = sqrt((dx/dt)**2 + (dy/dt)**2)
    def arc_length(self, x_t, y_t, max_t):
        # Defines a list of t for calculations with discrete points
        discrete_t = np.arange(1, max_t, 0.001)
        
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
            # Integral evaluated from last t val to x = distance
            t_vals.append(fsolve(lambda x: self.arc_length_integrand(x) - self.arc_length_integrand(t_vals[-1]) - distance, t_vals[-1])[0])
            np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            end_point = t_vals[-1]
        
        # Last value from fsolve could surpass self.t[-1], so we need to correct t_vals to remove points outside the intended bounds
        adjusted_t = [t for t in t_vals if t <= self.t[-1]]
        return adjusted_t

    def center_curvature_calc(self, eq_t, disc_x, disc_y):
        # let r(t) = (x(t), y(t))

        r_dot_x, r_dot_y = disc_x.derivative(1)(eq_t), disc_y.derivative(1)(eq_t)

        r_double_dot_x, r_double_dot_y = disc_x.derivative(2)(eq_t), disc_y.derivative(2)(eq_t)

        # k = ||r'(t) x r''(t)|| / ||r'(t)||**3

        cross = np.cross((r_dot_x, r_dot_y), (r_double_dot_x, r_double_dot_y))
        direction = np.sign(cross)

        return (abs(cross) / ((r_dot_x) ** 2 + (r_dot_y) ** 2) ** (3 / 2)), direction

    def center_curvature_from_dist(self, dist):
        # Calculates final t value of spline (allows the calculation of the proper arc length integrand)
        final_t = max([fsolve(lambda t: self.x(t) - self.optimal_points[-1][0], len(self.eq_t)),
                       fsolve(lambda x: self.final_spline_x(x) - self.optimal_points[-1][0], len(self.eq_t))])

        # Arc length integrand for optimized spline
        distance_integrand = self.arc_length(self.final_spline_x, self.final_spline_y, final_t)

        # Calculates t based on given distance (arc length)
        t = fsolve(lambda t: distance_integrand(t) - distance_integrand(1) - dist, 1)[0]

        curv, dir = self.center_curvature_calc(t, self.final_spline_x, self.final_spline_y)

        return curv, t, dir

class Path(Track):
    def __init__(self, gates, eq_t, width):
        self.gates = gates
        self.eq_t = eq_t
        self.width = width

        self.spline_x = CubicSpline(self.eq_t, [gate.spline_x(gate.t) for gate in self.gates])
        self.spline_y = CubicSpline(self.eq_t, [gate.spline_y(gate.t) for gate in self.gates])
        
        self.optimal_points, self.final_spline_x, self.final_spline_y = self.optimal_gate_pos(gates)


    def cost(self, params):
        points = [gate.gate_traverse(params[self.gates.index(gate)]) for gate in self.gates]

        points += [points[0]]

        adjusted_t = self.eq_t + [self.eq_t[-1] + (self.eq_t[-1] - self.eq_t[-2])]

        spline_x = CubicSpline(adjusted_t, [point[0] for point in points], bc_type = "periodic")
        spline_y = CubicSpline(adjusted_t, [point[1] for point in points], bc_type = "periodic")

        curvature = 0

        # Change 0.5 below to change accuracy of cost function (0.1 takes 4 mins to run, so might need a better method)
        for t in np.arange(0, adjusted_t[-1], 0.1):
            curvature += self.curvature_calc(t, spline_x, spline_y)[0]

        return curvature


    def optimal_gate_pos(self, gates):
        offsets = minimize(self.cost, [0 for x in range(len(gates))], bounds=[[-1 * self.width / 2, self.width / 2] for x in range(len(gates))]).x

        final_t = max([fsolve(lambda t: self.spline_x(t) - self.spline_x(self.eq_t[0]), self.eq_t[-1]), 
        fsolve(lambda x: self.spline_y(x) - self.spline_x(self.eq_t[0]), self.eq_t[-1])])[0]

        ts = self.eq_t + [final_t]

        optimal_points = []

        for gate in gates:
            optimal_points.append(gate.gate_traverse(offsets[gates.index(gate)]))

        optimal_points.append(optimal_points[0]) 

        spline_x = CubicSpline(ts, [point[0] for point in optimal_points], bc_type = "periodic")
        spline_y = CubicSpline(ts, [point[1] for point in optimal_points], bc_type = "periodic")

        return optimal_points, spline_x, spline_y


    def curvature_calc(self, t, spline_x, spline_y):
        # let r(t) = (x(t), y(t))

        r_dot_x, r_dot_y = spline_x.derivative(1)(t), spline_y.derivative(1)(t)

        r_double_dot_x, r_double_dot_y = spline_x.derivative(2)(t), spline_y.derivative(2)(t)

        # k = ||r'(t) x r''(t)|| / ||r'(t)||**3

        cross = np.cross((r_dot_x, r_dot_y), (r_double_dot_x, r_double_dot_y))
        direction = np.sign(cross)

        return (abs(cross) / ((r_dot_x)**2 + (r_dot_y)**2)**(3/2)), direction
    
    def curvature_from_dist(self, dist):
        # Calculates final t value of spline (allows the calculation of the proper arc length integrand)
        final_t = max([fsolve(lambda t: self.final_spline_x(t) - self.optimal_points[-1][0], len(self.eq_t)), 
        fsolve(lambda x: self.final_spline_x(x) - self.optimal_points[-1][0], len(self.eq_t))])
        
        # Arc length integrand for optimized spline
        distance_integrand = self.arc_length(self.final_spline_x, self.final_spline_y, final_t)

        # Calculates t based on given distance (arc length)
        t = fsolve(lambda t: distance_integrand(t) - distance_integrand(1) - dist, 1)[0]

        curv, dir = self.curvature_calc(t, self.final_spline_x, self.final_spline_y)

        return curv, t, dir


class Gate():
    def __init__(self, t, slope, spline_x, spline_y):
        self.spline_x = spline_x
        self.spline_y = spline_y
        self.t = float(t)
        self.direction = self.unit_vector(-1 / slope)
        
    def unit_vector(self, slope):
        magnitude = (1 + slope**2)**(1/2)
        x = 1 / magnitude
        y = slope / magnitude

        return [x, y]
    
    # Defines the bounds of the gate
    def bounds(self, distance):
        start_x, start_y = self.spline_x(self.t), self.spline_y(self.t)

        x_1, y_1 = start_x + self.direction[0] * distance / 2, start_y + self.direction[1] * distance / 2

        x_2, y_2 = start_x - self.direction[0] * distance / 2, start_y - self.direction[1] * distance / 2

        return [x_1, y_1], [x_2, y_2]
    
    # Returns certesian point a certain distance along gate
    def gate_traverse(self, distance):
        start_x, start_y = self.spline_x(self.t), self.spline_y(self.t)
        x_1, y_1 = start_x + self.direction[0] * distance, start_y + self.direction[1] * distance

        return [x_1, y_1]