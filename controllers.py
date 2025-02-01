import numpy as np
from cvxpy import *
import time as Time
import constants
from operator import itemgetter


class TrajectoryController(object):
    def __init__(self, traj):
        self.traj = traj
        self.u = np.array([0.0, 0.0])
        self.total_time = self.traj[-1][0]

    def __call__(self, time, robot, dt):
        setpoint = self.get_setpoint(time)
        next_setpoint = self.get_setpoint(time + dt)
        d_setpoint = (next_setpoint - setpoint)

        self.u[0] = np.linalg.norm(d_setpoint[1:3]) / dt
        self.u[1] = d_setpoint[3] / dt

        # print self.u

        return time > self.total_time, self.u

    def get_setpoint(self, time):
        if time >= self.total_time:
            return self.traj[-1]
        return self.traj[int((time / self.total_time) * len(self.traj))]

    def __str__(self):
        return 'trajectory controller\n u random error = {}'.format(constants.u_random_error)


class MPCTracking(TrajectoryController):
    def __init__(self, traj, horizon):
        super(MPCTracking, self).__init__(traj)
        self.horizon = horizon

    def __call__(self, time, robot, dt):
        # x = np.array([0.0, 0.0, 0.0]).T
        u = np.matrix([0.0, 0.0]).T
        x = robot.x

        # x = Variable((x.shape[0], self.horizon + 1))
        u = Variable((u.shape[0], self.horizon))

        constr = []
        cost = 0

        # constr += [x[:, 0] == x_0]

        for i in range(self.horizon):
            setpoint = self.get_setpoint(time)
            next_setpoint = self.get_setpoint(time + dt)
            d_setpoint = (next_setpoint - setpoint)

            vr = np.linalg.norm(d_setpoint[1:3]) / dt
            wr = d_setpoint[3] / dt

            A = np.array([[1, 0, -dt * np.sin(setpoint[3]) * vr],
                          [0, 1, dt * np.cos(setpoint[3]) * vr],
                          [0, 0, 1]])
            B = np.array([[dt * np.cos(setpoint[3]), 0],
                          [dt * np.sin(setpoint[3]), 0],
                          [0, dt]])

            # constr += [x[:, i + 1] == A * x[:, i] + B * u[:, i]]

            constr += [u[0, i] <= constants.max_v]
            constr += [u[1, i] <= constants.max_w]

            x_err = x - np.array(setpoint[1:])

            u_reference = np.array([vr, wr])

            u_err = u[:, i] - u_reference

            Q = constants.Q
            R = constants.R
            # print A.shape
            # print Q.dot(x_err).shape
            cost += sum_squares(Q.dot(A.dot(x_err)) + B * u_err * R)

        prob = Problem(Minimize(cost), constr)
        start = Time.time()

        result = prob.solve()
        # print result

        elapsed_time = Time.time() - start
        print ("calc time:{0}".format(elapsed_time) + "[sec]")
        # print ('prob.value = {}, u = '.format(prob.value, u.shape))

        # for variable in prob.variables():
        #     print "Variable %s: value %s" % (variable.name(), variable.value)
        # print prob.variables()[0].value

        return time > self.total_time, u[:, 0].value.flatten()

    def __str__(self):
        return "Q=\n {},\n R=\n {},\n Horizon= {},\n u random error= {}".format(constants.Q,
                                                                                constants.R,
                                                                                constants.horizon,
                                                                                constants.u_random_error)


class LyapunovController(TrajectoryController):
    def __init__(self, traj, horizon):
        super(LyapunovController, self).__init__(traj)
        self.horizon = horizon

    def __call__(self, time, robot, dt):
        goal_point = self.get_setpoint(time + self.horizon)
        l = np.sqrt((np.power(goal_point[1] - robot.x[0], 2)) + np.power(goal_point[2] - robot.x[1], 2))
        v_ = goal_point[3] - robot.x[2]
        v = constants.kl * l * np.cos(v_)
        w = constants.kl * np.sin(v_) * np.cos(v_) + constants.kv * np.tanh(constants.kv * v_)
        u = np.array([v, w])

        return time > self.total_time + 2, u

    def __str__(self):
        return "lyapunov controller\n kl={},\n kv={}, horizon={},\n u random error={}".format(
            constants.kl, constants.kv,
            self.horizon,
            constants.u_random_error)


class NonLinearController(TrajectoryController):
    def __init__(self, traj, horizon):
        super(NonLinearController, self).__init__(traj)
        self.horizon = horizon

    def __call__(self, time, robot, dt):
        setpoint = self.get_setpoint(time + self.horizon)
        next_setpoint = self.get_setpoint(time + self.horizon + dt)
        d_setpoint = (next_setpoint - setpoint)
        vd = self.u[0] = np.linalg.norm(d_setpoint[1:3]) / dt
        wd = self.u[1] = d_setpoint[3] / dt

        setpoint = setpoint[1:]

        state_error = setpoint - robot.x

        v = vd * np.cos(state_error[2]) + self.k1(vd, wd) * (
                (np.cos(robot.x[2]) * state_error[0]) + (np.sin(robot.x[2]) * state_error[1]))
        w = wd + constants.k2 * vd * (np.sin(state_error[2]) / state_error[2]) * (
                (np.cos(robot.x[2]) * state_error[0]) - (np.sin(robot.x[2]) * state_error[1])) + self.k3(vd, wd) * \
            state_error[2]

        u = np.array([v, w])
        return time > self.total_time, u

    @staticmethod
    def k1(v, w):
        return 2 * constants.C * (np.sqrt(np.power(w, 2) + constants.b * np.power(v, 2)))

    @staticmethod
    def k3(v, w):
        return 2 * constants.C * (np.sqrt(np.power(w, 2) + constants.b * np.power(v, 2)))

    def __str__(self):
        return "nonlinear controller\n b={},\n C={}, horizon={},\n u random error={}".format(
            constants.b, constants.C,
            self.horizon,
            constants.u_random_error)


class WangLiController(TrajectoryController):
    def __init__(self, traj, horizon):
        super(WangLiController, self).__init__(traj)
        self.horizon = horizon

    def __call__(self, time, robot, dt):
        setpoint = self.get_setpoint(time + self.horizon)
        next_setpoint = self.get_setpoint(time + self.horizon + dt)
        d_setpoint = (next_setpoint - setpoint)
        vd = self.u[0] = np.linalg.norm(d_setpoint[1:3]) / dt
        wd = self.u[1] = d_setpoint[3] / dt
        error_state = setpoint[1:] - robot.x
        rotation_mat = np.array([[np.cos(robot.x[2]), np.sin(robot.x[2]), 0],
                                 [-np.sin(robot.x[2]), np.cos(robot.x[2]), 0],
                                 [0, 0, 1]])
        error_state = rotation_mat.dot(error_state)
        v = vd * np.cos(error_state[2]) + constants.p1 * 2 / np.pi * np.arctan(error_state[0])
        w = wd + (constants.p2 * vd * error_state[1]) / (
                1 + np.power(error_state[0], 2) + np.power(error_state[1], 2)) * np.sin(error_state[2]) / error_state[
                2] + constants.p3 * 2 / np.pi * np.arctan(error_state[2])

        u = np.array([v, w])
        return time > self.total_time, u

    def __str__(self):
        return "wang li controller"


class PurePursuit(TrajectoryController):
    def __init__(self, traj, horizon):
        super(PurePursuit, self).__init__(traj)
        self.horizon = horizon
        self.last_point = self.get_setpoint(self.total_time)
        self.time_error = 0

    def __call__(self, time, robot, dt):
        goal_point = self.get_setpoint(time + self.horizon)
        d_setpoint = (self.get_setpoint(time + self.horizon + dt) - goal_point)
        # # next_setpoint = self.get_setpoint(time + dt + self.horizon)
        # try:
        #     if goal_point[1] == robot.x[0]:
        #         goal_point[1] += constants.epislon
        #     b = ((goal_point[1] - robot.x[0]) * (2.0 * robot.x[1] * np.tan(robot.x[2]) - (goal_point[1] - robot.x[0])) -
        #          goal_point[2] ** 2 + robot.x[1] ** 2) / (
        #                 2.0 * (robot.x[1] - goal_point[2]) + 2.0 * np.tan(robot.x[2]) * (
        #                 goal_point[1] - robot.x[0]))
        #     a = (2.0 * b * (robot.x[1] - goal_point[2]) + goal_point[1] ** 2 + goal_point[2] ** 2 - robot.x[0] ** 2 -
        #          robot.x[1] ** 2) / (2.0 * (goal_point[1] - robot.x[0]))
        #     radius = np.sqrt(np.power(robot.x[0] - a, 2) + np.power(robot.x[1] - b, 2))
        #     dx = np.sin(robot.x[2] - np.arctan2(goal_point[2] - robot.x[1], goal_point[1] - robot.x[0]))
        #     # dx = (np.tan(robot.x[2]) * goal_point[1] + (robot.x[1] - np.tan(robot.x[2])*robot.x[0]) - goal_point[2])
        #     # print dx, np.sin(robot.x[2] - np.arctan2(goal_point[2] - robot.x[1], goal_point[1] - robot.x[0]))
        #     # print np.tan(robot.x[2])
        # except:
        #     print "error"
        #     radius = np.inf
        #     dx = 0
        # speed, dis, stop = self.get_speed(robot, time)
        # self.time_error = time - point_time
        x = -((goal_point[2] - robot.x[1]) * np.sin(robot.x[2]) + (goal_point[1] - robot.x[0]) * np.cos(robot.x[2]))
        y = (goal_point[2] - robot.x[1]) * np.cos(robot.x[2]) - (goal_point[1] - robot.x[0]) * np.sin(robot.x[2])
        speed = np.linalg.norm(d_setpoint[1:3]) / dt
        radius = (x**2 + y**2) / (2 * x)
        # print x, y
        if x < 0:
            print 1
            leftRadius = radius - robot.robot_width / 2.0
            rightRadius = radius + robot.robot_width / 2.0
            ratio = leftRadius / rightRadius
            rightSpeed = 2.0 * speed / (1 + ratio)
            leftSpeed = ratio * rightSpeed
        elif x > 0:
            print 2
            leftRadius = radius + robot.robot_width / 2.0
            rightRadius = radius - robot.robot_width / 2.0
            ratio = rightRadius / leftRadius
            leftSpeed = 2.0 * speed / (1 + ratio)
            rightSpeed = ratio * leftSpeed
        else:
            leftSpeed = speed
            rightSpeed = speed
        # print "right radius= {},   left radius= {}".format(rightRadius, leftRadius)
        w = (2 * (leftSpeed - speed)) / robot.robot_width
        u = np.array([speed, w])
        # print u
        # return time > self.total_time, u

        # return np.linalg.norm(robot.x[:2] - self.get_setpoint(self.total_time)[1:3]) < 0.0001, u
        return time > self.total_time, u

    def __str__(self):
        return "pure pursuit controller"

    def get_speed(self, robot, time):
        range = 1
        range_time = time
        if time - range < 0:
            range_time = range
        points = [[i, np.linalg.norm(self.get_setpoint(i)[1:3] - robot.x[:2])] for i in
                  np.arange(range_time - range, range_time + range, 0.1)]
        # print type(points[0][1])
        # val, idx = np.min(points)
        # print val, idx
        # time = np.min(points, axis=0)
        point = points[0]
        for time, dis in points:
            if dis < point[1]:
                point = [time, dis]
        # print point[0]
        # print self.total_time,'/', point[0]
        self.time_error = point[0] - (time + self.time_error)
        # print point[0], self.total_time
        # print point[0]
        # print np.amin(points
        # points.sort(key=lambda x: x[0])
        # print point
        # time = np.min(points, key=itemgetter(1))[0]
        # np.min([for i in ]
        # print np.min((constants.max_a * distance_form_end, constants.max_a * distance_form_start))
        # print time

        return np.linalg.norm((self.get_setpoint(point[0] + self.dt) - self.get_setpoint(point[0]))[1:3]) / self.dt, \
               point[1], point[0] >= self.total_time
