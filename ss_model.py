import numpy as np
import constants
import time as Time


class DifferentialDrive:
    def __init__(self, robot_width, wheel_radius):
        self.robot_width = robot_width
        self.wheel_radius = wheel_radius
        self.last_time = 0
        self.x = np.array([0.3, 0, np.pi / 2])

        self.A = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

    def update_state(self, u, dt, time=None):
        B = np.array([[np.cos(self.x[2]) * dt, 0],
                      [np.sin(self.x[2]) * dt, 0],
                      [0, dt]])
        self.x = self.A.dot(self.x) + B.dot(u)
        return self.x


class Simulation(object):
    def __init__(self, robot, controller):
        self.robot = robot
        self.controller = controller

    def simulate(self, dt):
        time = 0
        states = []
        timestamps = []
        inputs = []
        # cicles_dt = []

        while True:
            st = Time.time()
            done, u = self.controller(time, self.robot, dt)
            u_errored = u + np.random.normal(scale=constants.u_random_std, size=u.shape)
            self.robot.update_state(u_errored, dt, time=time)
            states.append(self.robot.x)
            timestamps.append(time)
            inputs.append(u_errored)
            cicle_dt = Time.time() - st
            # print cicle_dt
            # cicles_dt.append(cicles_dt)

            if done.all():
                print u
                break

            time += dt
            # print cicle_dt
        # avg_cicle_dt = np.array(cicles_dt).mean()
        # print avg_cicle_dt
        return timestamps, states, inputs
