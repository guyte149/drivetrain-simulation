import pickle
import matplotlib
# matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from controllers import *
from ss_model import DifferentialDrive, Simulation


def draw_simulation(states, text):
    states = np.array(states)
    pos = states.T[0:2].T
    plt.scatter(pos.T[0], pos.T[1], c='r', s=0.3)
    # axbox = plt.axes([0.15, 0.67, 0.25, 0.3])
    # text_box = TextBox(axbox, 'constants', initial=text)
    # text_box.on_submit(plt.draw())

def draw_trajectory(traj):
    plt.plot(traj.T[1], traj.T[2])


def draw_all(traj, states, text, inputs, timestamps):
    plt.subplot(121)
    draw_trajectory(traj)
    draw_simulation(states, text)
    plt.subplot(122)
    inputs = np.array(inputs)
    plt.plot(timestamps, inputs.T[0])


def main():
    print matplotlib.get_backend()
    with open('trajectory.pkl') as f:
        traj = pickle.load(f)

    robot = DifferentialDrive(0.6, 0.1)
    # controller = MPCTracking(traj, constants.horizon)
    # controller = LyapunovController(traj, constants.lyapunov_horizon)
    controller = PurePursuit(traj, constants.pure_pursuit_horizon)
    # controller = NonLinearController(traj, constants.nonlinear_horizon)
    # controller = WangLiController(traj, constants.wangli_horizon)
    # controller = TrajectoryController(traj)
    sim = Simulation(robot, controller)
    timestamps, states, inputs = sim.simulate(0.01)

    plt.axis('equal')

    # draw_trajectory(traj)
    # draw_simulation(states, controller.__str__())
    draw_all(traj, states, controller.__str__(), inputs, timestamps)
    plt.show()


if __name__ == '__main__':
    main()
