import numpy as np

Q = np.array([[2, 0, 0],
              [0, 2, 0],
              [0, 0, 0.002]])

R = np.matrix([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])

horizon = 2
max_a = 3
max_v = 2
max_w = 1

# lyapunov control
kl = 4
kv = 0.05
lyapunov_horizon = 0.05


# u_random_error = np.array([1, 0.5])
u_random_error = np.array([0, 0])
# u_random_std = np.array([0.1, np.pi / 20])
u_random_std = np.array([0, 0])
######################

# Nonlinear control
b = 1
C = 1

nonlinear_horizon = 0.5

k2 = b

# wang li control
p1 = 1
p2 = 1
p3 = 1

wangli_horizon = 0.1


# pure puresuit control
pure_pursuit_horizon = 0.0001
epislon = 0.000000000000001
