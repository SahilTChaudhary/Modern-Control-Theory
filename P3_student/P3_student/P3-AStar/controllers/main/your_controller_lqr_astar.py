import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.signal import StateSpace, lsim, dlsim
from util import *
#from future import division, print_function

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        # Add additional member variables according to your need here.
        self.cumulative_error = 0
        self.previous_error = 0

    
    def dlqr(self, A, B, Q, R):
        S = np.matrix(linalg.solve_discrete_are(A, B, Q, R))
        K = -np.matrix(linalg.inv(B.T @ S @ B + R) @ (B.T @ S @ A))
        return K
        
        
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot, obstacleX, obstacleY = super().getStates(timestep)

        # Design your controllers in the spaces below.
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta)

        forwardindex = 40  # "looks ahead" mechanism
        _, closest_index = closestNode(X, Y, trajectory)
        if forwardindex + closest_index >= 8203:
            forwardindex = 0

        X_desired = trajectory[closest_index + forwardindex, 0]
        Y_desired = trajectory[closest_index + forwardindex, 1]

        psi_desired = np.arctan2(Y_desired - Y, X_desired - X)
        x_velocity = 15  # desired velocity

        #---------------|Lateral Controller|-------------------------

        # Please design your lateral controller below.
        # Lateral Controller
        A_lat = np.array(
            [
                [0, 1, 0, 0],
                [0, -4 * Ca / (m * xdot), 4 * Ca / m, -2 * Ca * (lf - lr) / (m * xdot)],
                [0, 0, 0, 1],
                [
                    0,
                    -2 * Ca * (lf - lr) / (Iz * xdot),
                    2 * Ca * (lf - lr) / Iz,
                    -2 * Ca * (lf ** 2 + lr ** 2) / (Iz * xdot),
                ],
            ]
        )

        B_lat = np.array([[0], [2 * Ca / m], [0], [2 * Ca * lf / Iz]])
        C_lat = np.identity(4)
        D_lat = np.array([[0], [0], [0], [0]])
        sys_ct_lat = StateSpace(A_lat, B_lat, C_lat, D_lat)
        sys_dt_lat = sys_ct_lat.to_discrete(delT)
        dt_A_lat = sys_dt_lat.A
        dt_B_lat = sys_dt_lat.B
        R_lat = 10
        Q_lat = np.array([[1000, 0, 0, 0], [0, 100, 0, 0], [0, 0, 60, 0], [0, 0, 0, 90]])
        K_lat = self.dlqr(dt_A_lat, dt_B_lat, Q_lat, R_lat)

        e1 = 0
        e2 = wrapToPi(psi - psi_desired)
        e1dot = ydot + xdot * e2
        e2dot = psidot
        e = np.hstack((e1, e1dot, e2, e2dot))
        delta = float(np.matmul(K_lat, e))
        # delta=clamp(delta, -np.pi/6, np.pi/6) #to comply assignment requirement

        #---------------|Longitudinal Controller|-------------------------

        velocity_error = x_velocity - xdot
        Kp_long = 150
        Ki_long = 69
        Kd_long = 12
        v_target = 9
        integral_error = 0
        previous_error = 0
        v = np.sqrt(xdot ** 2 + ydot ** 2)
        v_error = v_target - v
        integral_error += v_error * delT
        derivative_error = (v_error - previous_error) / delT
        P = Kp_long * v_error
        I = Ki_long * integral_error
        D = Kd_long * derivative_error
        F = P + I + D

        return X, Y, xdot, ydot, psi, psidot, F, delta, obstacleX, obstacleY
