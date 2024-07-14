# Fill in the respective function to implement the LQR/EKF SLAM controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *
from ekf_slam import EKF_SLAM

# Function that returns the output of the PID controller
def PID(error,Kp,Ki,Kd,delT,integral,prevErr):
    
    integral = integral + error*delT
    derivative = (error-prevErr)/delT
    
    output = Kp*error + Ki*integral + Kd*derivative
    prevErr = error
    
    return output, integral, prevErr

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
        
        self.counter = 0
        np.random.seed(99)

        # Add additional member variables according to your need here.
        self.e1prev = 0
        self.e2prev = 0
        self.prevErrLong = 0
        self.integralLong = 0

    def getStates(self, timestep, use_slam=False):

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Initialize the EKF SLAM estimation
        if self.counter == 0:
            # Load the map
            minX, maxX, minY, maxY = -120., 450., -500., 50.
            map_x = np.linspace(minX, maxX, 7)
            map_y = np.linspace(minY, maxY, 7)
            map_X, map_Y = np.meshgrid(map_x, map_y)
            map_X = map_X.reshape(-1,1)
            map_Y = map_Y.reshape(-1,1)
            self.map = np.hstack((map_X, map_Y)).reshape((-1))
            
            # Parameters for EKF SLAM
            self.n = int(len(self.map)/2)             
            X_est = X + 0.5
            Y_est = Y - 0.5
            psi_est = psi - 0.02
            mu_est = np.zeros(3+2*self.n)
            mu_est[0:3] = np.array([X_est, Y_est, psi_est])
            mu_est[3:] = np.array(self.map)
            init_P = 1*np.eye(3+2*self.n)
            W = np.zeros((3+2*self.n, 3+2*self.n))
            W[0:3, 0:3] = delT**2 * 0.1 * np.eye(3)
            V = 0.1*np.eye(2*self.n)
            V[self.n:, self.n:] = 0.01*np.eye(self.n)
            # V[self.n:] = 0.01
            print(V)
            
            # Create a SLAM
            self.slam = EKF_SLAM(mu_est, init_P, delT, W, V, self.n)
            self.counter += 1
        else:
            mu = np.zeros(3+2*self.n)
            mu[0:3] = np.array([X, 
                                Y, 
                                psi])
            mu[3:] = self.map
            y = self._compute_measurements(X, Y, psi)
            mu_est, _ = self.slam.predict_and_correct(y, self.previous_u)

        self.previous_u = np.array([xdot, ydot, psidot])

        print("True      X, Y, psi:", X, Y, psi)
        print("Estimated X, Y, psi:", mu_est[0], mu_est[1], mu_est[2])
        print("-------------------------------------------------------")
        
        if use_slam == True:
            return delT, mu_est[0], mu_est[1], xdot, ydot, mu_est[2], psidot
        else:
            return delT, X, Y, xdot, ydot, psi, psidot

    def _compute_measurements(self, X, Y, psi):
        x = np.zeros(3+2*self.n)
        x[0:3] = np.array([X, Y, psi])
        x[3:] = self.map
        
        p = x[0:2]
        psi = x[2]
        m = x[3:].reshape((-1,2))

        y = np.zeros(2*self.n)

        for i in range(self.n):
            y[i] = np.linalg.norm(m[i, :] - p)
            y[self.n+i] = wrapToPi(np.arctan2(m[i,1]-p[1], m[i,0]-p[0]) - psi)
            
        y = y + np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V)
        # print(np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V))
        return y

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the newly defined getStates method
        delT, X, Y, xdot, ydot, psi, psidot = self.getStates(timestep, use_slam=True)
        # You must not use true_X, true_Y and true_psi since they are for plotting purpose
        _, true_X, true_Y, _, _, true_psi, _ = self.getStates(timestep, use_slam=False)

        # You are free to reuse or refine your code from P3 in the spaces below.

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        .
        .
        .
        .
        .
        .
        .
        .
        .
        """
        Ca, m, lf, lr, Iz = 20000, 1888.6, 1.55, 1.39, 25854
        
        A = np.array([[0, 1, 0, 0],
        [0, -4*Ca/(m*xdot), 4*Ca/m, -2*Ca*(lf-lr)/(m*xdot)],
        [0, 0, 0, 1],
        [0, -2*Ca*(lf-lr)/(Iz*xdot), 2*Ca*(lf-lr)/Iz, -2*Ca*(lf**2+lr**2)/(Iz*xdot)]])

        B = np.array([[0],
        [2*Ca/m],
        [0],
        [2*Ca*lf/Iz]])
        
        C = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        D = np.zeros((4,1))
        
        (Ad, Bd, Cd, Dd, dt) = signal.cont2discrete((A,B,C,D), delT)
        
        Q = np.array([[2, 0, 0, 0],
                      [0, 2, 0, 0],
                      [0, 0, 30, 0],
                      [0, 0, 0, 40]])
        R = 50
        
        S = np.array(linalg.solve_discrete_are(Ad, Bd, Q, R))
        K = np.array(linalg.inv(Bd.T@S@Bd+R)@(Bd.T@S@Ad))
        
        e1, i = closestNode(X,Y,trajectory)
        
        minInd = i+100
        if minInd>=8203:
            minInd = 50
        e2 = wrapToPi(np.arctan2(trajectory[minInd][1] - Y, trajectory[minInd][0] - X) - psi)
        e2dot = (e2 - self.e2prev)/delT
        self.e2prev = e2
        
        # Cross Track Error
        e1 *= np.sign(e2)
        e1dot = (e1 - self.e1prev)/delT
        self.e1prev = e1
        
        latError = np.array([[e1], [e1dot], [e2], [e2dot]])
        delta = float(K @ latError)

        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        .
        .
        .
        .
        .
        .
        .
        .
        .
        """
        velocity = 40 # Target velocity
        errorLong = velocity - xdot
        
        F, self.integralLong, self.prevErrLong = PID(errorLong,50,0.1,10,delT,self.integralLong,self.prevErrLong)

        # Return all states and calculated control inputs (F, delta)
        return true_X, true_Y, xdot, ydot, true_psi, psidot, F, delta
