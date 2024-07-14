# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

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

        # Add additional member variables according to your need here.
        self.e1prev = 0
        self.e2prev = 0
        self.prevErrLong = 0
        self.integralLong = 0
        
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

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
        return X, Y, xdot, ydot, psi, psidot, F, delta
