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
        self.prevErrLatCte = 0
        self.integralLatCte = 0
        self.prevErrLatYaw = 0
        self.integralLatYaw = 0
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
        crossTrackError, i = closestNode(X,Y,trajectory)

        # To prevent out of bound error while checking future trajectory points
        minInd = i+100
        if minInd>=8203:
            minInd = 50
            
        yawError = wrapToPi(np.arctan2(trajectory[minInd][1] - Y, trajectory[minInd][0] - X) - psi)

        delta, self.integralLatYaw, self.prevErrLatYaw = PID(yawError,30,0.1,10,delT,self.integralLatYaw,self.prevErrLatYaw)
                
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
        velocity = 30 # Target velocity
        errorLong = velocity - xdot
        
        F, self.integralLong, self.prevErrLong = PID(errorLong,50,0.1,5,delT,self.integralLong,self.prevErrLong)
 
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
