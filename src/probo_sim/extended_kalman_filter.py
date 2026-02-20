"""
Extended Kalman Filter implementation for the simulator. Tracks the following states:

x = [x, y, theta]

We expect the following control inputs:

u = [v, w]
"""

import numpy as np
import sympy
from sympy.abc import x, y, v, w, R, theta
from sympy import Matrix, Symbol
import random


class ExtendedKalmanFilter:
    """
    This class implements the Extended Kalman Filter algorithm.
    """

    def __init__(self, dt: float, prior: np.ndarray):
        """
        Initialize an Extended Kalman Filter.

        A state vector includes the following:
            x position
            y position
            heading


        Args:
            dt: the length of each timestep, in seconds
            prior: the initial estimates for each state variable-
        """
        # set the timestep size to the given parameter
        self.DT: float = dt

        # set the state vector to the given prior
        self.x: np.ndarray = prior

        # set the process model to an identity matrix
        self.P: np.ndarray = np.eye(3)

        # define the nonlinear state transition model
        self.f_xu: Matrix = Matrix(
            [
                [x + v * sympy.cos(theta) * self.DT],  # calculation of x
                [y + v * sympy.sin(theta) * self.DT],  # calculation of y
                [theta + w * self.DT],  # calculation of theta
            ]
        )

        # TODO: define the Jacobian of the motion model symbolically
        self.F: Matrix = self.f_xu.jacobian(Matrix([x, y, theta]))

        # dictionary that maps Sympy symbols to numerical values. we will use these to substitute values into our symbolic matrices!
        self.subs: dict[Symbol, float] = {
            x: self.x[0],
            y: self.x[1],
            theta: self.x[2],
            v: 0,
            w: 0,
        }

    def predict(self, u: np.ndarray):
        """
        Predicts the next state vector and its covariance matrix using the state transition matrix and an input control vector. The Kalman Filter uses the following predict equations:

        x_t+1 = f(x,u)
        P_t+1 = F * P * F.T + Q

        where F is the Jacobian of f(x,u)

        Args:
            u: the input control vector
        """
        # TODO: set the value of each symbolic substitution to the actual numerical value being tracked by the EKF
        self.subs[x] = float(self.x[0])
        self.subs[y] = float(self.x[1])
        self.subs[theta] = float(self.x[2])
        self.subs[v] = float(u[0])
        self.subs[w] = float(u[1])

        # TODO: evaluate the nonlinear motion model f(x,u) at the subsitution values
        fxu_eval = sympy.matrix2numpy(self.f_xu.subs(self.subs))

        # TODO: evaluate the Jacobian matrix F at the substitution values
        F_eval = sympy.matrix2numpy(self.F.subs(self.subs))

        # TODO: calculate the next state prediction
        self.x = fxu_eval.flatten()

        # TODO: calculate the next covariance prediction
        self.P = F_eval @ self.P @ F_eval.T + self.get_Q()

        # return state vector and state covariance
        return self.x, self.P

    def update(
        self,
        H: np.ndarray,
        R: np.ndarray,
        z: np.ndarray | None,
        y_in: np.ndarray | None,
    ):
        """
        Updates the current state prediction using observations from the environment. The Extended Kalman Filter uses the following update equations:

        x = x + K * y
        P = P - K * H * P

        Where K and y are given by the following:
        y = z - h(x) (residual: error between observation and expected observation given estimated state vector)
        K = P * H.T * inv(S) (Kalman Gain: portion of total uncertainty that is from the prediction)
        S = H * P * H.T + R (total uncertainty in the system)

        where H is the Jacobian of h(x)

        Args:
            H: the Jacobian of the nonlinear measurement model, which relates the state space to the measurement space
            R: the measurement noise model (covariance)
            y: the residual, which is the error between the measured observation and the observation expected by the predicted state
        """

        # TODO: calculate the total uncertainty in the system
        S = H @ self.P @ H.T + R

        # TODO: calculate the Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S.astype('float64'))

        if y_in is None:
            y_in = z - (H @ self.x)

        # TODO: update state vector
        self.x = self.x + K @ y_in #pyright: ignore

        # TODO: update process model
        self.P = self.P - K @ H @ self.P

        # return state vector and process model
        return self.x, self.P

    def get_Q(self):
        """
        Generate white noise to apply to the process model after each prediction.
        """
        # TODO: explore different standard deviation values for this function!
        stdev = .001
        # return np.array(
        #     [
        #         [
        #             random.gauss(0, stdev),
        #             random.gauss(0, stdev),
        #             random.gauss(0, stdev),
        #         ],
        #         [
        #             random.gauss(0, stdev),
        #             random.gauss(0, stdev),
        #             random.gauss(0, stdev),
        #         ],
        #         [
        #             random.gauss(0, stdev),
        #             random.gauss(0, stdev),
        #             random.gauss(0, stdev),
        #         ],
        #     ]
        # )
        return np.diag([random.gauss(0, stdev), random.gauss(0, stdev), random.gauss(0, stdev)])
