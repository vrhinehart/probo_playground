"""
An abstract base class that all sensor classes must inherit from. This structure guarantees that all sensors have certain traits, including a name, sampling interval, and sampling function.

In addition to basic features, all sensors should have noise constants. Different sensors may use different distributions to model noise, and may take in different parameters to shape that noise. For example, one sensor might have a constant noise mean, while another might have noise that grows proportionally with distance or time.

Exteroceptive sensors measure the robot's relationship to the world. This includes GPS, cameras, LiDAR, and anything else that takes a measurement that can relate the robot's state to things beyond the robot.

Proprioceptive sensors measure the robot's relationship to its past states. This includes IMUs, wheel encoders, and anything else that measures how the robot's state is relatively changing, without relating the robot to the world.
"""

from abc import ABC, abstractmethod
from math import pi
import numpy as np
import random

import numpy as np

import sympy
from sympy.abc import x, y, k, j, theta
from sympy import symbols, Matrix, Symbol, pprint


class SensorInterface(ABC):
    """
    A basic Interface to standardize all sensors.

    Attributes:
        name: string identifier
        robot: reference robot. required for observing the environment
        interval: period between measurements
        last_meas_t: time of last sensor measurement
    """

    def __init__(self, name: str, robot, interval: float):
        """
        Initialize a sensor class instace.

        Args:
            name: reference identifier
            robot: reference robot
            interval: period between measurements
        """
        self._name = name
        self.robot = robot
        self._interval = interval
        self.last_meas_t = robot.env.time

    @property
    def name(self) -> str:
        """
        Getter for the name property.
        """
        return self._name

    @property
    def interval(self) -> float:
        """
        Getter for the interval property.
        """
        return self._interval

    @property
    def last_meas_t(self) -> float:
        """
        Getter for the time of last measurement property.
        """
        return self._last_meas_t

    @last_meas_t.setter
    def last_meas_t(self, value: float):
        """
        Setter for the time of last measurement property.
        """
        self._last_meas_t = value

    @abstractmethod
    def sample(self):
        """
        Sample the environment and return the noisy measurement(s).
        """
        pass


class TransEncoder(SensorInterface):
    """
    This class represents a wheel encoder set that measures the robot's motor speeds.
    Reports noisy estimates of linear and angular velocities.
    ONLY WORKS with differential updates. Translational updates don't have linear velocity odometry.

    Attributes:
        name: string identifier
        robot: reference robot
        interval: period between measurements
        last_meas_t: time of last measurement
        LIN_NOISE: absolute noise for linear velocity stdev
        ANG_NOISE: absolute noise for angular velocity stdev
    """

    def __init__(
        self,
        robot,
        name="trans_encoder",
        interval=0.0001,
        lin_noise=1,
        ang_noise=5,
    ):
        """
        Initialize an instance of the WheelEncoder class.

        Args:
            robot: reference robot
            name: reference identifier
            interval: period between measurements
            linear_noise_ratio: proportional noise for linear velocity
            angular_noise_ratio: proportional noise for angular
        """
        super().__init__(name, robot, interval)
        self.LIN_NOISE = lin_noise  # m/s
        self.ANG_NOISE = ang_noise  # rad/s

    def sample(self):  # pyright: ignore
        """
        Sample the robot's linear and angular velocity.
        """
        # true_lin, true_ang = self.robot.true_encoder_differential()
        # noisy_lin = random.gauss(true_lin, self.LIN_NOISE)
        # noisy_ang = random.gauss(true_ang,self.ANG_NOISE)
        true_vel, true_ang = self.robot.last_trans_vel
        noisy_x = random.gauss(true_vel.x, self.LIN_NOISE)
        noisy_y = random.gauss(true_vel.y, self.LIN_NOISE)
        noisy_ang = random.gauss(true_ang,self.ANG_NOISE)
        self.last_meas_t = self.robot.env.time
        return noisy_x, noisy_y, noisy_ang

class PolarEncoder(SensorInterface):
    """
    This class represents a wheel encoder set that measures the robot's motor speeds.
    Reports noisy estimates of linear and angular velocities.
    ONLY WORKS with differential updates. Translational updates don't have linear velocity odometry.

    Attributes:
        name: string identifier
        robot: reference robot
        interval: period between measurements
        last_meas_t: time of last measurement
        LIN_NOISE: absolute noise for linear velocity stdev
        ANG_NOISE: absolute noise for angular velocity stdev
    """

    def __init__(
        self,
        robot,
        name="polar_encoder",
        interval=0.0001,
        lin_noise=0.05,
        ang_noise=0.1,
    ):
        """
        Initialize an instance of the WheelEncoder class.

        Args:
            robot: reference robot
            name: reference identifier
            interval: period between measurements
            linear_noise_ratio: proportional noise for linear velocity
            angular_noise_ratio: proportional noise for angular
        """
        super().__init__(name, robot, interval)
        self.LIN_NOISE = lin_noise  # m/s
        self.ANG_NOISE = ang_noise  # rad/s

    def sample(self):  # pyright: ignore
        """
        Sample the robot's linear and angular velocity.
        """
        # true_lin, true_ang = self.robot.true_encoder_differential()
        # noisy_lin = random.gauss(true_lin, self.LIN_NOISE)
        # noisy_ang = random.gauss(true_ang,self.ANG_NOISE)
        true_vel, true_ang = self.robot.last_polar_vel
        noisy_vel = random.gauss(true_vel, self.LIN_NOISE)
        noisy_ang = random.gauss(true_ang,self.ANG_NOISE)
        self.last_meas_t = self.robot.env.time
        return noisy_vel, noisy_ang

class LandmarkPinger(SensorInterface):
    """
    This class represents a sensor that measures the range and bearing between the robot and the floating-point landmarks on the map. In practice, this sensor could be a ToF sensor, a node in a network of beacons, or even a camera.

    Attributes:
        name: reference identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        MAX_RANGE (int): maximum distance from a beacon for it to be visible
        RANGE_NOISE (float): absolute noise for range stdev
        RANGE_NOISE_RATIO (float): porportional noise for range stdev
        BEARING_NOISE (float): absolute noise for bearing stdev
    """

    def __init__(
        self,
        robot,
        name="landmark_pinger",
        interval=1.0,
        range_noise=0.5,
        range_prop_noise=0.05,
        bearing_noise=pi / 20,
        max_range=10.0,
    ):
        """
        Initialize an instance of the LandmarkPinger class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
        """
        super().__init__(name, robot, interval)
        # save max range and all noise constants as properties
        self.MAX_RANGE = max_range  # meters
        self.RANGE_NOISE = range_noise  # meters
        self.RANGE_PROP_NOISE = range_prop_noise
        self.BEARING_NOISE = bearing_noise  # radians
    
        # define the nonlinear measurement model symbolically
        self.h_x: Matrix = Matrix(
            [
                [sympy.sqrt((j - x) ** 2 + (k - y) ** 2)],  # calculation of r (range)
                [sympy.atan((j - x) / (k - y))],  # calculation of phi (bearing)
            ]
        )

        # define the Jacobian of h(x) symbolically
        self.H: Matrix = self.h_x.jacobian(Matrix([x, y, theta]))

        self.subs: dict[Symbol, float] = {
            x: 0.0,
            y: 0.0,
            theta: 0.0,
            k: 0.0,
            j: 0.0,
        }

    def sample(self):  # pyright: ignore
            """
            Reports noisy measurements of the bearing and range between the robot and all nearby landmarks.
            """
            true_prox = self.robot.env.get_proximity_to_landmarks()
            pings = {}
            for id, dist in true_prox.items():
                range, bearing = dist.to_polar()
                bearing = bearing - self.robot.env.robot_pose.theta
                if range > self.MAX_RANGE:
                    continue
                range_noise = random.gauss(0, self.RANGE_NOISE)
                range_prop_noise = random.gauss(0, self.RANGE_PROP_NOISE)
                bearing_noise = random.gauss(0, self.BEARING_NOISE)
                range += range_noise + range*range_prop_noise
                bearing += bearing_noise
                pings[id] = (range, bearing)
            self.last_meas_t = self.robot.env.time
            return pings

    def R(self, z):
        """
        Estimate variance of a given pinger measurement.

        Args:
            z (ndarray): pinger observation [[range 0], [0 bearing]]

        Returns:
            Sensor noise model for pinger measurement
        """
        bearing_stdev = self.BEARING_NOISE
        range_stdev = self.RANGE_NOISE + z[0] * self.RANGE_PROP_NOISE
        return np.diag([range_stdev, bearing_stdev]) ** 2

    def H_eval(self, x_in, lm_id):
        """
        Evaluate the Jacobian of h(x) at x, which reshapes a state vector to be in the observation space. This matrix is used to turn a state prediction into an observation prediction for a specific landmark.

        Args:
            x: the current state vector, to linearize with respect to
            lm_id: the ID of the landmark that we are predicting an observation of
        """
        # TODO: find the x and y position of the given landmark
        landmark = [landmark for landmark in self.robot.env.LANDMARKS if landmark.id == lm_id]
        landmark = landmark[0]
        lm_x = landmark.pos.x
        lm_y = landmark.pos.y

        # TODO: set the value of each symbolic substitution to the actual numerical value that was passed in
        self.subs[x] = float(x_in[0])
        self.subs[y] = float(x_in[1])
        self.subs[theta] = float(x_in[2])
        self.subs[j] = float(lm_x)  # note: we use j for landmark x position
        self.subs[k] = float(lm_y)  # note: we use k for landmark y position

        # TODO: evaluate the Jacobian at the subs values and convert it to a numpy array
        H_eval = sympy.matrix2numpy(self.H.subs(self.subs))

        # return
        return H_eval

    def y(self, z, x_in, lm_id):
        """
        Calculate the residual between an observation x and a predicted observation derived from a predicted state. The predicted observation is in reference to a specified landmark.
        """
        # TODO: find the x and y position of the given landmark
        landmark = [landmark for landmark in self.robot.env.LANDMARKS if landmark.id == lm_id]
        landmark = landmark[0]
        lm_x = landmark.pos.x
        lm_y = landmark.pos.y

        # TODO: set the value of each symbolic substitution to the actual numerical value that was passed in
        self.subs[x] = float(x_in[0])
        self.subs[y] = float(x_in[1]) # type: ignore
        self.subs[theta] = float(x_in[2])
        self.subs[j] = float(lm_x)  # note: we use j for landmark x position
        self.subs[k] = float(lm_y)  # note: we use k for landmark y position

        # TODO: evaluate the measurement model at the subs values and convert it to a numpy array
        hx_eval = sympy.matrix2numpy(self.h_x.subs(self.subs))

        # TODO: calculate the residual
        y_out = z - hx_eval.flatten()

        # return
        return y_out


class GPS(SensorInterface):
    """
    This class represents a sensor that measures the position of the robot.

    Attributes:
        name: reference identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        x_noise, y_noise: standard deviation of noise distribution
    """

    def __init__(
        self,
        robot,
        name="gps",
        interval=0.5,
        x_noise=0.5,
        y_noise=0.5,
    ):
        """
        Initialize an instance of the GPS class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
        """
        super().__init__(name, robot, interval)
        # TODO: save max range and all noise constants as properties
        self.x_noise = x_noise
        self.y_noise = y_noise

        self.R = np.diag((self.x_noise ** 2, self.y_noise ** 2))
        self.H = np.asarray(([1, 0, 0],
                            [0, 1, 0]))

    def sample(self): # pyright: ignore
        """
        Returns a tuple of noisy x and y gps reading
        """
        true_pos = self.robot.env.robot_pose.pos
        x_pos = random.gauss(true_pos.x, self.x_noise)
        y_pos = random.gauss(true_pos.y, self.y_noise)
        self.last_meas_t = self.robot.env.time
        return x_pos, y_pos

