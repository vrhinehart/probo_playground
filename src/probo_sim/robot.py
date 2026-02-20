"""
A simulated robotic agent with teleoperation and sensing capabilities.

The Robot class models the robotic agent that explores the world. The robot is remote-controlled by angular and linear velocity commands read from an external file. The robot can execute motor commands to move, and can sense both externally (GPS, landmarks, obstacles) and internally (odometry, IMU).
"""

from probo_sim.environment import Environment
from probo_sim.sensors import SensorInterface, PolarEncoder, TransEncoder, LandmarkPinger, GPS
from probo_sim.utils import Position
import numpy as np


class Robot:
    """
    A class that models a simulated robotic agent.

    Attributes:
        env: the environment this robot is operating in
        sensors: list of all robot sensors
    """

    def __init__(self, env: Environment):
        """
        Initialize an instance of the Robot class.

        Args:
            env: the environment this robot is operating in
        """
        self.env = env
        self.sensors = [TransEncoder(self), LandmarkPinger(self), GPS(self), PolarEncoder(self)]
        self.lin_dist = 0
        self.ang_dist = 0
        self.last_trans_vel = (Position(0,0),0)
        self.last_polar_vel = (0,0)

    def differential_to_translational(self, lin_vel: float, ang_vel: float):
        """
        Given forward linear and angular velocities, determine the robot's change in x, y, and heading

        Args:
            lin_vel: input linear velocity command
            ang_vel: input angular velocity command

        
        """
        dt = self.env.DT
        theta = self.env.robot_pose.theta
        if ang_vel == 0:
            dx = lin_vel * np.cos(theta) * dt
            dy = lin_vel * np.sin(theta) * dt
        else:
            rad = lin_vel/ang_vel
            dx = rad * np.sin(ang_vel*dt) * np.cos(theta) - rad*(1-np.cos(ang_vel*dt))*np.sin(theta)
            dy = rad * np.sin(ang_vel*dt) * np.sin(theta) + rad*(1-np.cos(ang_vel*dt))*np.cos(theta)
        dtheta = ang_vel*dt
        return Position(dx, dy), dtheta
    
    def robot_step_differential(self, lin_vel: float, ang_vel: float):
        """
        Differential-drive mode. Given forward linear and angular velocities, determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            lin_vel: input linear velocity command
            ang_vel: input angular velocity command

        Returns:
            Position(dx,dy): change in position
            d-theta: change in heading
        """
        dt = self.env.DT
        dist, dtheta = self.differential_to_translational(lin_vel, ang_vel)
        self.lin_dist += lin_vel * dt
        self.ang_dist += dtheta
        self.env.robot_step(dist, dtheta)
        self.last_trans_vel = (Position(dist.x/dt, dist.y/dt), dtheta/dt)
        self.last_polar_vel = (lin_vel, ang_vel)
        return dist, dtheta

    def robot_step_translational(self, x_vel: float, y_vel: float, ang_vel: float):
        """
        Swerve-drive mode. Given x, y, and angular velocities, determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            x_vel: input x velocity command
            y_vel: input y velocity command
            ang_vel: input angular velocity command

        Returns:
            Position(dx,dy): change in position
            d-theta: change in heading
        """
        dt = self.env.DT
        dx = x_vel * dt
        dy = y_vel * dt
        dtheta = ang_vel * dt
        move = Position(dx, dy)
        self.env.robot_step(move, dtheta)
        self.last_vel = (Position(dx/dt, dy/dt), dtheta/dt)
        return move, dtheta
    
    # def true_encoder_differential(self):
    #     """
    #     Returns the true linear and angular integrated encoder values for the Robot.
    #     Only accounts for motion made with the robot_step_differential function.
    #     """
    #     return self.lin_dist, self.ang_dist

    def take_sensor_measurements(self):
        """
        Return noisy sensor readings of the environment at this timestep, including data from all sensors, in a table format.
        """
        measurements = {}
        for sensor in self.sensors:
            if (sensor.last_meas_t + sensor.interval < self.env.time):# or self.env.time == 0:
                data = sensor.sample()
                measurements[sensor.name] = data
        return measurements

