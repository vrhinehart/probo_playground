"""
Main file for running the simulator.
"""

from environment import Environment
from robot import Robot
from kalman_filter import KalmanFilter
from extended_kalman_filter import ExtendedKalmanFilter
from utils import Position, Pose, Landmark, Bounds

if __name__ == "__main__":
    # set up the environment
    # TODO: choose values for each input parameter, using the expected datatype
    dimensions = None
    dt = None
    obstacles = []
    landmarks = []
    initial_robot_pose = None

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        initial_robot_pose,
    )

    # set up the robot
    robot = Robot(env)

    # set up the (Extended) Kalman Filter
    LINEAR = True
    if LINEAR:
        kf = KalmanFilter(
            dt,
            initial_robot_pose,
        )
    else:
        # set up the Extended Kalman Filter
        kf = ExtendedKalmanFilter(
            dt,
            initial_robot_pose,
        )

    # set up timekeeping
    # TODO: set the total_seconds variable to however long you want the simulator to run (not real-time!)
    total_seconds = None
    total_timesteps = total_seconds / env.DT

    # set up logging
    ground_truth_history = []
    sensor_data_history = []
    kalman_filter_history = []

    # set up input filepath and output filepaths
    input_commands_filepath = ""
    output_ground_truth_filepath = ""
    output_sensor_data_filepath = ""
    output_kalman_filter_filepath = ""

    # open up the instructions, pop the first
    with open(input_commands_filepath, "r") as cmd:
        # iterate through each timestep
        for step in range(int(total_timesteps) + 1):
            # TODO: take a ground truth snapshot and add it to the history

            # TODO: take sensor measurements and add it to the history

            if LINEAR:
                # TODO: call the Kalman Filter prediction step

                # TODO: call the Kalman Filter update step if new sensor data is available
                pass
            else:
                # TODO: call the Extended Kalman Filter prediction step

                # TODO: call the Extended Kalman Filter update step if new sensor data is available, for each GPS reading and for each landmark ping
                pass

            # TODO: retrieve the next motor command from the input file

            # TODO: execute the motor command)

    # at the end, write the histories into output files
    with open(output_ground_truth_filepath, "w") as gt_data:
        # TODO: write ground_truth_history to a file

    with open(output_sensor_data_filepath, "w") as sensor_data:
        # TODO: write sensor_data_history to a file

    with open(output_kalman_filter_filepath, "w") as kf_data:
        # TODO: write kalman_filter_history to a file
