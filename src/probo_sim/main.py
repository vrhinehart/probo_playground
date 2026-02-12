"""
Main file for running the simulator.
"""

from probo_sim.environment import Environment
from probo_sim.robot import Robot
from probo_sim.utils import Position, Pose, Landmark, Bounds
from probo_sim.kalman_filter import KalmanFilter
import pandas as pd
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == "__main__":
    # set up the environment
    # TODO: choose values for each input parameter, using the expected datatype
    dimensions = Bounds(0, 100, 0, 100)
    dt = 0.1
    obstacles = [Bounds(10, 30, 10, 30), Bounds(60, 70, 50, 90)]
    landmarks = [Landmark(Position(50,50),1), Landmark(Position(25, 30),2)]
    initial_robot_pose = Pose(Position(50,0),math.pi/2)

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        initial_robot_pose,
    )

    # set up the robot
    robot = Robot(env)

    # set up the Kalman Filter
    kf = KalmanFilter(
        dt,
        np.asarray((initial_robot_pose.pos.x, initial_robot_pose.pos.y, initial_robot_pose.theta)),
    )
    gps = [sensor for sensor in robot.sensors if sensor.name == "gps"]
    gps = gps[0]

    # set up timekeeping
    total_seconds = 20
    total_timesteps = total_seconds / env.DT

    # set up logging
    ground_truth_history = []
    sensor_data_history = []
    kalman_filter_history = []

    # set up input filepath and output filepaths
    input_commands_filepath = "./data/input.csv"
    output_ground_truth_filepath = "./data/truth.csv"
    output_sensor_data_filepath = "./data/sense.csv"
    output_kalman_filter_filepath = "./data/kalman.csv"

    # open up the instructions, pop the first
    with open(input_commands_filepath, "r") as cmd:
        reader = csv.DictReader(cmd)
        row = next(reader)
        lin_vel = ang_vel = 0
        # iterate through each timestep
        for step in range(int(total_timesteps) + 1):
            # TODO: take a ground truth snapshot and add it to the history
            snapshot = robot.env.take_state_snapshot()
            ground_truth_history.append(snapshot)
            # TODO: take sensor measurements and add it to the history
            sensor_data = robot.take_sensor_measurements()
            sensor_data_history.append(sensor_data)
            # TODO: call the Kalman Filter prediction step
            # xy_velocities, theta_velocity = robot.differential_to_translational(lin_vel, ang_vel)
            encoder_data = sensor_data["wheel_encoder"]
            kalman_x, kalman_P = kf.predict(np.asarray(encoder_data))
            kalman_filter_history.append(kalman_x)
            #TODO: call the Kalman Filter update step if new sensor data is available
            try:
                gps_data = sensor_data["gps"]
                gps_data = np.asarray(gps_data)
                kalman_x, kalman_P = kf.update(gps_data, gps.H, gps.R)
            except KeyError:
                pass
            # TODO: retrieve the next motor command from the input file
            # TODO: execute the motor command
            if float(row["timestamp"]) <= robot.env.time:
                lin_vel = float(row["linear_vel"])
                ang_vel = float(row["angular_vel"])
                try:
                    row = next(reader)
                except StopIteration:
                    pass
            robot.robot_step_differential(lin_vel, ang_vel)
 
    # at the end, write the histories into output files
    with open(output_ground_truth_filepath, "w") as gt_data:
        # TODO: write ground_truth_history to a file
        writer = csv.DictWriter(gt_data, fieldnames=ground_truth_history[0].keys())
        writer.writeheader()
        writer.writerows(ground_truth_history)
    with open(output_sensor_data_filepath, "w") as sensor_data:
        # TODO: write sensor_data_history to a file
        sensor_names = []
        for sensor in robot.sensors:
            sensor_names.append(sensor.name)
        writer = csv.DictWriter(sensor_data, fieldnames=sensor_names)
        writer.writeheader()
        writer.writerows(sensor_data_history)
    with open(output_kalman_filter_filepath, "w") as kalman_data:
        writer = csv.writer(kalman_data)
        writer.writerow(("x","y"))
        writer.writerows(kalman_filter_history)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot environment boundaries
    ax.set_xlim(env.DIMENSIONS.x_min, env.DIMENSIONS.x_max)
    ax.set_ylim(env.DIMENSIONS.y_min, env.DIMENSIONS.y_max)
    ax.set_aspect('equal')

    # Plot obstacles
    for obstacle in env.OBSTACLES:
        rect = patches.Rectangle(
            (obstacle.x_min, obstacle.y_min),
            obstacle.x_max - obstacle.x_min,
            obstacle.y_max - obstacle.y_min,
            linewidth=1, edgecolor='red', facecolor='red', alpha=0.5
        )
        ax.add_patch(rect)

        # Plot sensor rays
    gps_x = []
    gps_y = []
    for i, sensor_data in enumerate(sensor_data_history):
        robot_state = ground_truth_history[i]
        robot_x, robot_y, robot_hdg = robot_state['pose'].pos.x, robot_state['pose'].pos.y, robot_state['pose'].theta
        # if 'landmark_pinger' in sensor_data:
        #     for id, reading in sensor_data['landmark_pinger'].items():
        #         mag = reading[0]
        #         hdg = reading[1] + robot_hdg
        #         dx = mag * math.cos(hdg)
        #         dy = mag * math.sin(hdg)
        #         print(mag, hdg, dx, dy, robot_x + dx, robot_y + dy)
        #         ax.arrow(robot_x, robot_y, dx, dy,
        #                 head_width=0.5, head_length=0.3, fc='orange', ec='orange', alpha=0.3)
        if 'gps' in sensor_data:
            gps_x.append(sensor_data["gps"][0])
            gps_y.append(sensor_data["gps"][1])

    # plot GPS waypoints
    ax.plot(gps_x, gps_y, 'ko')

    # Plot landmarks
    for landmark in env.LANDMARKS:
        ax.plot(landmark.pos.x, landmark.pos.y, 'b*', markersize=15)

    # Plot robot poses
    gt_poses = [state['pose'] for state in ground_truth_history]
    gt_x = [pose.pos.x for pose in gt_poses]
    gt_y = [pose.pos.y for pose in gt_poses]
    ax.plot(gt_x, gt_y, '--g.', markersize=5)

    # Plot kalman pos history
    kf_array = np.array(kalman_filter_history)
    ax.plot(kf_array[:,0], kf_array[:,1], '--r.', markersize=5)



    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robot Trajectory and Sensor Observations')
    ax.grid(True, alpha=0.3)
    plt.show()