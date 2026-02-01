"""
Main file for running the simulator.
"""

from probo_sim.environment import Environment
from probo_sim.robot import Robot
from probo_sim.utils import Position, Pose, Landmark, Bounds
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
    dt = 1
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

    # set up timekeeping
    total_seconds = 20
    total_timesteps = total_seconds / env.DT

    # set up logging
    ground_truth_history = []
    sensor_data_history = []

    # set up input filepath and output filepaths
    input_commands_filepath = "./data/input.csv"
    output_ground_truth_filepath = "./data/truth.csv"
    output_sensor_data_filepath = "./data/sense.csv"

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
            sensor_data_history.append(robot.take_sensor_measurements())
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

    # Plot landmarks
    for landmark in env.LANDMARKS:
        ax.plot(landmark.pos.x, landmark.pos.y, 'b*', markersize=15)

    # Plot robot poses
    for state in ground_truth_history:
        ax.plot(state['pose'].pos.x, state['pose'].pos.y, 'g.', markersize=5)

    # Plot sensor rays
    for i, sensor_data in enumerate(sensor_data_history):
        robot_state = ground_truth_history[i]
        robot_x, robot_y, robot_hdg = robot_state['pose'].pos.x, robot_state['pose'].pos.y, robot_state['pose'].theta
        if 'landmark_pinger' in sensor_data:
            for id, reading in sensor_data['landmark_pinger'].items():
                mag = reading[0]
                hdg = reading[1] + robot_hdg
                dx = mag * math.cos(hdg)
                dy = mag * math.sin(hdg)
                print(mag, hdg, dx, dy, robot_x + dx, robot_y + dy)
                ax.arrow(robot_x, robot_y, dx, dy,
                        head_width=0.5, head_length=0.3, fc='orange', ec='orange', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robot Trajectory and Sensor Observations')
    ax.grid(True, alpha=0.3)
    plt.show()