from matplotlib.widgets import Button
import numpy as np
import random
from probo_sim.environment import Environment
from probo_sim.utils import Position, Pose, Bounds, Landmark
from probo_sim.robot import Robot

# filepath: /Users/vaughn/Documents/School/ProbRobo/probo_playground/test/test_environment.py
"""
Interactive test for the Environment class using Matplotlib.
Drive the robot with arrow keys (x/y translation) and PgUp/PgDn (rotation).
Press spacebar to execute the move.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Generate random environment
def create_random_environment():
    width = random.uniform(50, 100)
    height = random.uniform(50, 100)
    dimensions = Bounds(0, width, 0, height)
    dt = random.uniform(1, 10)
    
    # Random obstacles
    num_obstacles = random.randint(1, 5)
    obstacles = []
    for _ in range(num_obstacles):
        x_min = random.uniform(0, width * 0.7)
        y_min = random.uniform(0, height * 0.7)
        x_max = min(x_min + random.uniform(5, 20), width)
        y_max = min(y_min + random.uniform(5, 20), height)
        obstacles.append(Bounds(x_min, x_max, y_min, y_max))
    
    # Random landmarks
    num_landmarks = random.randint(1, 5)
    landmarks = []
    for i in range(num_landmarks):
        x = random.uniform(5, width - 5)
        y = random.uniform(5, height - 5)
        landmarks.append(Landmark(Position(x, y), i))
    
    # Random robot starting pose
    robot_pose = Pose(
        Position(random.uniform(5, width - 5), random.uniform(5, height - 5)),
        random.uniform(0, 2 * np.pi)
    )
    
    return Environment(dimensions, dt, obstacles, landmarks, robot_pose)


class InteractiveRobotTest:
    def __init__(self, mode: str):
        self.env = create_random_environment()
        self.robot = Robot(self.env)
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.vel_up = 0.0
        self.vel_right = 0.0
        self.vel_theta = 0.0

        if mode == "diff":
            self.diffmode = True
        else:
            self.diffmode = False
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.draw()
    
    def draw(self):
        self.ax.clear()
        
        dims = self.env.DIMENSIONS
        self.ax.set_xlim(dims.x_min, dims.x_max)
        self.ax.set_ylim(dims.y_min, dims.y_max)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw obstacles
        for obstacle in self.env.OBSTACLES:
            rect = patches.Rectangle(
                (obstacle.x_min, obstacle.y_min),
                obstacle.x_max - obstacle.x_min,
                obstacle.y_max - obstacle.y_min,
                linewidth=2, edgecolor='black', facecolor='gray', alpha=0.5
            )
            self.ax.add_patch(rect)
        
        # Draw landmarks
        for landmark in self.env.LANDMARKS:
            self.ax.plot(landmark.pos.x, landmark.pos.y, 'go', markersize=8, label=f'Landmark {landmark.id}')
        
        # Draw robot
        pose = self.env.get_robot_pose()
        arrow_length = 3
        dx_arrow = arrow_length * np.cos(pose.theta)
        dy_arrow = arrow_length * np.sin(pose.theta)
        self.ax.arrow(pose.pos.x, pose.pos.y, dx_arrow, dy_arrow,
                     head_width=1.5, head_length=1, fc='blue', ec='blue', linewidth=2)
        self.ax.plot(pose.pos.x, pose.pos.y, 'bo', markersize=8)
        
        # Draw cursor (next move)
        if self.diffmode:
            self.cursor_dist, self.cursor_dtheta = self.robot.differential_to_translational(self.vel_up,-self.vel_right)
        else:
            self.cursor_dist = Position(self.vel_right * self.env.DT, self.vel_up * self.env.DT)
            self.cursor_dtheta = self.vel_theta * self.env.DT
        pose = self.env.get_robot_pose()
        cursor_pos = pose.pos + self.cursor_dist
        cursor_theta = pose.theta + self.cursor_dtheta
        dx_cursor = arrow_length * np.cos(cursor_theta)
        dy_cursor = arrow_length * np.sin(cursor_theta)
        self.ax.arrow(cursor_pos.x, cursor_pos.y, dx_cursor, dy_cursor,
                     head_width=1.5, head_length=1, fc='red', ec='red', linewidth=2, alpha=0.7)
        self.ax.plot(cursor_pos.x, cursor_pos.y, 'ro', markersize=8, alpha=0.7)
        
        # Draw landmark proximity lines
        proximity = self.env.get_proximity_to_landmarks()
        if proximity:
            robot_pose = self.env.get_robot_pose()
            for id, dist in proximity.items():
                self.ax.plot([robot_pose.pos.x, robot_pose.pos.x - dist.x], [robot_pose.pos.y, robot_pose.pos.y - dist.y],
                   'g--', alpha=0.3, linewidth=1)
        
        self.ax.set_title(f'Robot Environment Test (Time: {self.env.time:.1f}s)\n'
                         f'Cursor: vx={self.vel_right:.1f}, vy={self.vel_up:.1f}, dθ={self.cursor_dtheta:.2f}')
        self.fig.canvas.draw()
    
    def on_key_press(self, event):
        step_size = 2.0
        angle_step = np.pi / 50
        
        if event.key == 'up':
            self.vel_up += step_size
        elif event.key == 'down':
            self.vel_up -= step_size 
        elif event.key == 'right':
            self.vel_right += (angle_step if self.diffmode else step_size)
        elif event.key == 'left':
            self.vel_right -= (angle_step if self.diffmode else step_size)
        elif event.key == 'pageup':
            self.vel_theta += angle_step
        elif event.key == 'pagedown':
            self.vel_theta -= angle_step
        elif event.key == ' ':  # spacebar
            if self.diffmode:
                self.robot.robot_step_differential(self.vel_up, -self.vel_right)
            else:
                self.robot.robot_step_translational(self.vel_right, self.vel_up, self.vel_theta)
            self.vel_up = 0.0
            self.vel_right = 0.0
            self.vel_theta = 0.0
        
        self.draw()


if __name__ == '__main__':
    test = InteractiveRobotTest("diff")
    plt.show()