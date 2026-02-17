"""
A simulation environment for a mobile robot operating in two dimensions.

The Environment class models the world that the robots navigate in. The world is continuous and two-dimensional. The world possesses an outer border, internal obstacles, and identifiable landmarks. The world also manages the passage of time and the motion of robotic agents within the world over time.

Critically, the environment tracks the robot's state. In this case, the robot's state is a vector that includes three state variables: x position, y position, and heading.
"""

from probo_sim.utils import Position, Pose, Bounds, Landmark, BearingRange
import copy

class Environment:
    """
    A class that models the world simulation environment and the robot's state.

    Attributes:
        dimensions: the horizontal and vertical size of the world
        dt: the length of each timestep, in seconds
        obstacles: a list of obstacles
        landmarks: a list of landmarks
        robot_pose: the position and heading of the robot in the world
    """

    def __init__(
        self,
        dimensions: Bounds,
        dt: float,
        obstacles: list[Bounds],
        landmarks: list[Landmark],
        robot_starting_pose: Pose,
    ):
        """
        Initialize an instance of the Environment class.

        Args:
            dimensions: the horizontal and vertical size of the world
            dt: the length of each timestep, in seconds
            obstacles: a list of obstacles
            landmarks: a list of landmarks
            robot_starting_pose: the initial position and heading of the robot
        """
        self.DIMENSIONS = dimensions
        self.DT = dt
        self.time = 0
        self.OBSTACLES = obstacles
        self.LANDMARKS = landmarks
        self.robot_pose = robot_starting_pose

    def robot_step(self, movement: Position, dtheta: float):
        """
        Update the robot's position and heading in the world. The robot should not be able to pass through obstacles or outside of the world bounds.

        Args:
            dx: change in x position
            dy: change in y position
            dtheta: change in heading

        Returns:
            Nothing, but update the robot_pose property at the end
        """
        movement = self.make_valid_motion(movement)
        self.robot_pose.pos += movement
        self.robot_pose.theta += dtheta
        self.time += self.DT
        pass

    def make_valid_motion(self, movement: Position):
        """
        Given attempted x and y motion by the robot, determine what motion is physically possible (i.e. doesn't go through any obstacles or barriers). Return the actual motion that will be executed.

        Args:
            dx: attempted change in x position
            dy: attempted change in y position

        Returns:
            dx: change in x position that should be executed
            dy: change in y position that should be executed
        """
        new_pos = self.robot_pose.pos + movement
        if not self.DIMENSIONS.within_bounds(new_pos):
            new_pos = self.DIMENSIONS.crossing_point(self.robot_pose.pos, new_pos)
        for obstacle in self.OBSTACLES:
            if not obstacle.outside_bounds(new_pos):
                new_pos = obstacle.crossing_point(self.robot_pose.pos, new_pos)

        return new_pos - self.robot_pose.pos

    def get_robot_pose(self):
        """
        Return the true robot pose.
        """
        return self.robot_pose

    def get_proximity_to_landmarks(self):
        """
        Return a dictionary of the robot's x and y range to all landmarks.
        """
        ranges = {}
        for landmark in self.LANDMARKS:
            ranges[landmark.id] = landmark.pos - self.robot_pose.pos
        return ranges

    def take_state_snapshot(self):
        """
        Return true state information about this timestep, including time, robot position, and the robot's bearing/range to landmarks, in a table format.
        """
        snapshot = {"time":self.time,
                    "pose":self.robot_pose,
                    "landmarks":self.get_proximity_to_landmarks()}
        return copy.deepcopy(snapshot)

    def get_environment_info(self):
        """
        Return static information about the environment, including dimensions, timestep size, locations and dimensions of obstacles, and locations of landmarks.
        """
        info = {"dimensions":self.DIMENSIONS,
                "timestep":self.DT,
                "obstacles":self.OBSTACLES,
                "landmarks":self.LANDMARKS}
        return info
