"""
This file contains several useful custom datatypes for you to use at your convenience! Generally, they provide structure for data that is commonly grouped together anyway (such as x and y coordinates, rectangle dimensions, and sensor measurements).

There is nothing you need to edit or fill in within this file, but feel free to alter the existing datatypes and add more as you see fit!
"""

from dataclasses import dataclass
import random
import math


@dataclass(unsafe_hash=True)
class Position:
    """
    Represents an xy coordinate.
    """

    x: float = 0.0
    y: float = 0.0

    def to_dict(self):
        """
        Return in dictionary format.
        """
        return {
            "x": self.x,
            "y": self.y,
        }

    def to_string(self):
        """
        Return in string format.
        """
        return f"X{self.x}Y{self.y}"
    
    def to_polar(self):
        """
        Return magnitude, bearing (zero right)
        """
        mag = math.sqrt(self.x ** 2 + self.y ** 2)
        bearing = math.atan2(self.y, self.x)
        return mag, bearing

    
    def __add__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        return Position(self.x+other.x, self.y+other.y)
    
    def __sub__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        return Position(self.x-other.x, self.y-other.y)



@dataclass(unsafe_hash=True)
class Pose:
    """
    Represents an xy coordinate with an associated heading.
    """

    pos: Position = Position()
    theta: float = 0.0

    def to_dict(self):
        """
        Return in dictionary format.
        """
        return {
            "pos": self.pos.to_dict(),
            "theta": self.theta,
        }

    def to_string(self):
        """
        Return in string format.
        """
        return self.pos.to_string() + f"T{self.theta}"


@dataclass(frozen=True)
class Bounds:
    """
    Represents any bounded area, including obstacles such as walls or the environment itself. The edge of a Bounds instance is considered to be contained by that instance.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def outside_bounds(self, pos: Position) -> bool:
        """
        Check if an xy coordinate is outside the bounds (inclusive).
        """
        return not (self.x_min < pos.x < self.x_max and self.y_min < pos.y < self.y_max)

    def within_bounds(self, pos: Position) -> bool:
        """
        Check if an xy coordinate is within the bounds (inclusive).
        """
        return self.x_min <= pos.x <= self.x_max and self.y_min <= pos.y <= self.y_max
    
    def crossing_point(self, start: Position, end: Position) -> Position:
        """
        Returns the single intersection point of a movement with the Bounds.
        Assumes segment starts outside and is small enough to cross at most one edge.
        
        start: first Position of movement
        end: second Position of movement
        
        Returns Position of intersection point or raises ValueError if none found.
        """

        dx = end.x - start.x
        dy = end.y - start.y

        # Check intersection with vertical edges
        if dx != 0:
            for x_edge in (self.x_min, self.x_max):
                t = (x_edge - start.x) / dx
                if 0 <= t <= 1:
                    y = start.y + t * dy
                    if self.y_min <= y <= self.y_max:
                        return Position(x_edge, y)

        # Check intersection with horizontal edges
        if dy != 0:
            for y_edge in (self.y_min, self.y_max):
                t = (y_edge - start.y) / dy
                if 0 <= t <= 1:
                    x = start.x + t * dx
                    if self.x_min <= x <= self.x_max:
                        return Position(x, y_edge)
                    
        raise ValueError("No intersection found")

    def to_dict(self):
        """
        Return in dictionary format.
        """
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }

    def to_string(self):
        """
        Return in string format.
        """
        return f"X{self.x_min}-{self.x_max}Y{self.y_min}-{self.y_max}"


@dataclass(frozen=True)
class Landmark:
    """
    Represents an identifiable floating-point landmark.
    """

    pos: Position
    id: int

    def to_dict(self):
        """
        Return in dictionary format.
        """
        return {
            "id": self.id,
            "pos": self.pos.to_dict(),
        }

    def to_string(self):
        """
        Return in string format.
        """
        return f"L{self.id}" + self.pos.to_string()


@dataclass(frozen=True)
class BearingRange:
    """
    Represents the relationship between the robot and a landmark.
    """

    landmark_id: float
    bearing: float
    range: float

    def to_dict(self):
        """
        Return in dictionary format.
        """
        return {
            "landmark_id": self.landmark_id,
            "bearing": self.bearing,
            "range": self.range,
        }

    def to_string(self):
        """
        Return in string format.
        """
        return f"LM{self.landmark_id}B{self.bearing}R{self.range}"
