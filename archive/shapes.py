import math
from abc import ABC, abstractmethod
import random
from roboticMovement.robotConfig import RoboticArm

global_arm = RoboticArm()
# global_arm = 1

class Location:
    def __init__(self, x=None, y=None):
        min_x = 100 + 50  # 150
        max_x = 373 - 50  # 323
        max_y = 200 - 50  # 150 (top)
        min_y = -167 + 50 # -117 (bottom)

        if x is None:
            x = random.uniform(min_x, max_x)
        if y is None:
            y = random.uniform(min_y, max_y)
            
        self.x = x
        self.y = y

class Shape(ABC):
    def __init__(self, location):
        self.location = location 
        self.arm = global_arm           

    @abstractmethod
    def draw(self):
        pass

    def move_to(self, x, y, draw):
        self.arm.set_position(x, y, draw)
        
    def _rotate_point(self, dx, dy, angle_deg):
        """Rotate a point (dx,dy) by angle_deg degrees."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a

class Circle(Shape):
    def __init__(self, location=None, radius=10, steps=36):
        if location is None:
            location = Location()
            
        super().__init__(location)
        self.radius = radius
        self.steps = steps

    def draw(self):
        print(f"Drawing a circle with radius {self.radius}")
        angle_step = 2 * math.pi / self.steps
        loc = self.location

        # Raise the arm
        self.move_to(loc.x, loc.y, False)
        
        # Move to starting point
        start_x = loc.x + self.radius
        start_y = loc.y
        self.move_to(start_x, start_y, True)
        
        # Draw circle
        for i in range(1, self.steps + 1):
            angle = i * angle_step
            next_x = loc.x + self.radius * math.cos(angle)
            next_y = loc.y + self.radius * math.sin(angle)
            self.move_to(next_x, next_y, True)
        
        # Raise the arm after drawing
        self.move_to(loc.x, loc.y, False)
        
class Square(Shape):
    def __init__(self, location=None,  length=100, rotation=0):
        if location is None:
            location = Location()
        super().__init__(location)
        self.length = length
        self.rotation = rotation
    
    def draw(self):
        print(f"Drawing a square with rotation {self.rotation} and length {self.length}")
        half = self.length / 2.0
        loc = self.location

        # Define the vertices of an axis-aligned square relative to center.
        vertices = [(half, half), (half, -half), (-half, -half), (-half, half)]
        # Rotate and translate vertices.
        transformed = []
        for dx, dy in vertices:
            rx, ry = self._rotate_point(dx, dy, self.rotation)
            transformed.append((loc.x + rx, loc.y + ry))
        # Draw the square.
        self.move_to(transformed[0][0], transformed[0][1], False)
        self.move_to(transformed[0][0], transformed[0][1], True)
        for vx, vy in transformed[1:]:
            self.move_to(vx, vy, True)
        self.move_to(transformed[0][0], transformed[0][1], True)
        self.move_to(transformed[0][0], transformed[0][1], False)
        
class Triangle(Shape):
    def __init__(self, location=None, length=100, rotation=0):
        if location is None:
            location = Location()
        super().__init__(location)
        self.length = length
        self.rotation = rotation
    
    def draw(self):
        print(f"Drawing a triangle with rotation {self.rotation} and length {self.length}")
        R = self.length / math.sqrt(3)
        loc = self.location

        vertices = []
        for i in range(3):
            angle = math.radians(self.rotation + i * 120)
            vertices.append((loc.x + R * math.cos(angle), loc.y + R * math.sin(angle)))
        
        self.move_to(vertices[0][0], vertices[0][1], False)
        self.move_to(vertices[0][0], vertices[0][1], True)
        for vx, vy in vertices[1:]:
            self.move_to(vx, vy, True)
        self.move_to(vertices[0][0], vertices[0][1], True)
        self.move_to(vertices[0][0], vertices[0][1], False)
        
class Dot(Shape):
    def __init__(self, location=None):
        if location is None:
            location = Location()
        super().__init__(location)
    
    def draw(self):
        print("Drawing a dot")
        loc = self.location
        self.move_to(loc.x, loc.y, False)
        self.move_to(loc.x, loc.y, True)
        self.move_to(loc.x, loc.y, False)

class Line(Shape):
    def __init__(self, start=None, end=None):
        if start is None:
            start = Location()
        if end is None:
            end = Location()
        # For Line, we consider the start as the shape's location.
        super().__init__(start)
        self.start = start
        self.end = end
    
    def draw(self):
        print(f"Drawing a line from ({self.start.x:.2f}, {self.start.y:.2f}) to ({self.end.x:.2f}, {self.end.y:.2f})")
        self.move_to(self.start.x, self.start.y, False)
        self.move_to(self.start.x, self.start.y, True)
        self.move_to(self.end.x, self.end.y, True)
        self.move_to(self.end.x, self.end.y, False)

class Rectangle(Shape):
    def __init__(self, location=None, width=100, height=50, rotation=0):
        if location is None:
            location = Location()
        super().__init__(location)
        self.width = width
        self.height = height
        self.rotation = rotation
    
    def draw(self):
        print(f"Drawing a rectangle with width {self.width}, height {self.height}, and rotation {self.rotation}")
        loc = self.location
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        vertices = [(half_w, half_h), (half_w, -half_h), (-half_w, -half_h), (-half_w, half_h)]
        transformed = []
        for dx, dy in vertices:
            rx, ry = self._rotate_point(dx, dy, self.rotation)
            transformed.append((loc.x + rx, loc.y + ry))
        self.move_to(transformed[0][0], transformed[0][1], False)
        self.move_to(transformed[0][0], transformed[0][1], True)
        for vx, vy in transformed[1:]:
            self.move_to(vx, vy, True)
        self.move_to(transformed[0][0], transformed[0][1], True)
        self.move_to(transformed[0][0], transformed[0][1], False)

class Polygon(Shape):
    def __init__(self, location=None, sides=5, radius=50, rotation=0):
        if location is None:
            location = Location()
        super().__init__(location)
        self.sides = sides
        self.radius = radius
        self.rotation = rotation
    
    def draw(self):
        print(f"Drawing a polygon with {self.sides} sides, radius {self.radius}, and rotation {self.rotation}")
        loc = self.location
        vertices = []
        angle_step = 360 / self.sides
        for i in range(self.sides):
            angle = self.rotation + i * angle_step
            rad = math.radians(angle)
            vertices.append((loc.x + self.radius * math.cos(rad),
                             loc.y + self.radius * math.sin(rad)))
        self.move_to(vertices[0][0], vertices[0][1], False)
        self.move_to(vertices[0][0], vertices[0][1], True)
        for vx, vy in vertices[1:]:
            self.move_to(vx, vy, True)
        self.move_to(vertices[0][0], vertices[0][1], True)
        self.move_to(vertices[0][0], vertices[0][1], False)

class Spiral(Shape):
    def __init__(self, location=None, start_radius=5, end_radius=100, revolutions=3):
        if location is None:
            location = Location()
        super().__init__(location)
        self.start_radius = start_radius
        self.end_radius = end_radius
        self.revolutions = revolutions
    
    def draw(self):
        print(f"Drawing a spiral from radius {self.start_radius} to {self.end_radius} over {self.revolutions} revolutions")
        loc = self.location
        total_steps = 100
        self.move_to(loc.x, loc.y, False)
        # Start at angle 0 with start_radius.
        self.move_to(loc.x + self.start_radius, loc.y, True)
        for step in range(1, total_steps + 1):
            t = step / total_steps
            current_radius = self.start_radius + t * (self.end_radius - self.start_radius)
            angle = t * self.revolutions * 2 * math.pi
            next_x = loc.x + current_radius * math.cos(angle)
            next_y = loc.y + current_radius * math.sin(angle)
            self.move_to(next_x, next_y, True)
        self.move_to(loc.x, loc.y, False)

class Star(Shape):
    def __init__(self, location=None, points=5, outer_radius=50, inner_radius=25, rotation=0):
        if location is None:
            location = Location()
        super().__init__(location)
        self.points = points
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.rotation = rotation

    def draw(self):
        print(f"Drawing a star with {self.points} points, outer radius {self.outer_radius}, and inner radius {self.inner_radius}")
        loc = self.location
        vertices = []
        angle_step = 360 / (self.points * 2)
        for i in range(self.points * 2):
            r = self.outer_radius if i % 2 == 0 else self.inner_radius
            angle = self.rotation + i * angle_step
            rad = math.radians(angle)
            vertices.append((loc.x + r * math.cos(rad),
                             loc.y + r * math.sin(rad)))
        self.move_to(vertices[0][0], vertices[0][1], False)
        self.move_to(vertices[0][0], vertices[0][1], True)
        for vx, vy in vertices[1:]:
            self.move_to(vx, vy, True)
        self.move_to(vertices[0][0], vertices[0][1], True)
        self.move_to(vertices[0][0], vertices[0][1], False)

class Cube(Shape):
    def __init__(self, location=None, side_length=50, offset=(20, -20), rotation=0):
        if location is None:
            location = Location()
        super().__init__(location)
        self.side_length = side_length
        self.offset = offset  # Offset for the back face.
        self.rotation = rotation

    def draw(self):
        print(f"Drawing a cube with side length {self.side_length}, offset {self.offset}, and rotation {self.rotation}")
        loc = self.location
        half = self.side_length / 2.0
        # Front face vertices (square centered at loc)
        front_vertices = [(half, half), (half, -half), (-half, -half), (-half, half)]
        front_transformed = []
        for dx, dy in front_vertices:
            rx, ry = self._rotate_point(dx, dy, self.rotation)
            front_transformed.append((loc.x + rx, loc.y + ry))
        
        # Compute back face center by rotating the offset
        offset_rotated = self._rotate_point(self.offset[0], self.offset[1], self.rotation)
        back_center = (loc.x + offset_rotated[0], loc.y + offset_rotated[1])
        back_transformed = []
        for dx, dy in front_vertices:
            rx, ry = self._rotate_point(dx, dy, self.rotation)
            back_transformed.append((back_center[0] + rx, back_center[1] + ry))
        
        # Draw front face
        self.move_to(front_transformed[0][0], front_transformed[0][1], False)
        self.move_to(front_transformed[0][0], front_transformed[0][1], True)
        for vx, vy in front_transformed[1:]:
            self.move_to(vx, vy, True)
        self.move_to(front_transformed[0][0], front_transformed[0][1], True)
        
        # Draw back face
        self.move_to(back_transformed[0][0], back_transformed[0][1], False)
        self.move_to(back_transformed[0][0], back_transformed[0][1], True)
        for vx, vy in back_transformed[1:]:
            self.move_to(vx, vy, True)
        self.move_to(back_transformed[0][0], back_transformed[0][1], True)
        
        # Connect corresponding vertices
        for (fx, fy), (bx, by) in zip(front_transformed, back_transformed):
            self.move_to(fx, fy, False)
            self.move_to(fx, fy, True)
            self.move_to(bx, by, True)
            self.move_to(bx, by, False)

# Some pattern classes

class ConcentricCircles(Shape):
    def __init__(self, location=None, base_radius=10, count=5, step=10):
        if location is None:
            location = Location()
        super().__init__(location)
        self.base_radius = base_radius
        self.count = count
        self.step = step
    
    def draw(self):
        print(f"Drawing {self.count} concentric circles starting from radius {self.base_radius} with step {self.step}")
        for i in range(self.count):
            circle = Circle(location=self.location, radius=self.base_radius + i * self.step)
            circle.draw()

class RandomDotsPattern(Shape):
    def __init__(self, location=None, dot_count=20):
        if location is None:
            location = Location()
        super().__init__(location)
        self.dot_count = dot_count
    
    def draw(self):
        print(f"Drawing a random dots pattern with {self.dot_count} dots")
        for i in range(self.dot_count):
            dot_loc = Location()  # Random location for each dot.
            dot = Dot(location=dot_loc)
            dot.draw()

class GridPattern(Shape):
    def __init__(self, location=None, rows=3, cols=3, spacing=50):
        if location is None:
            location = Location()
        super().__init__(location)
        self.rows = rows
        self.cols = cols
        self.spacing = spacing
    
    def draw(self):
        print(f"Drawing a grid pattern with {self.rows} rows and {self.cols} columns, spacing {self.spacing}")
        # Center the grid at self.location.
        start_x = self.location.x - (self.cols - 1) * self.spacing / 2
        start_y = self.location.y - (self.rows - 1) * self.spacing / 2
        for i in range(self.rows):
            for j in range(self.cols):
                x = start_x + j * self.spacing
                y = start_y + i * self.spacing
                dot = Dot(location=Location(x, y))
                dot.draw()