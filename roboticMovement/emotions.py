#!/usr/bin/env python3

import math
import random
from roboticMovement.shapes import (
    Circle, Triangle, Square, Dot, Line, Rectangle, Polygon,
    Spiral, Star, Cube, ConcentricCircles, RandomDotsPattern, GridPattern
)
from roboticMovement.robotConfig import RoboticArm

# You can still use a global robotic arm if needed.
global_arm = RoboticArm()

# Location class: if x, y are omitted, it generates a random location.
class Location:
    def __init__(self, x=None, y=None):
        min_x = 150  # 100+50
        max_x = 323  # 373-50
        max_y = 150  # 200-50 (top)
        min_y = -117 # -167+50 (bottom)
        if x is None:
            x = random.uniform(min_x, max_x)
        if y is None:
            y = random.uniform(min_y, max_y)
        self.x = x
        self.y = y

class EmotionHappy():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Happy")
        loc = self.location
        center_circle = Circle(loc, radius=30, steps=36)
        center_circle.draw()
        for i in range(8):
            angle = i * 45  # every 45 degrees
            offset_x = 50 * math.cos(math.radians(angle))
            offset_y = 50 * math.sin(math.radians(angle))
            star_loc = Location(loc.x + offset_x, loc.y + offset_y)
            star = Star(star_loc, points=5, outer_radius=15, inner_radius=7)
            star.draw()

class EmotionSad():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Sad")
        loc = self.location
        # Draw a central circle.
        circle = Circle(loc, radius=30, steps=36)
        circle.draw()
        # Draw three "tear" dots below the circle.
        for i in range(-1, 2):
            tear_loc = Location(loc.x + i * 15, loc.y - 40)
            dot = Dot(tear_loc)
            dot.draw()

class EmotionAngry():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Angry")
        loc = self.location
        # Draw a central circle representing the core of anger.
        circle = Circle(loc, radius=30, steps=36)
        circle.draw()
        # Surround the circle with 6 spiky triangles.
        for i in range(6):
            angle = i * 60
            offset_x = 40 * math.cos(math.radians(angle))
            offset_y = 40 * math.sin(math.radians(angle))
            tri_loc = Location(loc.x + offset_x, loc.y + offset_y)
            triangle = Triangle(tri_loc, length=20, rotation=angle)
            triangle.draw()

class EmotionSurprised():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Surprised")
        loc = self.location
        # Draw an outer circle.
        outer = Circle(loc, radius=35, steps=36)
        outer.draw()
        # Draw an inner circle.
        inner = Circle(loc, radius=15, steps=36)
        inner.draw()
        # Draw 4 short radiating lines.
        for i in range(4):
            angle = i * 90
            end_loc = Location(
                loc.x + 25 * math.cos(math.radians(angle)),
                loc.y + 25 * math.sin(math.radians(angle))
            )
            line = Line(loc, end_loc)
            line.draw()

class EmotionFearful():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Fearful")
        loc = self.location
        # Draw a central dot.
        Dot(loc).draw()
        # Scatter 5 additional dots around the center.
        for i in range(5):
            offset_angle = random.uniform(0, 360)
            offset_radius = random.uniform(20, 30)
            offset_x = offset_radius * math.cos(math.radians(offset_angle))
            offset_y = offset_radius * math.sin(math.radians(offset_angle))
            dot_loc = Location(loc.x + offset_x, loc.y + offset_y)
            Dot(dot_loc).draw()

class EmotionDisgusted():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Disgusted")
        loc = self.location
        # Draw an irregular grid pattern.
        grid = GridPattern(loc, rows=2, cols=4, spacing=20)
        grid.draw()
        # Overlay with a few random dots.
        for i in range(3):
            offset = random.uniform(-10, 10)
            loc_offset = Location(loc.x + offset, loc.y + offset)
            Dot(loc_offset).draw()

class EmotionExcited():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Excited")
        loc = self.location
        # Draw a central dot.
        Dot(loc).draw()
        # Draw 12 radiating lines to create a burst effect.
        for i in range(12):
            angle = i * 30
            end_loc = Location(
                loc.x + 40 * math.cos(math.radians(angle)),
                loc.y + 40 * math.sin(math.radians(angle))
            )
            Line(loc, end_loc).draw()
        # Optionally, add a spiral for dynamic energy.
        spiral = Spiral(loc, start_radius=5, end_radius=40, revolutions=2)
        spiral.draw()

class EmotionCalm():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Calm")
        loc = self.location
        # Draw smooth, evenly spaced concentric circles.
        concentric = ConcentricCircles(loc, base_radius=10, count=4, step=10)
        concentric.draw()

class EmotionAnxious():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Anxious")
        loc = self.location
        # Draw overlapping squares with slight random shifts.
        for i in range(3):
            offset_x = random.uniform(-10, 10)
            offset_y = random.uniform(-10, 10)
            square_loc = Location(loc.x + offset_x, loc.y + offset_y)
            Square(square_loc, length=20, rotation=random.uniform(0, 360)).draw()
        # Overlay with several random dots.
        for i in range(5):
            offset_x = random.uniform(-15, 15)
            offset_y = random.uniform(-15, 15)
            dot_loc = Location(loc.x + offset_x, loc.y + offset_y)
            Dot(dot_loc).draw()

class EmotionBored():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Bored")
        loc = self.location
        # Use a simple grid of dots to suggest monotony.
        grid = GridPattern(loc, rows=3, cols=3, spacing=15)
        grid.draw()

class EmotionConfused():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Confused")
        loc = self.location
        # Overlap a circle, square, and triangle with different rotations.
        Circle(loc, radius=20, steps=36).draw()
        Square(loc, length=40, rotation=30).draw()
        Triangle(loc, length=40, rotation=60).draw()

class EmotionHopeful():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Hopeful")
        loc = self.location
        # Draw a series of triangles ascending upward.
        for i in range(3):
            offset_y = i * 20
            new_loc = Location(loc.x, loc.y + offset_y)
            Triangle(new_loc, length=20, rotation=0).draw()

class EmotionContent():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Content")
        loc = self.location
        # Arrange a circle, a square, and a star in a balanced formation.
        offsets = [(0, 10), (-15, -10), (15, -10)]
        shapes = [
            Circle(Location(loc.x + offsets[0][0], loc.y + offsets[0][1]), radius=15, steps=36),
            Square(Location(loc.x + offsets[1][0], loc.y + offsets[1][1]), length=30, rotation=0),
            Star(Location(loc.x + offsets[2][0], loc.y + offsets[2][1]), points=5, outer_radius=15, inner_radius=7)
        ]
        for shape in shapes:
            shape.draw()

class EmotionFrustrated():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Frustrated")
        loc = self.location
        # Draw several erratic, jagged lines radiating from the center.
        for i in range(4):
            angle = random.uniform(0, 360)
            end_loc = Location(
                loc.x + 30 * math.cos(math.radians(angle)),
                loc.y + 30 * math.sin(math.radians(angle))
            )
            Line(loc, end_loc).draw()
        # Draw a few erratic triangles.
        for i in range(3):
            Triangle(loc, length=25, rotation=random.uniform(0, 360)).draw()

class EmotionMelancholy():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Melancholy")
        loc = self.location
        # Draw a subdued central circle.
        Circle(loc, radius=20, steps=36).draw()
        # Draw a vertical line of dots.
        for i in range(3):
            dot_loc = Location(loc.x, loc.y - (i + 1) * 10)
            Dot(dot_loc).draw()

class EmotionSerene():
    def __init__(self, location=None):
        self.location = location if location is not None else Location()
    def draw(self):
        print("Drawing Emotion: Serene")
        loc = self.location
        # Draw a gentle spiral alongside soft concentric circles.
        Spiral(loc, start_radius=5, end_radius=30, revolutions=1).draw()
        ConcentricCircles(loc, base_radius=5, count=3, step=10).draw()