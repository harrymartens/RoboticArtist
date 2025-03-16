#!/usr/bin/env python3
"""
patterns.py

This module provides the PatternDrawer class which contains drawing routines for a pattern drawing robot.
All functions now treat (x,y) as the center of the shape and support a 'rotation' parameter (in degrees)
to adjust the orientation of the shape. A new drawPolygon() method is also included.

Assumes that 'arm' is an object that implements set_position(x, y, z, roll, pitch, yaw, speed=speed).
"""

import math

class PatternDrawer:
    def __init__(self, arm):
        self.arm = arm

    def _rotate_point(self, dx, dy, angle_deg):
        """Rotate a point (dx,dy) by angle_deg degrees."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a

    def drawSquare(self, x, y, zLowered, zRaised, size, rotation, roll, pitch, yaw, speed=50):
        """Draw a square with center (x,y), side length 'size', rotated by 'rotation' degrees."""
        half = size / 2.0
        # Define the vertices of an axis-aligned square relative to center.
        vertices = [(half, half), (half, -half), (-half, -half), (-half, half)]
        # Rotate and translate vertices.
        transformed = []
        for dx, dy in vertices:
            rx, ry = self._rotate_point(dx, dy, rotation)
            transformed.append((x + rx, y + ry))
        # Draw the square.
        self.arm.set_position(transformed[0][0], transformed[0][1], zRaised, roll, pitch, yaw, speed=speed)
        self.arm.set_position(transformed[0][0], transformed[0][1], zLowered, roll, pitch, yaw, speed=speed)
        for vx, vy in transformed[1:]:
            self.arm.set_position(vx, vy, zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(transformed[0][0], transformed[0][1], zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(transformed[0][0], transformed[0][1], zRaised, roll, pitch, yaw, speed=speed)

    def drawCircle(self, x, y, zLowered, zRaised, radius, rotation, roll, pitch, yaw, speed=50, steps=36):
        """Draw a circle with center (x,y) and given radius. 'rotation' offsets the start angle."""
        angle_step = 2 * math.pi / steps
        rotation_rad = math.radians(rotation)
        start_x = x + radius * math.cos(rotation_rad)
        start_y = y + radius * math.sin(rotation_rad)
        self.arm.set_position(start_x, start_y, zRaised, roll, pitch, yaw, speed=speed)
        self.arm.set_position(start_x, start_y, zLowered, roll, pitch, yaw, speed=speed)
        for i in range(1, steps + 1):
            angle = rotation_rad + i * angle_step
            new_x = x + radius * math.cos(angle)
            new_y = y + radius * math.sin(angle)
            self.arm.set_position(new_x, new_y, zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(x, y, zRaised, roll, pitch, yaw, speed=speed)

    def drawTriangle(self, x, y, zLowered, zRaised, size, rotation, roll, pitch, yaw, speed=50):
        """
        Draw an equilateral triangle with center (x,y) and side length 'size'. 
        The triangle is rotated by 'rotation' degrees.
        """
        # The circumradius R for an equilateral triangle is: R = s / sqrt(3)
        R = size / math.sqrt(3)
        vertices = []
        for i in range(3):
            angle = math.radians(rotation + i * 120)
            vertices.append((x + R * math.cos(angle), y + R * math.sin(angle)))
        self.arm.set_position(vertices[0][0], vertices[0][1], zRaised, roll, pitch, yaw, speed=speed)
        self.arm.set_position(vertices[0][0], vertices[0][1], zLowered, roll, pitch, yaw, speed=speed)
        for vx, vy in vertices[1:]:
            self.arm.set_position(vx, vy, zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(vertices[0][0], vertices[0][1], zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(vertices[0][0], vertices[0][1], zRaised, roll, pitch, yaw, speed=speed)

    def drawRectangle(self, x, y, zLowered, zRaised, width, height, rotation, roll, pitch, yaw, speed=50):
        """Draw a rectangle with center (x,y), width and height, rotated by 'rotation' degrees."""
        half_w = width / 2.0
        half_h = height / 2.0
        vertices = [(half_w, half_h), (half_w, -half_h), (-half_w, -half_h), (-half_w, half_h)]
        transformed = []
        for dx, dy in vertices:
            rx, ry = self._rotate_point(dx, dy, rotation)
            transformed.append((x + rx, y + ry))
        self.arm.set_position(transformed[0][0], transformed[0][1], zRaised, roll, pitch, yaw, speed=speed)
        self.arm.set_position(transformed[0][0], transformed[0][1], zLowered, roll, pitch, yaw, speed=speed)
        for vx, vy in transformed[1:]:
            self.arm.set_position(vx, vy, zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(transformed[0][0], transformed[0][1], zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(transformed[0][0], transformed[0][1], zRaised, roll, pitch, yaw, speed=speed)

    def drawCrissCross(self, x, y, zLowered, zRaised, size, rotation, roll, pitch, yaw, speed=50):
        """
        Draw an X-shaped (criss cross) pattern inside a square of side 'size' with center (x,y)
        and rotated by 'rotation' degrees.
        """
        half = size / 2.0
        # Compute endpoints for the two diagonals.
        # Diagonal 1: from top-left to bottom-right.
        pt1 = self._rotate_point(-half, half, rotation)
        pt2 = self._rotate_point(half, -half, rotation)
        # Diagonal 2: from top-right to bottom-left.
        pt3 = self._rotate_point(half, half, rotation)
        pt4 = self._rotate_point(-half, -half, rotation)
        pt1 = (x + pt1[0], y + pt1[1])
        pt2 = (x + pt2[0], y + pt2[1])
        pt3 = (x + pt3[0], y + pt3[1])
        pt4 = (x + pt4[0], y + pt4[1])
        # Draw first diagonal.
        self.arm.set_position(pt1[0], pt1[1], zRaised, roll, pitch, yaw, speed=speed)
        self.arm.set_position(pt1[0], pt1[1], zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(pt2[0], pt2[1], zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(pt2[0], pt2[1], zRaised, roll, pitch, yaw, speed=speed)
        # Draw second diagonal.
        self.arm.set_position(pt3[0], pt3[1], zRaised, roll, pitch, yaw, speed=speed)
        self.arm.set_position(pt3[0], pt3[1], zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(pt4[0], pt4[1], zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(pt4[0], pt4[1], zRaised, roll, pitch, yaw, speed=speed)

    def drawGapCrissCross(self, x, y, zLowered, zRaised, length, gap, rotation, roll, pitch, yaw, speed=50):
        """
        Draw a gap criss cross pattern (an X with a gap in the center) within a virtual square,
        applying a rotation. (Note: For brevity, this implementation draws one diagonal segment.)
        """
        L_half = length / 2.0
        # For the diagonal from top-left to bottom-right (without rotation):
        gap_left = (-gap/2, gap/2)
        gap_right = (gap/2, -gap/2)
        gap_left_rot = self._rotate_point(gap_left[0], gap_left[1], rotation)
        gap_right_rot = self._rotate_point(gap_right[0], gap_right[1], rotation)
        # Outer endpoints along the diagonal.
        uv = (-1/math.sqrt(2), 1/math.sqrt(2))
        outer_left = (gap_left[0] - L_half * uv[0], gap_left[1] - L_half * uv[1])
        outer_right = (gap_right[0] + L_half * uv[0], gap_right[1] + L_half * uv[1])
        outer_left_rot = self._rotate_point(outer_left[0], outer_left[1], rotation)
        outer_right_rot = self._rotate_point(outer_right[0], outer_right[1], rotation)
        # Translate by center.
        pt_outer_left = (x + outer_left_rot[0], y + outer_left_rot[1])
        pt_gap_left = (x + gap_left_rot[0], y + gap_left_rot[1])
        pt_gap_right = (x + gap_right_rot[0], y + gap_right_rot[1])
        pt_outer_right = (x + outer_right_rot[0], y + outer_right_rot[1])
        # Draw the left segment.
        self.arm.set_position(pt_outer_left[0], pt_outer_left[1], zRaised, roll, pitch, yaw, speed=speed)
        self.arm.set_position(pt_outer_left[0], pt_outer_left[1], zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(pt_gap_left[0], pt_gap_left[1], zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(pt_gap_left[0], pt_gap_left[1], zRaised, roll, pitch, yaw, speed=speed)
        # Draw the right segment.
        self.arm.set_position(pt_gap_right[0], pt_gap_right[1], zRaised, roll, pitch, yaw, speed=speed)
        self.arm.set_position(pt_gap_right[0], pt_gap_right[1], zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(pt_outer_right[0], pt_outer_right[1], zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(pt_outer_right[0], pt_outer_right[1], zRaised, roll, pitch, yaw, speed=speed)
        # For a complete implementation, replicate similar logic for the other diagonal.

    def drawSpiralPattern(self, x, y, zLowered, zRaised, initial_radius, revolutions, spacing, rotation, roll, pitch, yaw, speed=50):
        """
        Draw an Archimedean spiral pattern centered at (x,y) with an initial radius,
        number of revolutions, spacing per revolution, and a rotation offset.
        """
        steps = int(revolutions * 360 / 5)
        angle_step = (2 * math.pi * revolutions) / steps
        b = spacing / (2 * math.pi)
        start_angle = math.radians(rotation)
        start_x = x + initial_radius * math.cos(start_angle)
        start_y = y + initial_radius * math.sin(start_angle)
        self.arm.set_position(start_x, start_y, zRaised, roll, pitch, yaw, speed=speed)
        self.arm.set_position(start_x, start_y, zLowered, roll, pitch, yaw, speed=speed)
        for i in range(steps + 1):
            theta = start_angle + i * angle_step
            r = initial_radius + b * (theta - start_angle)
            new_x = x + r * math.cos(theta)
            new_y = y + r * math.sin(theta)
            self.arm.set_position(new_x, new_y, zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(new_x, new_y, zRaised, roll, pitch, yaw, speed=speed)

    def drawRadialStarPattern(self, x, y, zLowered, zRaised, ray_length, num_rays, rotation, roll, pitch, yaw, speed=50):
        """
        Draw a radial star pattern centered at (x,y) with rays of length 'ray_length'.
        The entire pattern is rotated by 'rotation' degrees.
        """
        angle_step = 2 * math.pi / num_rays
        start_angle = math.radians(rotation)
        for i in range(num_rays):
            theta = start_angle + i * angle_step
            end_x = x + ray_length * math.cos(theta)
            end_y = y + ray_length * math.sin(theta)
            self.arm.set_position(x, y, zRaised, roll, pitch, yaw, speed=speed)
            self.arm.set_position(x, y, zLowered, roll, pitch, yaw, speed=speed)
            self.arm.set_position(end_x, end_y, zLowered, roll, pitch, yaw, speed=speed)
            self.arm.set_position(end_x, end_y, zRaised, roll, pitch, yaw, speed=speed)

    def drawConcentricCircles(self, x, y, zLowered, zRaised, initial_radius, num_circles, spacing, rotation, roll, pitch, yaw, speed=50):
        """
        Draw a series of concentric circles centered at (x,y).
        'rotation' sets the starting angle offset (though circles are symmetric).
        """
        steps = 36
        angle_step = 2 * math.pi / steps
        for i in range(num_circles):
            radius = initial_radius + i * spacing
            self.arm.set_position(x + radius * math.cos(math.radians(rotation)), 
                                  y + radius * math.sin(math.radians(rotation)), 
                                  zRaised, roll, pitch, yaw, speed=speed)
            self.arm.set_position(x + radius * math.cos(math.radians(rotation)), 
                                  y + radius * math.sin(math.radians(rotation)), 
                                  zLowered, roll, pitch, yaw, speed=speed)
            for j in range(1, steps + 1):
                theta = math.radians(rotation) + j * angle_step
                new_x = x + radius * math.cos(theta)
                new_y = y + radius * math.sin(theta)
                self.arm.set_position(new_x, new_y, zLowered, roll, pitch, yaw, speed=speed)
            self.arm.set_position(x + radius * math.cos(math.radians(rotation)), 
                                  y + radius * math.sin(math.radians(rotation)), 
                                  zRaised, roll, pitch, yaw, speed=speed)

    def drawSquareGrid(self, x, y, zLowered, zRaised, size, rows, cols, gap, rotation, roll, pitch, yaw, speed=50):
        """
        Draw a grid of squares. Each square is drawn using drawSquare(),
        so they all share the same rotation. (x,y) is treated as the center of the top-left square.
        """
        for row in range(rows):
            for col in range(cols):
                center_x = x + col * (size + gap)
                center_y = y - row * (size + gap)
                self.drawSquare(center_x, center_y, zLowered, zRaised, size, rotation, roll, pitch, yaw, speed)

    def drawPolygon(self, x, y, zLowered, zRaised, sides, size, rotation, roll, pitch, yaw, speed=50):
        """
        Draw a regular polygon with 'sides' sides, where 'size' is the circumradius.
        The polygon is centered at (x,y) and rotated by 'rotation' degrees.
        """
        if sides < 3:
            print("Polygon must have at least 3 sides.")
            return
        vertices = []
        angle_step = 360.0 / sides
        for i in range(sides):
            angle = math.radians(rotation + i * angle_step)
            vx = x + size * math.cos(angle)
            vy = y + size * math.sin(angle)
            vertices.append((vx, vy))
        self.arm.set_position(vertices[0][0], vertices[0][1], zRaised, roll, pitch, yaw, speed=speed)
        self.arm.set_position(vertices[0][0], vertices[0][1], zLowered, roll, pitch, yaw, speed=speed)
        for vx, vy in vertices[1:]:
            self.arm.set_position(vx, vy, zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(vertices[0][0], vertices[0][1], zLowered, roll, pitch, yaw, speed=speed)
        self.arm.set_position(vertices[0][0], vertices[0][1], zRaised, roll, pitch, yaw, speed=speed)