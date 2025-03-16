#!/usr/bin/env python3
"""
movementTest.py

This script initializes the robot arm and demonstrates various drawing patterns using the
PatternDrawer class from the patterns module.
"""

# arm.set_position(xStart, yStart, zRaised, roll,0,0, speed=speed)
# Top Left
# arm.set_position(100, 200, zRaised, 0,pitch,0, speed=speed)
# Top Right
# arm.set_position(373, 200, zRaised, 0,170,0, speed=speed)
# # Bottom Right
# arm.set_position(373, -167, zRaised, 0,190,0, speed=speed)
# # Bottom Left
# arm.set_position(100, -167, zRaised, 0,190,0, speed=speed)

from patterns import PatternDrawer
from xarm.wrapper import XArmAPI
arm=XArmAPI('192.168.1.111')

def getRobotArm():
    return arm

# Initialize the robot arm.
arm = getRobotArm()

# Enable motion and configure the arm.
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# Create a PatternDrawer instance with the robot arm.
drawer = PatternDrawer(arm)

# Define common parameters.
xStart = 150
yStart = 150
zRaised = 152
zLowered = 138
roll = 0
pitch = 200
yaw = 0
speed = 100

x = xStart
y = yStart

repeatCount = 5

# Uncomment the sections below to test different patterns:

# --- Draw multiple squares in a row ---
# for i in range(repeatCount):
#     drawer.drawSquare(x, y, zLowered, zRaised, 10, roll, pitch, yaw, speed)
#     x += 20

# --- Draw multiple triangles ---
# x = xStart
# y = yStart - 20
# for i in range(repeatCount):
#     drawer.drawTriangle(x, y, zLowered, zRaised, 15, roll, pitch, yaw, speed)
#     x += 30

# --- Draw multiple circles ---
# x = xStart
# y = yStart - 50
# for i in range(repeatCount):
#     drawer.drawCircle(x, y, zLowered, zRaised, 10, roll, pitch, yaw, speed)
#     x += 30

# --- Draw multiple rectangles ---
# x = xStart
# y = yStart - 80
# for i in range(repeatCount):
#     drawer.drawRectangle(x, y, zLowered, zRaised, 10, 20, roll, pitch, yaw, speed)
#     x += 20

# --- Draw a criss-cross pattern ---
# y = yStart - 100
# for i in range(repeatCount):
#     drawer.drawCrissCross(x, y, zLowered, zRaised, 50, roll, pitch, yaw, speed)
#     x += 20

# --- Draw a gap criss-cross pattern ---
# y = yStart - 100
# drawer.drawGapCrissCross(x, y, zLowered, zRaised, 50, 10, roll, pitch, yaw, speed)

# --- Draw concentric circles ---
# y = yStart - 100
# drawer.drawConcentricCircles(x, y, zLowered, zRaised, 10, 5, 10, roll, pitch, yaw, speed)

# --- Draw a spiral pattern ---
# y = yStart - 100
# drawer.drawSpiralPattern(x, y, zLowered, zRaised, 10, 5, 10, roll, pitch, yaw, speed)

# --- Draw a radial star pattern ---
y = yStart - 100
x= xStart
drawer.drawRadialStarPattern(x, y, zLowered, zRaised, 50, 8, roll, pitch, yaw, speed)

# --- Draw a grid of squares ---
# drawer.drawSquareGrid(100, 100, zLowered, zRaised, 30, 3, 3, 10, roll, pitch, yaw, speed)