import cv2
import numpy as np
from xarm import XArmAPI

# xArm connection details
IP_ADDRESS = '192.168.1.100'  # Replace with your xArm's IP address

def initialize_xarm(ip_address):
    # Connect to the xArm6 robot
    arm = XArmAPI(ip_address, is_radian=True)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)
    return arm

def convert_contours_to_paths(contours, scaling_factor=0.1, z_height=150, roll=-180, pitch=0, yaw=0):
    """
    Convert contours to a list of paths for the xArm6 robot.
    
    :param contours: List of contours (detected from the image)
    :param scaling_factor: Scale to convert pixel coordinates to robot coordinates
    :param z_height: The height for the drawing plane
    :param roll, pitch, yaw: Orientation of the end effector
    :return: List of paths in the format [x, y, z, roll, pitch, yaw]
    """
    paths = []
    for contour in contours:
        for point in contour:
            x = point[0][0] * scaling_factor
            y = point[0][1] * scaling_factor
            paths.append([300 + x, y, z_height, roll, pitch, yaw])
    return paths

def move_robot_along_paths(arm, paths, speed=300):
    """
    Move the xArm6 robot along the specified paths.
    
    :param arm: The xArm6 API instance
    :param paths: List of paths for the robot to follow
    :param speed: Movement speed of the robot
    """
    # Move to the first point
    arm.set_position(*paths[0], wait=True)
    arm.set_pause_time(0.2)
    
    # Move along the path
    for path in paths:
        ret = arm.set_position(*path, radius=0, is_radian=False, wait=False, speed=speed)
        if ret < 0:
            print('set_position failed, ret={}'.format(ret))
            break

def process_image_and_draw(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours to paths for the robot
    paths = convert_contours_to_paths(contours, scaling_factor=0.1)

    # Initialize the robot
    arm = initialize_xarm(IP_ADDRESS)

    # Move the robot along the paths
    move_robot_along_paths(arm, paths)

    # Return to the home position
    arm.move_gohome(wait=True)

    # Disconnect from the robot
    arm.disconnect()

# Example usage
process_image_and_draw('images/dolphin.png')
