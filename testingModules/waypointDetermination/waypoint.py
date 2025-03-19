import cv2
import numpy as np
from scipy.spatial import KDTree
import math
# from xarm.wrapper import XArmAPI

def process_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if image loaded successfully
    if image is None:
        print("Error: Unable to load image.")
        return

    # Apply GaussianBlur to reduce noise before edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)

    # Apply Canny edge detector with optimized thresholds
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours from the edge-detected image with more accurate mode
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Initialize list to store lines of coordinates
    lines = []

    # Iterate over each contour to extract coordinates
    for contour in contours:
        line = np.array([point[0] for point in contour])
        lines.append(line)
    
    # Create paths using a greedy algorithm optimized with KDTree for nearest neighbor search
    sorted_lines = []
    for line in lines:
        if line.size > 0:
            sorted_coordinates = []
            current_point = line[0]
            line = np.delete(line, 0, axis=0)
            sorted_coordinates.append(current_point.tolist())

            kdtree = KDTree(line)

            while line.size > 0:
                # Find the closest point to the current point using KDTree
                _, closest_index = kdtree.query(current_point)
                closest_point = line[closest_index]

                # Calculate the distance to the closest point
                distance = np.linalg.norm(current_point - closest_point)

                # Set a threshold to determine if the point is part of the same shape
                distance_threshold = 20  # Adjust this value as needed

                # Only add the point if it is within the distance threshold
                if distance <= distance_threshold:
                    sorted_coordinates.append(closest_point.tolist())
                    line = np.delete(line, closest_index, axis=0)
                    if line.size > 0:
                        kdtree = KDTree(line)
                    current_point = closest_point
                else:
                    # If the closest point is too far, start a new segment
                    if line.size > 0:
                        current_point = line[0]
                        line = np.delete(line, 0, axis=0)
                        sorted_coordinates.append(current_point.tolist())
                        kdtree = KDTree(line)
            sorted_lines.append(sorted_coordinates)

    # Create an empty image to display the final set of paths
    path_image = np.zeros_like(image)
    path_image = cv2.cvtColor(path_image, cv2.COLOR_GRAY2BGR)

    # Overlay the lines on the empty image
    for sorted_coordinates in sorted_lines:
        for i in range(1, len(sorted_coordinates)):
            start_point = tuple(sorted_coordinates[i - 1])
            end_point = tuple(sorted_coordinates[i])
            # Only draw the line if it is within the distance threshold
            distance = math.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)
            if distance <= 20:
                cv2.line(path_image, start_point, end_point, (255, 255, 255), 1)
    
    # Display the final path image
    cv2.imshow('Final Path Image', path_image)
    
    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return sorted_lines

# def draw_with_robot(lines, arm_ip):
#     # Initialize the xArm API
#     arm = XArmAPI(arm_ip)
    
#     # Connect to the robot arm
#     arm.connect()

#     # Set initial parameters
#     arm.motion_enable(True)
#     arm.set_mode(0)
#     arm.set_state(0)

#     # Set the drawing speed
#     speed = 50  # Adjust as needed
#     z_height_draw = 10  # Height while drawing (in mm)
#     z_height_move = 50  # Height while moving between lines (in mm)

#     # Iterate over each line to draw the path
#     for line in lines:
#         if not line:
#             continue
        
#         # Move to the starting point of the line
#         start_x, start_y = line[0]
#         arm.set_position(start_x, start_y, z_height_move, speed=speed, wait=True)
#         arm.set_position(start_x, start_y, z_height_draw, speed=speed, wait=True)

#         # Draw the line
#         for i in range(1, len(line)):
#             x, y = line[i]
#             arm.set_position(x, y, z_height_draw, speed=speed, wait=True)

#         # Lift the pen after finishing the line
#         arm.set_position(line[-1][0], line[-1][1], z_height_move, speed=speed, wait=True)

#     # Disconnect the arm
#     arm.disconnect()

if __name__ == "__main__":
    # Example usage
    image_path = 'images/obama.jpg'  # Replace with your image path
    arm_ip = '192.168.1.1'  # Replace with your robot's IP address

    lines = process_image(image_path)
    # if lines:
    #     draw_with_robot(lines, arm_ip)