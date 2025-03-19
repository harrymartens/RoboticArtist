def map_pixel_to_robot(pixel, image_width, image_height, min_x, max_x, min_y, max_y):
    """
    Maps a pixel coordinate (x, y) from the image coordinate space to the robot drawing area,
    scaling the image as large as possible (covering the canvas) while maintaining its aspect ratio.
    Coordinates are then clamped to lie within [min_x, max_x] and [min_y, max_y].

    Parameters:
      pixel: Tuple (x, y) representing the pixel coordinate.
      image_width: Width of the original image.
      image_height: Height of the original image.
      min_x, max_x: Horizontal limits of the drawing area.
      min_y, max_y: Vertical limits of the drawing area.
      
    Returns:
      Tuple (x_robot, y_robot) with coordinates in the robot's drawing space.
    """
    x_pixel, y_pixel = pixel

    # Dimensions of the drawing area.
    drawing_width = max_x - min_x
    drawing_height = max_y - min_y

    # Compute scale factors for each axis.
    scale_x = drawing_width / image_width
    scale_y = drawing_height / image_height

    # Use the larger scale factor so the scaled image covers the entire drawing area.
    scale = max(scale_x, scale_y)

    # Calculate the dimensions of the scaled image.
    scaled_width = image_width * scale
    scaled_height = image_height * scale

    # Center the scaled image in the drawing area.
    # When covering, parts of the image may lie outside the drawing area.
    offset_x = min_x - (scaled_width - drawing_width) / 2
    offset_y = min_y - (scaled_height - drawing_height) / 2

    # Map the pixel coordinate using the scale and offset.
    x_robot = offset_x + x_pixel * scale
    y_robot = offset_y + y_pixel * scale

    # Clamp the coordinates to ensure they lie within the drawing area.
    x_robot = max(min_x, min(max_x, x_robot))
    y_robot = max(min_y, min(max_y, y_robot))

    return int(x_robot), int(y_robot)


def executeDrawingCommands(arm, component_contours, image_width=500, image_height=500):
    """
    Given a dictionary (from findConnectedComponents) of contours,
    this function maps each pixel coordinate to the robot's drawing area and sends
    movement commands to the robot via its set_position method.
    
    Parameters:
      arm: Instance of RoboticArm.
      component_contours: Dictionary mapping component labels to contours (NumPy arrays).
      image_width: Width of the image in pixels (default 500).
      image_height: Height of the image in pixels (default 500).
    """
    # Drawing area limits defined as per your specification:
    drawing_min_x = 150
    drawing_max_x = 323
    drawing_min_y = -117
    drawing_max_y = 150

    for contour in component_contours.values():
        # Get the first point of the contour and convert it to robot coordinates.
        first_pt = contour[0][0]  # contour shape: (N,1,2)
        start_x, start_y = map_pixel_to_robot(first_pt, image_width, image_height,
                                                drawing_min_x, drawing_max_x,
                                                drawing_min_y, drawing_max_y)
        # Move to the start point with the pen raised.
        arm.set_position(start_x, start_y, draw=False)
        
        # Now iterate over each point in the contour.
        for point in contour:
            pt = point[0]  # Extract the [x, y] coordinate.
            x_robot, y_robot = map_pixel_to_robot(pt, image_width, image_height,
                                                  drawing_min_x, drawing_max_x,
                                                  drawing_min_y, drawing_max_y)
            # Move with the pen down to draw the line.
            arm.set_position(x_robot, y_robot, draw=True)
        
        # After finishing the component, lift the pen.
        arm.set_position(x_robot, y_robot, draw=False)