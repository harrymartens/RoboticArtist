# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import random

# def load_image(image_path):
#     """
#     Load an image in grayscale.

#     Parameters:
#         image_path (str): The path to the image file.

#     Returns:
#         image (ndarray): The loaded grayscale image.
#     """
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError("Error: Unable to load image.")
#     return image

# def kmeans_segmentation(image, k=4):
#     """
#     Apply K-means clustering to segment the image into k layers based on pixel intensity.

#     Parameters:
#         image (ndarray): The grayscale image.
#         k (int): Number of clusters/layers.

#     Returns:
#         layers (list of ndarray): List of thresholded layers.
#     """
#     # Flatten the image array and convert to float32
#     Z = image.reshape((-1, 1)).astype(np.float32)

#     # Define criteria and apply K-means clustering
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     attempts = 10
#     _, labels, centers = cv2.kmeans(
#         Z, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS
#     )

#     # Convert centers to uint8 and map labels to create segmented image
#     centers = np.uint8(centers)
#     segmented_image = centers[labels.flatten()]
#     segmented_image = segmented_image.reshape(image.shape)

#     # Create layers based on unique center values
#     layers = []
#     unique_centers = np.unique(segmented_image)
#     for center_value in unique_centers:
#         layer = np.where(segmented_image == center_value, 255, 0).astype(np.uint8)
#         layers.append(layer)
#     return layers

# def visualize_kmeans_layers(layers):
#     """
#     Display the K-means segmented layers.

#     Parameters:
#         layers (list of ndarray): List of binary layers obtained from K-means segmentation.
#     """
#     num_layers = len(layers)
#     plt.figure(figsize=(15, 5))
#     for idx, layer in enumerate(layers):
#         plt.subplot(1, num_layers, idx + 1)
#         plt.imshow(layer, cmap='gray')
#         plt.title(f'Layer {idx + 1}')
#         plt.axis('off')
#     plt.show()

# def get_connected_components(layer_image, min_area=100):
#     """
#     Extract connected components (white regions) from a binary image,
#     filtering out small components based on a minimum area threshold.

#     Parameters:
#         layer_image (ndarray): Binary image layer.
#         min_area (int): Minimum area threshold to include a component.

#     Returns:
#         components (list of ndarray): List of masks for each connected component.
#         areas (list of int): List of areas corresponding to each component.
#     """
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
#         layer_image, connectivity=8
#     )
#     components = []
#     areas = []
#     for i in range(1, num_labels):  # Skip background label 0
#         area = stats[i, cv2.CC_STAT_AREA]
#         if area >= min_area:
#             component_mask = np.zeros_like(layer_image)
#             component_mask[labels == i] = 255
#             components.append(component_mask)
#             areas.append(area)
#     return components, areas

# def sample_points(component_mask, num_samples=500):
#     """
#     Sample points within a connected component.

#     Parameters:
#         component_mask (ndarray): Mask of the connected component.
#         num_samples (int): Number of points to sample.

#     Returns:
#         sampled_points (ndarray): Array of sampled (y, x) coordinates.
#     """
#     y_indices, x_indices = np.where(component_mask == 255)
#     indices = list(zip(y_indices, x_indices))
#     if len(indices) > num_samples:
#         sampled_indices = np.random.choice(len(indices), num_samples, replace=False)
#         sampled_points = np.array([indices[i] for i in sampled_indices])
#     else:
#         sampled_points = np.array(indices)
#     return sampled_points

# def line_crosses_black(pt1, pt2, component_mask):
#     """
#     Check if a line segment between two points crosses any black pixels in the component mask.

#     Parameters:
#         pt1, pt2 (tuple): Coordinates of the two points (x, y).
#         component_mask (ndarray): Mask of the connected component.

#     Returns:
#         crosses_black (bool): True if the line crosses black pixels, False otherwise.
#     """
#     # Bresenham's Line Algorithm
#     x1, y1 = pt1
#     x2, y2 = pt2
#     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#     dx = abs(x2 - x1)
#     dy = -abs(y2 - y1)
#     sx = 1 if x1 < x2 else -1
#     sy = 1 if y1 < y2 else -1
#     err = dx + dy
#     while True:
#         if y1 < 0 or y1 >= component_mask.shape[0] or x1 < 0 or x1 >= component_mask.shape[1]:
#             return True  # Out of bounds
#         if component_mask[y1, x1] == 0:  # Black pixel
#             return True
#         if x1 == x2 and y1 == y2:
#             break
#         e2 = 2 * err
#         if e2 >= dy:
#             err += dy
#             x1 += sx
#         if e2 <= dx:
#             err += dx
#             y1 += sy
#     return False

# def fitness_function(individual, component_mask, penalty_factor=1000):
#     """
#     Calculate the fitness of an individual path.

#     Parameters:
#         individual (ndarray): Array of points representing the path.
#         component_mask (ndarray): Mask of the connected component.
#         penalty_factor (float): Penalty factor for crossing black areas.

#     Returns:
#         fitness (float): Fitness value of the individual.
#     """
#     D_t = np.sum(np.linalg.norm(np.diff(individual, axis=0), axis=1))
#     A_d = len(np.unique(individual, axis=0))
#     penalty = 0

#     for i in range(len(individual) - 1):
#         pt1 = (individual[i][1], individual[i][0])  # (x, y)
#         pt2 = (individual[i+1][1], individual[i+1][0])
#         if line_crosses_black(pt1, pt2, component_mask):
#             penalty += 1

#     fitness = (A_d / D_t) - (penalty_factor * penalty)
#     return fitness if D_t != 0 else 0

# def initialize_population(points, population_size):
#     """
#     Initialize the population for the genetic algorithm.

#     Parameters:
#         points (ndarray): Array of points within the component.
#         population_size (int): Number of individuals in the population.

#     Returns:
#         population (list of ndarray): List of individuals (paths).
#     """
#     population = []
#     for _ in range(population_size):
#         individual = points.copy()
#         np.random.shuffle(individual)
#         population.append(individual)
#     return population

# def selection(population, fitnesses, num_parents):
#     """
#     Select parents based on fitness for reproduction.

#     Parameters:
#         population (list of ndarray): List of individuals.
#         fitnesses (list of float): Fitness values corresponding to the individuals.
#         num_parents (int): Number of parents to select.

#     Returns:
#         parents (list of ndarray): Selected parents.
#     """
#     fitnesses = np.array(fitnesses)
#     # Avoid division by zero
#     fitnesses = fitnesses - fitnesses.min() + 1e-6
#     probabilities = fitnesses / fitnesses.sum()
#     parents_indices = np.random.choice(
#         len(population), size=num_parents, p=probabilities
#     )
#     parents = [population[i] for i in parents_indices]
#     return parents

# def crossover(parent1, parent2):
#     """
#     Perform crossover between two parents to produce an offspring.

#     Parameters:
#         parent1, parent2 (ndarray): Parent individuals.

#     Returns:
#         child (ndarray): Offspring individual.
#     """
#     size = len(parent1)
#     child = [None] * size

#     # Randomly select crossover points
#     start, end = sorted(np.random.choice(range(size), 2, replace=False))

#     # Copy a slice from parent1 to child
#     child[start:end] = parent1[start:end]

#     # Keep track of genes already in child using a set of tuples
#     genes_in_child = set(tuple(map(tuple, parent1[start:end])))

#     pointer = end
#     for i in range(size):
#         idx = (end + i) % size
#         gene_tuple = tuple(parent2[idx])
#         if gene_tuple not in genes_in_child:
#             child[pointer % size] = parent2[idx]
#             genes_in_child.add(gene_tuple)
#             pointer += 1

#     return np.array(child)

# def mutation(individual, mutation_rate):
#     """
#     Mutate an individual by swapping points.

#     Parameters:
#         individual (ndarray): Individual to mutate.
#         mutation_rate (float): Probability of mutation.

#     Returns:
#         mutated_individual (ndarray): Mutated individual.
#     """
#     individual = individual.copy()
#     num_mutations = max(1, int(len(individual) * mutation_rate))
#     for _ in range(num_mutations):
#         idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
#         individual[[idx1, idx2]] = individual[[idx2, idx1]]
#     return individual

# def genetic_algorithm(
#     points,
#     component_mask,
#     population_size=50,
#     num_generations=50,
#     mating_rate=0.3,
#     mutation_rate=0.1,
#     penalty_factor=1000,
# ):
#     """
#     Run the genetic algorithm to optimize the drawing path.

#     Parameters:
#         points (ndarray): Array of points within the component.
#         component_mask (ndarray): Mask of the connected component.
#         population_size (int): Number of individuals in the population.
#         num_generations (int): Number of generations to run the algorithm.
#         mating_rate (float): Proportion of the population to use as parents.
#         mutation_rate (float): Probability of mutation.
#         penalty_factor (float): Penalty factor for crossing black areas.

#     Returns:
#         best_individual (ndarray): Best path found by the algorithm.
#     """
#     population = initialize_population(points, population_size)
#     num_parents = int(population_size * mating_rate)
#     for generation in range(num_generations):
#         fitnesses = [
#             fitness_function(individual, component_mask, penalty_factor)
#             for individual in population
#         ]
#         parents = selection(population, fitnesses, num_parents)
#         offspring = []
#         while len(offspring) < population_size:
#             parent1, parent2 = random.sample(parents, 2)
#             child = crossover(parent1, parent2)
#             child = mutation(child, mutation_rate)
#             offspring.append(child)
#         population = offspring
#         # Optional: Print progress
#         if (generation + 1) % 10 == 0 or generation == 0:
#             best_fitness = max(fitnesses)
#             print(f"Generation {generation+1}: Best Fitness = {best_fitness:.4f}")
#     # Return the best individual
#     fitnesses = [
#         fitness_function(individual, component_mask, penalty_factor)
#         for individual in population
#     ]
#     best_index = np.argmax(fitnesses)
#     return population[best_index]

# def visualize_paths(component_mask, path):
#     """
#     Visualize the optimized path over the component.

#     Parameters:
#         component_mask (ndarray): Mask of the connected component.
#         path (ndarray): Optimized path.
#     """
#     h, w = component_mask.shape
#     canvas = np.zeros((h, w, 3), dtype=np.uint8)
#     # Draw the component in white
#     component_rgb = cv2.cvtColor(component_mask, cv2.COLOR_GRAY2BGR)
#     canvas = cv2.addWeighted(canvas, 1.0, component_rgb, 0.5, 0)
#     # Draw the path
#     for i in range(len(path) - 1):
#         pt1 = (int(path[i][1]), int(path[i][0]))  # (x, y)
#         pt2 = (int(path[i + 1][1]), int(path[i + 1][0]))
#         cv2.line(canvas, pt1, pt2, (0, 0, 255), 1)  # Red color
#     # Display the image
#     plt.figure(figsize=(6, 6))
#     plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
#     plt.title('Optimized Path Over Component')
#     plt.axis('off')
#     plt.show()

# def visualize_full_paths(image_shape, all_layer_paths, layer_colors):
#     """
#     Visualize all optimized paths over the original image, using different colors for each layer.

#     Parameters:
#         image_shape (tuple): Shape of the original image (height, width).
#         all_layer_paths (list of lists of ndarray): List of paths for each layer.
#         layer_colors (list of tuple): List of BGR color tuples for each layer.
#     """
#     h, w = image_shape
#     canvas = np.zeros((h, w, 3), dtype=np.uint8)

#     for layer_idx, paths in enumerate(all_layer_paths):
#         color = layer_colors[layer_idx]
#         for path in paths:
#             for i in range(len(path) - 1):
#                 pt1 = (int(path[i][1]), int(path[i][0]))  # (x, y)
#                 pt2 = (int(path[i + 1][1]), int(path[i + 1][0]))
#                 cv2.line(canvas, pt1, pt2, color, 1)  # Layer-specific color

#     # Display the image
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
#     plt.title('Optimized Paths Over Entire Image (All Layers)')
#     plt.axis('off')
#     plt.show()

# def generate_drawing_commands(path):
#     """
#     Generate drawing commands based on the optimized path.

#     Parameters:
#         path (ndarray): Optimized path.

#     Returns:
#         commands (list of str): List of drawing commands.
#     """
#     commands = []
#     commands.append("Pen up")
#     start_point = path[0]
#     commands.append(f"Move to ({start_point[1]}, {start_point[0]})")  # (x, y)
#     commands.append("Pen down")
#     for point in path[1:]:
#         commands.append(f"Draw to ({point[1]}, {point[0]})")
#     commands.append("Pen up")
#     return commands

# # Main Execution
# def threshold_background(image_path):    # Load and segment the image
#     image = load_image(image_path)
#     num_layers = 4  # Number of layers to segment
#     layers = kmeans_segmentation(image, k=num_layers)

#     # Visualize the K-means layers
#     visualize_kmeans_layers(layers)

#     # Define colors for each layer (in BGR format)
#     layer_colors = [
#         (0, 0, 255),    # Red
#         (0, 255, 0),    # Green
#         (255, 0, 0),    # Blue
#         (0, 255, 255),  # Yellow
#         # Add more colors if you have more layers
#     ]

#     # Ensure we have enough colors
#     if len(layer_colors) < num_layers:
#         raise ValueError("Not enough colors specified for the number of layers.")

#     all_layer_paths = []

#     # Set parameters
#     min_component_area = 100  # Minimum area for components
#     num_samples_per_component = 500  # Number of points to sample in each component

#     # Process each layer
#     for layer_idx, selected_layer in enumerate(layers):
#         print(f"Processing Layer {layer_idx + 1}/{num_layers}")
#         # Extract connected components with area filtering
#         components, areas = get_connected_components(selected_layer, min_area=min_component_area)

#         # Sort components by area in descending order
#         components_with_areas = sorted(zip(components, areas), key=lambda x: x[1], reverse=True)
#         components = [comp for comp, area in components_with_areas]
#         areas = [area for comp, area in components_with_areas]

#         layer_paths = []

#         # Process each component in the layer
#         for idx, (component_mask, area) in enumerate(zip(components, areas)):
#             print(f"  Component {idx+1}/{len(components)} (Area: {area})")
#             # Sample points within the component
#             points = sample_points(component_mask, num_samples=num_samples_per_component)

#             if len(points) < 2:
#                 print(f"  Component {idx+1} has insufficient points. Skipping.")
#                 continue

#             # Run the genetic algorithm
#             best_path = genetic_algorithm(
#                 points,
#                 component_mask,
#                 population_size=30,
#                 num_generations=50,
#                 mating_rate=0.3,
#                 mutation_rate=0.1,
#                 penalty_factor=1000
#             )

#             # Collect the path
#             layer_paths.append(best_path)

#             # Optional: Visualize individual component paths
#             # visualize_paths(component_mask, best_path)

#             # Generate drawing commands
#             commands = generate_drawing_commands(best_path)
#             # Output commands (optional)
#             # for cmd in commands:
#             #     print(cmd)

#         # Add the layer paths to the list
#         all_layer_paths.append(layer_paths)

#     # Visualize all paths over the entire image using different colors for each layer
#     visualize_full_paths(image.shape, all_layer_paths, layer_colors)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize

# Step 1: Load and Preprocess the Image
def preprocess_image(image_path, output_size=(512, 512)):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to the appropriate size
    resized_image = cv2.resize(image, output_size)
    
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    
    return resized_image, blurred_image

# Step 2: Apply Canny Edge Detection
def apply_canny_edge_detection(blurred_image, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return edges

# Step 3: Extract Lines for Robot to Draw
def extract_lines(edges):
    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lines = []
    for contour in contours:
        for point in contour:
            lines.append((point[0][0], point[0][1]))
    
    return lines

# Step 4: Segment the Image using KMeans
def segment_image_kmeans(blurred_image, k=4):
    # Flatten the image and convert to float32
    pixel_values = blurred_image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8 and reshape to original image dimensions
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(blurred_image.shape)
    
    return segmented_image, labels.reshape(blurred_image.shape)

# Step 5: Path Planning for Coloring Regions with Artistic Patterns
def path_plan_coloring_patterns(labels, k=4, params=None):
    if params is None:
        params = {'spacing': 15, 'angle_variation': 360}
    
    patterns = []
    for i in range(k):
        # Create a mask for each segment
        mask = np.uint8(labels == i)
        pattern = np.zeros_like(mask)
        
        # Apply different artistic drawing patterns based on the intensity of the segment
        intensity = np.mean(mask * 255)
        rows, cols = mask.shape
        spacing = max(5, params['spacing'] - int(intensity / 17))  # Adjust spacing based on intensity (darker = denser patterns, min spacing = 5)
        
        for row in range(0, rows, spacing):
            for col in range(0, cols, spacing):
                if mask[row, col] == 1:
                    # Draw more artistic shading patterns to represent the texture
                    if i % 3 == 0:
                        # Gradient-based random strokes for natural texture
                        angle = random.uniform(0, params['angle_variation'])  # Random angle for a natural look
                        length = spacing
                        x_end = int(col + length * np.cos(np.radians(angle)))
                        y_end = int(row + length * np.sin(np.radians(angle)))
                        cv2.line(pattern, (col, row), (x_end, y_end), 255, 1)
                    elif i % 3 == 1:
                        # Small angle random hatching for texture
                        perturb = int(spacing / 3)
                        x1 = col + np.random.randint(-perturb, perturb)
                        y1 = row + np.random.randint(-perturb, perturb)
                        x2 = col + spacing + np.random.randint(-perturb, perturb)
                        y2 = row + spacing + np.random.randint(-perturb, perturb)
                        cv2.line(pattern, (x1, y1), (x2, y2), 255, 1)
                    else:
                        # Curved strokes to represent texture
                        radius = spacing // 2
                        start_angle = random.randint(0, 180)
                        cv2.ellipse(pattern, (col, row), (radius, radius // 2), 0, start_angle, start_angle + 180, 255, 1)
        
        patterns.append(pattern)
    return patterns

# Step 6: Skeletonize the Image for Contour Enhancement
def skeletonize_image(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    
    while not done:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()
        
        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True
    
    return skel

# Step 7: Visualize the Results
def visualize_results(edges, segments, lines, patterns, skeleton):
    plt.figure(figsize=(15, 15))
    
    # Show Canny Edge Detection result
    plt.subplot(3, 3, 1)
    plt.title("Canny Edge Detection")
    plt.imshow(edges, cmap='gray')
    
    # Show each segmented layer
    for i, segment in enumerate(segments):
        cv2.imshow("SEG", segment)
        cv2.waitKey(0)
        plt.subplot(3, 3, i + 2)
        plt.title(f"Segment {i + 1}")
        plt.imshow(segment, cmap='gray')
    
    # Show drawing patterns for each segment
    for i, pattern in enumerate(patterns):
        plt.subplot(3, 3, i + 6)
        plt.title(f"Drawing Pattern {i + 1}")
        plt.imshow(pattern, cmap='gray')
    
    # Show reconstructed image with lines, drawing patterns, and skeletonized contours
    reconstructed_image = np.zeros_like(edges)
    for line in lines:
        cv2.circle(reconstructed_image, line, 1, (255, 255, 255), -1)
    
    for pattern in patterns:
        reconstructed_image = cv2.add(reconstructed_image, pattern)
    
    reconstructed_image = cv2.add(reconstructed_image, skeleton)
    
    plt.subplot(3, 3, 9)
    plt.title("Reconstructed Image with Lines and Patterns")
    plt.imshow(reconstructed_image, cmap='gray')
    cv2.imshow("Final", reconstructed_image)
    cv2.waitKey(0)
    
    plt.show()

# Step 8: Optimization Function for Parameter Tuning
def optimize_parameters(image_path):
    def objective_function(params):
        # Parameters to be optimized
        params_dict = {'spacing': int(params[0]), 'angle_variation': params[1]}
        
        # Run the pipeline with given parameters
        _, blurred_image = preprocess_image(image_path)
        edges = apply_canny_edge_detection(blurred_image)
        _, labels = segment_image_kmeans(blurred_image)
        patterns = path_plan_coloring_patterns(labels, params=params_dict)
        skeleton = skeletonize_image(edges)
        
        # Calculate a score based on the patterns produced (e.g., minimizing the amount of white space)
        reconstructed_image = np.zeros_like(edges)
        for pattern in patterns:
            reconstructed_image = cv2.add(reconstructed_image, pattern)
        
        non_zero_count = cv2.countNonZero(reconstructed_image)
        return -non_zero_count  # We want to maximize the non-zero pixels (i.e., coverage)
    
    # Initial guesses for parameters
    initial_params = [10, 180]
    bounds = [(5, 20), (0, 360)]  # Bounds for spacing and angle variation
    
    # Run optimization
    result = minimize(objective_function, initial_params, bounds=bounds, method='Powell')
    
    # Return optimized parameters
    return {'spacing': int(result.x[0]), 'angle_variation': result.x[1]}

# Main function to run the pipeline
def main(image_path):
    # Optimize parameters for the best result
    optimized_params = optimize_parameters(image_path)
    print(f"Optimized Parameters: {optimized_params}")
    
    resized_image, blurred_image = preprocess_image(image_path)
    
    # Apply Canny edge detection
    edges = apply_canny_edge_detection(blurred_image)
    
    # Extract lines for the robot to draw
    lines = extract_lines(edges)
    
    # Segment the blurred image using KMeans
    segmented_image, labels = segment_image_kmeans(blurred_image)
    
    # Path planning for coloring regions with artistic drawing patterns
    patterns = path_plan_coloring_patterns(labels, params=optimized_params)
    
    # Skeletonize the edge-detected image for finer details
    skeleton = skeletonize_image(edges)
    
    # Visualize the results
    visualize_results(edges, [segmented_image], lines, patterns, skeleton)


def threshold_background(image_path):
    main(image_path)
