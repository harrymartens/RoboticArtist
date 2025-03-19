import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def determine_waypoints(line_image):
    # Your existing code remains unchanged
    skeleton = cv2.ximgproc.thinning(line_image)

    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polylines = [cnt.reshape(-1, 2) for cnt in contours if cnt.shape[0] > 1]

    epsilon = 0.1  # You can adjust this value
    simplified_polylines = []
    for polyline in polylines:
        simplified = cv2.approxPolyDP(polyline, epsilon, False)
        if simplified.shape[0] > 1:
            simplified_polylines.append(simplified.reshape(-1, 2))

    start_points = np.array([polyline[0] for polyline in simplified_polylines])
    distance_matrix = squareform(pdist(start_points))

    G = nx.from_numpy_array(distance_matrix)

    tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)

    optimized_polylines = [simplified_polylines[i] for i in tsp_path]

    plt.figure(figsize=(8, 8))
    for idx, polyline in enumerate(optimized_polylines):
        plt.plot(polyline[:, 0], -polyline[:, 1], linewidth=1, color='black')
        plt.text(polyline[0, 0], -polyline[0, 1], str(idx), fontsize=8, color='red')


    # print(optimized_polylines)
    
    # for idx, polyline in enumerate(optimized_polylines):
    #     print(f"Line {idx + 1}:")
    #     print("  Pen up")
    #     start_point = polyline[0]
    #     print(f"  Move to start point ({start_point[0]:.2f}, {start_point[1]:.2f})")
    #     print("  Pen down")
    #     for point in polyline[1:]:
    #         print(f"  Draw to ({point[0]:.2f}, {point[1]:.2f})")
    #     print("  Pen up")
    #     print()
        
    plt.axis('equal')
    plt.axis('off')
    plt.show()


    
    