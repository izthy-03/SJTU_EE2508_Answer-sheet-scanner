import cv2 
import numpy as np
import imutils as im

def sort_rect_nodes(nodes):
    """
    Sort the nodes of a rectangle in the order of top-left, bottom-left, top-right, bottom-right.

    Parameters:
    nodes (numpy.ndarray): The nodes of a rectangle.

    Returns:
    numpy.ndarray: The sorted nodes.
    """
    nodes = nodes.reshape(4, 2)
    nodes = nodes[np.argsort(nodes[:, 1])]
    if nodes[0][0] > nodes[1][0]:
        nodes[[0, 1]] = nodes[[1, 0]]
    if nodes[2][0] < nodes[3][0]:   
        nodes[[2, 3]] = nodes[[3, 2]]

    return nodes