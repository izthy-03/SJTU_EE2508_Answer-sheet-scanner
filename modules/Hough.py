import cv2
import numpy as np
import imutils as im
from MACROS import *
from utils import *


def hough_longest_line(img, edge=None, constrain:callable=None, verbose=False):
    """
    Find the longest line in the image using Hough transform.

    Parameters:
    img (numpy.ndarray): The input image.
    verbose (bool): Whether to show the intermediate results.

    Returns:
    numpy.ndarray: The longest line in the image.
    """
    if edge is None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = sobel_binarize(gray)
        edge = edge_detection(binary)

    # Find the longest line
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
    if lines is None:
        raise InvalidLineError

    if verbose:
        temp = np.copy(img)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(temp, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.namedWindow("Lines", cv2.WINDOW_NORMAL)
        cv2.imshow("Lines", temp)

    # Find the longest line
    longest_line = np.array([[0, 0, 0, 0]])
    for line in lines:
        # Validate the line
        if constrain is not None and not constrain(line):
            continue 
        if line_length(line) > line_length(longest_line):
            longest_line = line

    if verbose:
        temp = np.copy(img)
        x1, y1, x2, y2 = longest_line[0]
        cv2.line(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.namedWindow("Longest line", cv2.WINDOW_NORMAL)
        cv2.imshow("Longest line", temp)

    return longest_line


