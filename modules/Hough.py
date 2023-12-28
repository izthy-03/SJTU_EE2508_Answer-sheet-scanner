import cv2
import numpy as np
import imutils as im
from MACROS import *
from utils import *


def hough_longest_line(img, edge=None, constrain:callable=None, verbose=False, maxLineGap=20, minLineGap=20):
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
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=maxLineGap)
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
    longest_line = np.array([0, 0, 0, 0])
    for line in lines:
        # Validate the line
        if constrain is not None and not constrain(line[0]):
            continue 
        if line_length(line[0]) > line_length(longest_line):
            longest_line = line[0]

    if verbose:
        temp = np.copy(img)
        x1, y1, x2, y2 = longest_line
        cv2.line(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.namedWindow("Longest line", cv2.WINDOW_NORMAL)
        cv2.imshow("Longest line", temp)

    if line_length(longest_line) == 0:
        return None
    
    return longest_line


def hough_intersection(edge, img=None):
    lines = cv2.HoughLines(edge, 1, np.pi / 180, threshold=100)

    # 画出直线
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 找到交点
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]
            rho1, theta1 = line1
            rho2, theta2 = line2
            A = np.array([[np.cos(theta1), np.sin(theta1)],
                        [np.cos(theta2), np.sin(theta2)]])
            b = np.array([rho1, rho2])
            intersection = np.linalg.solve(A, b)
            intersections.append(tuple(map(int, intersection)))

    # 在图像上标记交点
    if img is not None:
        for point in intersections:
            cv2.circle(img, point, 5, (0, 255, 0), -1)
        cv2.namedWindow("Intersections", cv2.WINDOW_NORMAL)
        cv2.imshow("Intersections", img)

    return intersections