import cv2 
import numpy as np
import imutils as im
from MACROS import *


def sobel_binarize(gray):
    # Enhence the image with Sobel operator
    h = cv2.Sobel(gray, cv2.CV_32F, 0, 1, -1)
    v = cv2.Sobel(gray, cv2.CV_32F, 1, 0, -1)
    gray = cv2.add(h, v)
    gray = cv2.convertScaleAbs(gray, alpha=ADJUST_CONTRAST_ALPHA, beta=ADJUST_CONTRAST_BETA)
    gray = cv2.GaussianBlur(gray, *GAUSSIAN_KERNEL, GAUSSIAN_KERNEL_SIGMA)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    # binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                             cv2.THRESH_BINARY, 11, 2)
    return binary

def edge_detection(binary):
    """
    Perform edge detection on the input image.

    Parameters:
    binary (numpy.ndarray): Binary scale image.

    Returns:
    numpy.ndarray: The edge-detected image.
    """
    # Do open operation to remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, MORPH_KERNEL)
    # kernel = np.ones((1, 1), np.uint8)    
    # tmp = cv2.erode(binary, kernel, iterations=1)
    # tmp = cv2.dilate(tmp, kernel, iterations=2)
    # tmp = cv2.erode(tmp, kernel, iterations=1)
    # binary = cv2.dilate(tmp, kernel, iterations=2)
    edge = im.auto_canny(binary)
    # edge = cv2.Canny(img, 50, 150)
    cv2.imshow("Edge", edge)

    return edge


def line_length(line):
    """
    Calculate the length of the given line.

    Parameters:
    line (numpy.ndarray): The input line.

    Returns:
    float: The length of the given line.
    """
    x1, y1, x2, y2 = line[0]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def line_angle(line):
    """
    Calculate the angle of the given line.

    Parameters:
    line (numpy.ndarray): The input line.

    Returns:
    float: The angle of the given line.
    """
    x1, y1, x2, y2 = line[0]
    return np.arctan2(y2 - y1, x2 - x1)