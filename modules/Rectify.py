import cv2
import imutils as im
import numpy as np
from modules.Hough import houghProcessor as hp
from MACROS import *
from utils import sort_rect_nodes

def rectify(img, verbose=False):

    original = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray)
    edge = edge_detection(binary)

    # Find the largest contour
    contour = get_largest_contour(edge)
    cv2.drawContours(img, [contour], 0, (0, 255, 0), 1)

    poly_node_list = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.1, True)
    if not poly_node_list.shape[0] == 4:
        print("Cannot find the largest rectangle in the image.")
        return original
    
    poly_node_list = np.float32(sort_rect_nodes(poly_node_list))    

    # transform the contour region to a rectangle
    warped = perspective_transform(original, poly_node_list.reshape(4, 2))

    if verbose:
        cv2.namedWindow("Rectified", cv2.WINDOW_NORMAL)
        cv2.imshow("Rectified", warped)
        cv2.waitKey(0)

    return img

def binarize(img):
    # Enhence the image with Sobel operator
    h = cv2.Sobel(img, cv2.CV_32F, 0, 1, -1)
    v = cv2.Sobel(img, cv2.CV_32F, 1, 0, -1)
    img = cv2.add(h, v)
    img = cv2.convertScaleAbs(img, alpha=ADJUST_CONTRAST_ALPHA, beta=ADJUST_CONTRAST_BETA)
    img = cv2.GaussianBlur(img, (GAUSSIAN_KERNEL_HSIZE, GAUSSIAN_KERNEL_HSIZE), GAUSSIAN_KERNEL_SIGMA)
    _, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    # binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                             cv2.THRESH_BINARY, 11, 2)
    return binary

def edge_detection(img):
    # Do open operation to remove noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=2)
    edge = im.auto_canny(img)
    # edge = cv2.Canny(img, 50, 150)
    return edge

def get_largest_contour(img):
    """
    Get the largest contour in the given image.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The largest contour in the image.
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large = max(contours, key=lambda c: cv2.contourArea(c))
    return large

def perspective_transform(img, src):
    """
    Apply perspective transform to the input image.

    Parameters:
    img (numpy.ndarray): The input image.
    src (numpy.ndarray): The source points.
    result (numpy.ndarray): The result image.
    h (int): The height of the image.
    w (int): The width of the image.

    Returns:
    numpy.ndarray: The result image.
    """
    h, w = img.shape[:2]
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped