import cv2
import numpy as np

from modules.Hough import hough_longest_line
from utils import *
from MACROS import *


def rectify(img, verbose=False):

    original = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = sobel_binarize(gray)
    edge = edge_detection(binary)

    contour_flag = False

    # 尝试寻找最大的矩形并进行透视变换
    try:
        # Find the largest contour
        contour = get_largest_contour(edge)
        if contour is None:
            raise InvalidContourError
        temp = np.copy(img)
        cv2.drawContours(temp, [contour], 0, (0, 255, 0), 1)
        if verbose:
            cv2.imshow("Contour", temp)

        poly_node_list = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.1, True)
        if poly_node_list.shape[0] != 4:
            print("Cannot find the largest rectangle in the image.")
            raise InvalidContourError
        
        poly_node_list = np.float32(sort_rect_nodes(poly_node_list))    

        # transform the contour region to a rectangle
        warped = perspective_transform(original, poly_node_list)
        img = warped
        contour_flag = True

    except Exception as e:
        print(e)

    # 尝试寻找右侧定位线并进行旋转变换
    try:
        if not contour_flag:
            line = hough_longest_line(img, edge=edge, verbose=False)
        else:
            line = hough_longest_line(img, verbose=False)
        print(line)

        angle = line_angle(line) * 180 / np.pi
        print(angle)

        # TODO: 旋转角度的计算有问题，需要图像中心到直线的的垂向量来进行修正
        # TODO: 旋转后的图像尺寸不对，且超出原尺寸的部分被裁剪掉了
        rotated = rotate_transform(img, 90 + angle)
        img = rotated

    except Exception as e:
        print(e)

    if verbose:
        # ***可缩放窗口，但是再次运行会保留上次的窗口尺寸
        # cv2.namedWindow("Rectified", cv2.WINDOW_NORMAL)
        cv2.imshow("Rectified", img)
        cv2.imshow("Edge", edge)

    return img

def get_largest_contour(edge):
    """
    Get the largest contour in the given image.

    Parameters:
    edge (numpy.ndarray): The input edge image.

    Returns:
    numpy.ndarray: The largest contour in the image.
    """
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped

def rotate_transform(img, angle, center=None):
    """
    Apply rotation transform to the input image.

    Parameters:
    img (numpy.ndarray): The input image.
    angle (float): The angle to rotate.

    Returns:
    numpy.ndarray: The result image.
    """
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

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
    if nodes[2][0] > nodes[3][0]:   
        nodes[[2, 3]] = nodes[[3, 2]]

    return nodes