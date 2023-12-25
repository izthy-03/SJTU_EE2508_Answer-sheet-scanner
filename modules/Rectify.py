import cv2
import numpy as np

from modules.Hough import hough_longest_line
from utils import *
from MACROS import *


def rectify(img, verbose=False) -> sheetStats:

    original = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = sobel_binarize(gray)
    edge = edge_detection(binary)

    if verbose:
        cv2.imshow("edge to find a4", edge)

    contour_flag = False

    # 尝试寻找最大的矩形并进行透视变换
    try:
        # Find the largest contour
        # TODO: Check contour validity
        contour = get_largest_contour(edge)
        if contour is None or cv2.contourArea(contour) / (img.shape[0] * img.shape[1]) < CONTOUR_AREA_THRESH:
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
        print("Contour found") if verbose else None
        
        if verbose:
            cv2.imshow("Perspective transform", warped)

    except Exception as e:
        print(e) if verbose else None

    # 尝试寻找右侧定位线并进行旋转变换
    # Locate line constraint
    check = lambda line: line_length(line) > img.shape[0] * LOCATE_LINE_DUTY_THRESH

    if not contour_flag:
        line = hough_longest_line(img, edge=edge, constrain=check,verbose=verbose)
    else:
        line = hough_longest_line(img, constrain=check, verbose=verbose)

    print("Locate line:", line) if verbose else None

    if line_length(line) == 0:
        raise InvalidLineError

    # TODO: 旋转角度的计算有问题，需要图像中心到直线的的垂向量来进行修正
    # TODO: 旋转后的图像尺寸不对，且超出原尺寸的部分被裁剪掉了
    img_center = np.array([img.shape[1] // 2, img.shape[0] // 2])
    angle = get_rotate_angle(img_center, line[0])
    if contour_flag:
        candidate = [0, 90, -90, 180, -180]
        angle = min(candidate, key=lambda x: abs(x - angle))

    print("      angle:", angle) if verbose else None
    # rotated = rotate_transform(img, -angle)
    rotated = dumpRotateImage(img, angle)
    img = rotated


    if verbose:
        # ***可缩放窗口，但是再次运行会保留上次的窗口尺寸
        # cv2.namedWindow("Rectified", cv2.WINDOW_NORMAL)
        temp = np.copy(original)
        x1, y1, x2, y2 = line[0]
        cv2.line(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Locate line", temp)
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
    # contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
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
    # TODO: 等比例缩放
    h0, w0 = src.shape[:2]
    h, w = img.shape[:2]
    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped

def rotate_transform(img, angle, center=None, dst=None):
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
    if dst is None:
        dst = (w, h)
    rotated = cv2.warpAffine(img, M, dst)
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

def get_rotate_angle(img_center, line):
    #确定直线
    line_point1 = np.array([line[0], line[1]])  # 直线上的点1
    line_point2 = np.array([line[2], line[3]])  # 直线上的点2

    # 计算直线的向量
    line_vector = line_point2 - line_point1

    # 计算点到直线起点的向量
    point_vector = line_point1 - img_center

    # 计算点到直线的投影长度
    projection_length = np.abs(np.dot(point_vector, line_vector) / np.linalg.norm(line_vector))

    # 计算点到直线的法向量
    normal_vector = point_vector + projection_length * line_vector / np.linalg.norm(line_vector)
    # print(normal_vector) 

    # 计算向量的模长
    magnitude = np.linalg.norm(normal_vector)

    # 计算向量与水平线的夹角（弧度）
    # angle_radians = np.arccos(np.dot(normal_vector, np.array([1, 0])) / magnitude)
    angle_radians = np.angle(normal_vector[0] + normal_vector[1] * 1j)

    # 将弧度转换为角度
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def dumpRotateImage(img, degree):
    height, width = img.shape[:2]
    heightNew = int(width * np.fabs(np.sin(np.radians(degree))) + height * np.fabs(np.cos(np.radians(degree))))
    widthNew = int(height * np.fabs(np.sin(np.radians(degree))) + width * np.fabs(np.cos(np.radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    # 加入平移操作
    widthNew = int(widthNew)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation
 
