import cv2
import numpy as np
import imutils as im

from modules.Hough import hough_longest_line
from utils import *
from MACROS import *

def segment(img, verbose=False):
    
    original = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = edge_detection(sobel_binarize(gray))

    bin = binarize_and_enhence(gray)
    locate_line = hough_longest_line(img, edge)
    print(locate_line)
    bin = filter_locate_line(bin, locate_line)
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=3)

    # edge = cv2.Canny(bin, 0, 255)
    edge = im.auto_canny(bin)
    x1, y1, x2, y2 = locate_line[0]
    y_mid = (y1 + y2) // 2
    len = line_length(locate_line)
    check = lambda line: line[0][1] < y_mid and line[0][3] < y_mid 

    region_line = hough_longest_line(img, bin, constrain=check, verbose=verbose)
    print(region_line)
    if verbose:
        cv2.imshow("Binary", bin)


def binarize_and_enhence(gray):
    # Convert to binary image with OTSU method
    gray = cv2.convertScaleAbs(gray, alpha=1.4, beta=0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=2)
    _, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    bin = ~bin

    # Morph open operation to remove noise
    kernel = np.ones((5, 5), np.uint8)
    bin = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)
    return bin


def filter_locate_line(bin, locate_line):
    x1, _, x2, _ = locate_line[0]
    column_scan = (x1 + x2) // 2
    # Scan from the locate line to the left
    locate_line_duty = np.sum(bin[:, column_scan])
    while column_scan > 0:
        if np.sum(bin[:, column_scan]) < locate_line_duty * SEGMENT_LOCATE_LINE_THRESH:
            break
        column_scan -= 1

    # print(column_scan)
    # Filter the locate line
    bin[:, column_scan:] = 0
    return bin