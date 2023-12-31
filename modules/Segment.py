import cv2
import numpy as np
import imutils as im

from modules.Hough import hough_longest_line
from utils import *
from MACROS import *

def segment(img, verbose=False):
    
    sheet = sheetStats()

    original = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edge = edge_detection(sobel_binarize(gray))
    edge = edge_detection(gray)

    bin = binarize_and_enhence(gray)
    locate_line = hough_longest_line(img, edge)
    print("Locate line", locate_line) if verbose else None
    sheet.locate_line = locate_line

    bin, locate_bin = filter_locate_line(bin, locate_line)
    sheet.locate_bin = locate_bin
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=3)


    # edge = cv2.Canny(bin, 0, 255)
    edge = im.auto_canny(bin)
    _, y1, _, y2 = locate_line
    y_mid = (y1 + y2) // 2
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    len = line_length(locate_line)

    # Constraints for horizontal region line
    check1 = lambda line: y_min < line[1] < y_mid and y_min < line[3] < y_mid and \
                            np.abs(line_angle(line)) < 0.1 and line_length(line) > len * 0.5
    check2 = lambda line: y_mid < line[1] < y_max + 10 and y_mid < line[3] < y_max + 10 and \
                            np.abs(line_angle(line)) < 0.1 and line_length(line) > len * 0.5

    # Find the horizontal region lines
    region_line_h1 = hough_longest_line(img, bin, constrain=check1, maxLineGap=20, verbose=verbose)
    region_line_h2 = hough_longest_line(img, bin, constrain=check2, maxLineGap=20, verbose=verbose)
    print("Horizontal region lines:", region_line_h1, region_line_h2) if verbose else None

    if region_line_h1 is None and region_line_h2 is None:
        raise InvalidLineError
    
    region_line_h1 = region_line_h2 if region_line_h1 is None else region_line_h1
    region_line_h2 = region_line_h1 if region_line_h2 is None else region_line_h2

    sheet.region_line_h1 = region_line_h1
    sheet.region_line_h2 = region_line_h2

    # Find the vertical region lines
    l, r = find_vertical_boundary(bin, (region_line_h1, region_line_h2))
    if l < 0 or r > bin.shape[1]:
        l = 0; r = bin.shape[1]
        # raise InvalidBoundaryError
        
    sheet.region_line_v1 = np.array([l, 0, l, bin.shape[0]])
    sheet.region_line_v2 = np.array([r, 0, r, bin.shape[0]])

    # Segment the binary image into info and answer regions
    info_bin = np.copy(bin)
    answer_bin = np.copy(bin)
    info_bin[region_line_h1[1] - 8:, :] = 0
    info_bin[:min(locate_line[1], locate_line[3]), :] = 0
    answer_bin[:region_line_h1[1] + 8, :] = 0
    answer_bin[region_line_h2[1] - 8:, :] = 0
    sheet.info_bin = info_bin
    sheet.answer_bin = answer_bin

    if verbose:
        temp = np.copy(original)
        cv2.line(temp, (l, 0), (l, bin.shape[0]), (0, 255, 0), 2)
        cv2.line(temp, (r, 0), (r, bin.shape[0]), (0, 255, 0), 2)
        x1, y1, x2, y2 = region_line_h1
        cv2.line(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1, y1, x2, y2 = region_line_h2
        cv2.line(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.namedWindow("Region lines", cv2.WINDOW_NORMAL)
        cv2.imshow("Region lines", temp)
        cv2.namedWindow("Info", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Answer", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Locate", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
        cv2.imshow("Binary", bin)
        cv2.imshow("Info", info_bin)
        cv2.imshow("Answer", answer_bin)
        cv2.imshow("Locate", locate_bin)
        
    return sheet


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
    x1, y1, x2, y2 = locate_line
    y1, y2 = np.sort([y1, y2])
    column_scan = (x1 + x2) // 2
    # Scan from the locate line to the left
    locate_line_duty = np.sum(bin[:, column_scan])
    while column_scan > 0:
        if np.sum(bin[:, column_scan]) < locate_line_duty * SEGMENT_LOCATE_LINE_THRESH:
            break
        column_scan -= 1

    # print(column_scan)
    # Filter the locate line
    reverse = np.copy(bin)
    bin[:, column_scan:] = 0

    reverse[:, :column_scan] = 0
    reverse[:y1 - 5, :] = 0
    reverse[y2 + 5:, :] = 0
    column_scan = (x1 + x2) // 2
    while column_scan < bin.shape[1]:
        if np.sum(reverse[:, column_scan]) < locate_line_duty * SEGMENT_LOCATE_LINE_THRESH:
            break
        column_scan += 1
    reverse[:, column_scan:] = 0
    return bin, reverse

def find_vertical_boundary(bin, horizontal_lines):
    x1, y1, x2, y2 = horizontal_lines[0]
    x3, y3, x4, y4 = horizontal_lines[1]
    x1, x2, x3, x4 = np.sort([x1, x2, x3, x4])

    left, right = x1, x4
    y = (y1 + y2) // 2 if np.abs(y1 - y2) < np.abs(y3 - y4) else (y3 + y4) // 2
    print(left, right, y)
    step = 0
    scan_line = lambda x: np.sum(bin[y - 5:y + 5, x]) > 0
    while step < left and scan_line(left - step):
        step += 1

    while step < left and not scan_line(left - step):
        step += 1
    # print("Left", left) 
    # print("Left step:", step)

    return left - step * 2, right + step * 2
    

def fix_region_lines(region_line_h1, region_line_h2):
    pass