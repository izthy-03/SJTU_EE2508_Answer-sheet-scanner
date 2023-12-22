import cv2
import numpy as np
import pandas

from utils import *
from MACROS import *


format = pandas.read_csv("./assets/format.csv", header=None).values
row_dict = format[:, 23]
# print(row_dict)

def parse(img, sheet: sheetStats, verbose=False):
    locate_y_seq = np.sort(sheet.locate_centers[1:, 1])
    locate_x_seq = np.linspace(sheet.region_line_v1[0], sheet.region_line_v2[0], COLUMN_NUMBER + 1)
    locate_x_gap = (locate_x_seq[-1] - locate_x_seq[0]) / COLUMN_NUMBER
    locate_y_gap = (locate_y_seq[-1] - locate_y_seq[0]) / (LOCATE_CONTOUR_NUMBER - 1)
    locate_y_seq -= locate_y_gap / 2
    # print(locate_x_seq)
    in_boundary = lambda row, column: row in range(0, LOCATE_CONTOUR_NUMBER) and column in range(0, COLUMN_NUMBER)
    meaningless = lambda row, column: format[row, column] == -1

    for centroid in sheet.info_centers[1:]:
        row, column = get_contour_row_column(centroid, locate_x_seq, locate_y_seq)
        # print(row, column)
        if not in_boundary(row, column) or meaningless(row, column):
            continue

        # Save the exam number and subject
        if PARSE_INFO_BASE <= format[row, column] < PARSE_SUBJECT_BASE:
            sheet.exam_number[format[row, column] - PARSE_INFO_BASE] = row_dict[row]
        elif format[row, column] >= PARSE_SUBJECT_BASE:
            sheet.subject = PARSE_SUBJECT[format[row, column] - PARSE_SUBJECT_BASE]

        if verbose:
            cv2.circle(img, (int(centroid[0]), int(centroid[1])), 2, (0, 255, 0), 2)
            cv2.putText(img, str(row_dict[row]), (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    for centroid in sheet.answer_centers[1:]:
        row, column = get_contour_row_column(centroid, locate_x_seq, locate_y_seq)
        # print(row, column)
        if not in_boundary(row, column) or meaningless(row, column):
            continue
        
        # Save the answer
        if PARSE_ANSWER_BASE <= format[row, column] < PARSE_INFO_BASE:
            old = sheet.answers.get(format[row, column] - 1, "")
            sheet.answers[format[row, column] - 1] = old + row_dict[row]
        if verbose:
            cv2.circle(img, (int(centroid[0]), int(centroid[1])), 2, (0, 255, 0), 2)
            cv2.putText(img, str(row_dict[row]), (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if verbose:
        for i in range(len(locate_x_seq)):
            cv2.line(img, (int(locate_x_seq[i]), 0), (int(locate_x_seq[i]), img.shape[0]), (0, 255, 255), 2)
        for i in range(len(locate_y_seq)):
            cv2.line(img, (0, int(locate_y_seq[i])), (img.shape[1], int(locate_y_seq[i])), (0, 255, 255), 2)
        cv2.namedWindow("locate x", cv2.WINDOW_NORMAL)
        cv2.imshow("locate x", img)
    
    return sheet


def get_contour_row_column(centroid, x_seq, y_seq):
    cx, cy = centroid
    x_gap = (x_seq[-1] - x_seq[0]) / COLUMN_NUMBER
    y_gap = (y_seq[-1] - y_seq[0]) / (LOCATE_CONTOUR_NUMBER - 1)
    # y_seq -= y_gap / 2

    row = (cy - y_seq[0]) // y_gap
    column = (cx - x_seq[0]) // x_gap

    return int(row), int(column)