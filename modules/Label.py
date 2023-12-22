import cv2
import numpy as np

from utils import *
from MACROS import *


def label(img, sheet: sheetStats, verbose=False):
    num_ans, label_ans, stats_ans, centers_ans = cv2.connectedComponentsWithStats(sheet.answer_bin, connectivity=8)
    num_info, label_info, stats_info, centers_info = cv2.connectedComponentsWithStats(sheet.info_bin, connectivity=8)
    num_locate, label_locate, stats_locate, centers_locate = cv2.connectedComponentsWithStats(sheet.locate_bin, connectivity=8)

    sheet.info_centers = centers_info
    sheet.answer_centers = centers_ans
    sheet.locate_centers = centers_locate

    if verbose:
        temp = np.copy(img)
        for i in range(1, num_ans):
            cv2.circle(temp, (int(centers_ans[i][0]), int(centers_ans[i][1])), 2, (0, 255, 0), 2)
            cv2.putText(temp, str(i), (int(centers_ans[i][0]), int(centers_ans[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for i in range(1, num_info):
            cv2.circle(temp, (int(centers_info[i][0]), int(centers_info[i][1])), 2, (0, 128, 255), 2)
            cv2.putText(temp, str(i), (int(centers_info[i][0]), int(centers_info[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
        for i in range(1, num_locate):
            cv2.circle(temp, (int(centers_locate[i][0]), int(centers_locate[i][1])), 2, (255, 255, 51), 2)
            cv2.putText(temp, str(i), (int(centers_locate[i][0]), int(centers_locate[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 51), 2)
        cv2.namedWindow("Label centers", cv2.WINDOW_NORMAL)
        cv2.imshow("Label centers", temp)

    # print(num_locate)
    if (num_locate - 1) != LOCATE_CONTOUR_NUMBER:
        raise InvalidLocateContourNumberError(num_locate)

    return sheet