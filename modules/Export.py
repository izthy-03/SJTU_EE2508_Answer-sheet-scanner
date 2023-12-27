import cv2
import numpy as np
import pandas
from collections import Counter

from utils import *
from MACROS import *


std_path = "./assets/standard.csv"
result_path = "./results/result.csv"

def export(sheet:sheetStats, verbose=False):
    std = pandas.read_csv(std_path, header=None).values
    std = std[1:, :][0]
    # print(std)

    for i in range(0, EXAM_NUMBER_DIGIT):
        sheet.number += sheet.exam_number.get(i, "?")

    for i in range(0, QUESTION_NUMBER):
        if sheet.answers.get(i) is None:
            continue
        # print(i + 1, sheet.answers[i])

        if Counter(sheet.answers[i]) == Counter(std[i]):
            sheet.results[i] = 1
        else:
            sheet.results[i] = 0
    sheet.score = SCORE_PER_QUESTION * np.sum(sheet.results)

    print(sheet.number, sheet.subject, sheet.score, "pts")
    
    return sheet