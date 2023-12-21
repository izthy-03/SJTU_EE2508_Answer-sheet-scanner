import cv2
import numpy as np
import pandas

from utils import *
from MACROS import *


format = pandas.read_csv("./assets/format.csv", header=None).values

def parse(img, sheet: sheetStats, verbose=False):
    locate_y_seq = sheet.locate_centers[1:, 1]
    print(locate_y_seq)