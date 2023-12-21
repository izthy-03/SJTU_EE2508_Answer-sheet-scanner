import cv2
import numpy as np
import imutils as im

from modules.Hough import hough_longest_line
from utils import *
from MACROS import *


def label(img, sheetStats, verbose=False):
    pass