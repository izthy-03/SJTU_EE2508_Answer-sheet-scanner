import numpy as np


RESIZE_THRESHOLD_PIXEL = 2000
RESIZE_SCALE = 0.3
ADJUST_CONTRAST_ALPHA = 1.4

ADJUST_CONTRAST_BETA = 0

GAUSSIAN_KERNEL_HSIZE = 3
GAUSSIAN_KERNEL_SIGMA = 0
GAUSSIAN_KERNEL = (GAUSSIAN_KERNEL_HSIZE, GAUSSIAN_KERNEL_HSIZE), GAUSSIAN_KERNEL_SIGMA

MORPH_KERNEL_SIZE = 2
MORPH_KERNEL = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)

LOCATE_LINE_DUTY_THRESH = 0.5
LOCATE_CONTOURS_NUMBER = 46 + 1

SEGMENT_LOCATE_LINE_THRESH = 0.2
SEGMENT_REGION_LINE_THRESH = 1


class sheetStats:
    def __init__(self) -> None:
        self.locate_line = None
        self.region_line_h1 = None
        self.region_line_h2 = None
        self.region_line_v1 = None
        self.region_line_v2 = None

        self.info_bin = None
        self.answer_bin = None 
        self.locate_bin = None

        self.info_centers = None
        self.answer_centers = None
        self.locate_centers = None

class InvalidContourError(Exception):
    def __str__(self):
        return "Invalid or no contour detected."
    
class InvalidLineError(Exception):
    def __str__(self):
        return "Invalid or no line detected."
    
class InvalidBoundaryError(Exception):
    def __str__(self):
        return "Invalid boundary."
    
class InvalidLocateContourNumberError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.num = args[0]

    def __str__(self):
        return "Invalid number of locate contours: %d" % self.num