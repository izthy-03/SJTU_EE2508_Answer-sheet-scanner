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

SEGMENT_LOCATE_LINE_THRESH = 0.2
SEGMENT_REGION_LINE_THRESH = 1

class InvalidContourError(Exception):
    def __str__(self):
        return "Invalid or no contour detected."
    
class InvalidLineError(Exception):
    def __str__(self):
        return "Invalid or no line detected."