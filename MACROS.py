RESIZE_THRESHOLD_PIXEL = 2000

ADJUST_CONTRAST_ALPHA = 1.8

ADJUST_CONTRAST_BETA = 0

GAUSSIAN_KERNEL_HSIZE = 3
GAUSSIAN_KERNEL_SIGMA = 0.5

class InvalidContourError(Exception):
    def __str__(self):
        return "Invalid or no contour detected."
