import cv2
from MACROS import *

def initialize(img, verbose=False):
    """
    Initialize the image.

    Parameters:
    img (numpy.ndarray): The input image to be initialized.
    verbose (bool): Whether to show the image or not.

    Returns:
    numpy.ndarray: The initialized image.
    """
    # if verbose:
    #     cv2.imshow("Original", img)

    img = resize(img, RESIZE_THRESHOLD_PIXEL)
    img = adjust_contrast(img, ADJUST_CONTRAST_ALPHA, ADJUST_CONTRAST_BETA)
    img = smooth(img, GAUSSIAN_KERNEL_HSIZE, GAUSSIAN_KERNEL_SIGMA)

    if verbose:
        cv2.imshow("Adjusted", img)
        # cv2.waitKey(0)

    return img



def resize(img, threshold):
    if img.shape[0] > threshold:
        # scale = min(threshold / img.shape[0], threshold / img.shape[1])
        scale = RESIZE_SCALE
        img = cv2.resize(img, None, fx=scale, fy=scale, 
                         interpolation=cv2.INTER_LINEAR)
        print("Resized to {}x{}".format(img.shape[0], img.shape[1]))
    return img


def adjust_contrast(img, alpha=1.0, beta=0):
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def smooth(img, hsize, sigma):
    img = cv2.GaussianBlur(img, (hsize, hsize), sigma)
    return img