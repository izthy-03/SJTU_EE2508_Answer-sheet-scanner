import cv2
from MACROS import *

def Initialize(img, verbose=False):
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

    img = resize(img)
    img = adjust_contrast(img, ADJUST_CONTRAST_ALPHA, ADJUST_CONTRAST_BETA)

    if verbose:
        cv2.imshow("Adjusted", img)
    return img



def resize(img):
    """
    Resize the input image if its height is greater than the RESIZE_THRESHOLD_PIXEL.

    Parameters:
    img (numpy.ndarray): The input image to be resized.

    Returns:
    numpy.ndarray: The resized image.
    """
    if img.shape[0] > RESIZE_THRESHOLD_PIXEL:
        img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    return img

def adjust_contrast(img, alpha=1.0, beta=0):
    """
    Adjusts the size, contrast, and brightness of an image.
    
    :param img: The original image.
    :param alpha: The contrast control parameter. The higher the value, the higher the contrast. Default is 1.0.
    :param beta: The brightness control parameter. The higher the value, the higher the brightness. Default is 0.
    :return: The adjusted image.
    """
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted