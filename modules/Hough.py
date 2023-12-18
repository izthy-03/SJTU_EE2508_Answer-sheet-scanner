import cv2
import numpy as np
import imutils as im

class houghProcessor:
    def __init__(self, img, binary=None) -> None:
        self.img = img
        self.binary = binary if binary is not None else self.binarize()
        self.edge = self.edge_detection()
    
    def binarize(self):
        """
        Binarize the input image.
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return binary
    

    def edge_detection(self):
        """
        Detect edges of the input image.
        """
        kernel = np.ones((1, 1), np.uint8)
        temp = np.copy(self.img)
        temp = cv2.erode(temp, kernel, iterations=1)
        temp = cv2.dilate(temp, kernel, iterations=2)
        temp = cv2.erode(temp, kernel, iterations=1)
        temp = cv2.dilate(temp, kernel, iterations=2)
        temp = im.auto_canny(temp)
        return temp


    def process(self):
        # TODO
        return None