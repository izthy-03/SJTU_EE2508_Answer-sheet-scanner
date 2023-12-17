from modules.Initialize import *
from modules.Rectify import *
import cv2

class AnswerSheetScanner:
    def __init__(self, img, verbose=False):
        self.img = img
        self.verbose = verbose

    def process(self):
        I1 = Initialize(self.img, self.verbose)
        I2 = Rectify(I1, self.verbose)
        
        if self.verbose:
            cv2.imshow("Rectified", I2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        