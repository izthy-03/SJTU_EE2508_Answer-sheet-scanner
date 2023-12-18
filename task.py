from modules.Initialize import initialize
from modules.Rectify import rectify
# from modules.Hough import  *
import cv2

class answerSheetScanner:
    def __init__(self, img, verbose=False) -> None:
        self.img = img
        self.verbose = verbose

    def process(self):
        tmp1 = initialize(self.img, self.verbose)
        tmp2 = rectify(tmp1, self.verbose)
        

        