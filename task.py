from modules.Initialize import initialize
from modules.Rectify import rectify
from modules.Segment import segment
from modules.Label import label
from modules.Parse import parse
from modules.Export import export
from MACROS import *
import cv2


class answerSheetScanner():
    def __init__(self, img, verbose=False) -> None:
        self.img = img
        self.verbose = verbose
        self.sheet = sheetStats()

    def process(self):
        tmp1 = initialize(self.img, self.verbose)
        tmp2 = rectify(tmp1, self.verbose)
        self.sheet = segment(tmp2, self.verbose)
        self.sheet = label(tmp2, self.sheet, self.verbose)
        self.sheet = parse(tmp2, self.sheet, self.verbose)
        export(self.sheet, verbose=self.verbose)

        if self.verbose:
            cv2.waitKey(0)
        