from modules.Initialize import initialize
from modules.Rectify import rectify
from modules.Segment import segment
from modules.Label import label
from modules.Parse import parse
from modules.Export import export
from MACROS import *
import cv2
import threading

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
        self.sheet = export(self.sheet, verbose=self.verbose)

        # if self.verbose:
        cv2.waitKey(0)


class frameBuffer():
    def __init__(self, video_src) -> None:
        self.mutex = threading.Lock()
        self.src = video_src
        self.enable = False
        self.buffer = None

    def stream_on(self):
        self.cap = cv2.VideoCapture(self.src)
        self.enable = True
        while self.enable:
            _, img = self.cap.read()
            # cv2.namedWindow('Camera', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            # cv2.imshow("Camera", img)
            self.mutex.acquire()
            self.buffer = img
            self.mutex.release()
        self.stream_off()

    def get_newest_frame(self):
        self.mutex.acquire()
        img = self.buffer
        self.mutex.release()
        return img

    def turn_off(self):
        self.enable = False

    def stream_off(self):
        self.cap.release()