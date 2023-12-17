import task
import cv2

imgpath = "./img/1.jpg"

if __name__ == "__main__":
    img = cv2.imread(imgpath)
    scanner = task.AnswerSheetScanner(img, verbose=True)
    scanner.process()