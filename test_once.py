"""
单次测试
"""
import task
import cv2

imgpath = "./assets/img/good1.jpg"

if __name__ == "__main__":
    img = cv2.imread(imgpath)
    scanner = task.answerSheetScanner(img, verbose=True)
    try:
        scanner.process()
    except Exception as e:
        print(e)
        print("Failed to process the image.")
        cv2.waitKey(0)
        raise e

    print("Quit.")