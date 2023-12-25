import cv2
import task
import threading

rtsp_url = "rtsp://admin:123456@10.180.249.36:8554/live"
cap = cv2.VideoCapture(rtsp_url)
# cap = cv2.VideoCapture(0)


if __name__ == "__main__":
    buffer = task.frameBuffer(rtsp_url)
    buffer_thread = threading.Thread(target=buffer.stream_on)
    buffer_thread.start()
    while buffer.get_newest_frame() is None:
        continue

    while True:
        # _, img = cap.read()
        img = buffer.get_newest_frame()
        cv2.namedWindow('Camera', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("Camera", img)
        scanner = task.answerSheetScanner(img, verbose=True)
        try:
            scanner.process()
            break
        except Exception as e:
            # print(e)
            # print("Failed to process the image.")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit.")
                break
            continue

    buffer.turn_off()
    cv2.destroyAllWindows()