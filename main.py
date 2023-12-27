import cv2
import task
import threading


# Video source
video_src = "rtsp://admin:123456@10.180.249.36:8554/live"
# video_src = 0


if __name__ == "__main__":
    # start the buffer thread to get the newest frame
    buffer = task.frameBuffer(video_src)
    buffer_thread = threading.Thread(target=buffer.stream_on)
    buffer_thread.start()
    # spin wait until the buffer is ready
    while buffer.get_newest_frame() is None:
        continue

    print("Press q to stop scanning.")

    while True:
        # _, img = cap.read()
        img = buffer.get_newest_frame()
        scanner = task.answerSheetScanner(img, verbose=False)
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