import cv2


CAMERA_1_RTSP = "rtsp://admin:Shuborno%40500@192.168.0.200:554/Streaming/Channels/102"
CAMERA_2_RTSP = "rtsp://admin:Shuborno%40500@192.168.0.201:554/Streaming/Channels/102"
FRAME_SIZE = (640, 480)


def open_camera(rtsp_url):
    return cv2.VideoCapture(rtsp_url)


def read_and_resize_frame(capture):
    ok, frame = capture.read()
    if not ok or frame is None:
        return False, None

    frame = cv2.resize(frame, FRAME_SIZE)
    return True, frame


def main():
    cap1 = open_camera(CAMERA_1_RTSP)
    cap2 = open_camera(CAMERA_2_RTSP)

    if not cap1.isOpened():
        print("Camera 1 not connected")

    if not cap2.isOpened():
        print("Camera 2 not connected")

    try:
        while True:
            ok1, frame1 = read_and_resize_frame(cap1)
            ok2, frame2 = read_and_resize_frame(cap2)

            if ok1:
                cv2.imshow("Camera 1 - Entry", frame1)

            if ok2:
                cv2.imshow("Camera 2 - Exit", frame2)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
