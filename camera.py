import os
import time
from urllib.parse import urlsplit, urlunsplit

import cv2


USE_HD = False
DEFAULT_RTSP_URL = "rtsp://admin:Shuborno%40500@192.168.0.200:554/Streaming/Channels/102"


def configure_rtsp_options(rtsp_transport, open_timeout_ms, read_timeout_ms):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{rtsp_transport}|"
        f"stimeout;{open_timeout_ms * 1000}|"
        f"timeout;{read_timeout_ms * 1000}"
    )


def mask_url_password(url):
    parsed = urlsplit(url)
    if parsed.password is None:
        return url

    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"

    username = parsed.username or ""
    netloc = f"{username}:***@{host}" if username else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def resize_frame(frame, target_width):
    if target_width <= 0 or frame.shape[1] <= target_width:
        return frame

    scale = target_width / frame.shape[1]
    target_height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


class CameraStream:
    def __init__(
        self,
        rtsp_url,
        rtsp_transport="tcp",
        open_timeout_ms=8000,
        read_timeout_ms=8000,
        reconnect_delay=1.0,
    ):
        self.rtsp_url = rtsp_url
        self.rtsp_transport = rtsp_transport
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.reconnect_delay = reconnect_delay
        self.capture = None

    def open(self):
        configure_rtsp_options(
            self.rtsp_transport,
            self.open_timeout_ms,
            self.read_timeout_ms,
        )

        rtsp_url = self.rtsp_url
        capture = cv2.VideoCapture(rtsp_url)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms)
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms)
        if USE_HD:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        else:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Current Resolution: {width} x {height}")

        if not capture.isOpened():
            print("Error: Cannot connect to camera")

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture = capture
        return self.is_opened

    @property
    def is_opened(self):
        return self.capture is not None and self.capture.isOpened()

    def read(self):
        if not self.is_opened:
            self.open()

        ok, frame = self.capture.read()
        if ok:
            return True, frame

        self.reconnect()
        return False, None

    def reconnect(self):
        self.release()
        time.sleep(self.reconnect_delay)
        self.open()

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
