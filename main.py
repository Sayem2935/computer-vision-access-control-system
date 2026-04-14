import argparse
import time

import cv2

from camera import CameraStream, DEFAULT_RTSP_URL, mask_url_password, resize_frame
from database import DB_PATH, PeopleDatabase
from recognition import (
    create_person_from_encoding,
    extract_face_encoding,
    get_face_image_path,
    get_face_location,
    match_face_encoding,
    remember_person_encoding,
)
from tracker import PersonTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect, track, recognize, and count people from an RTSP stream."
    )
    parser.add_argument("--rtsp", default=DEFAULT_RTSP_URL, help="RTSP stream URL.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path/name.")
    parser.add_argument("--width", type=int, default=960, help="Resize frame width.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--tracker", default=None, help="Reserved for tracker config compatibility.")
    parser.add_argument("--device", default=None, help="Inference device: cpu, mps, 0, etc.")
    parser.add_argument(
        "--max-disappeared",
        type=int,
        default=20,
        help="Frames to keep a missing track before marking it as exited.",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=80,
        help="Maximum centroid distance for matching detections to existing tracks.",
    )
    parser.add_argument("--rtsp-transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--open-timeout-ms", type=int, default=8000)
    parser.add_argument("--read-timeout-ms", type=int, default=8000)
    parser.add_argument("--face-scale", type=float, default=0.5)
    parser.add_argument("--face-model", choices=("hog", "cnn"), default="hog")
    parser.add_argument("--face-tolerance", type=float, default=0.47)
    parser.add_argument(
        "--recognition-interval",
        type=int,
        default=5,
        help="Run face recognition every N frames to reduce CPU load.",
    )
    parser.add_argument(
        "--no-face-recognition",
        action="store_true",
        help="Disable face recognition and use tracking IDs only.",
    )
    parser.add_argument("--database", default=DB_PATH, help="SQLite database path.")
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable per-frame debug prints.",
    )
    return parser.parse_args()


def draw_person_track(frame, track, name=None):
    x1, y1, x2, y2 = track.box
    identity = name or "Unknown"
    label = f"{identity} {track.confidence:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 190, 255), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 190, 255),
        2,
        cv2.LINE_AA,
    )


def draw_face_boxes(frame, faces):
    for face in faces:
        x1, y1, x2, y2 = face.box
        if frame[y1:y2, x1:x2].size == 0:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 80), 2)
        cv2.putText(
            frame,
            face.name,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 80, 80),
            2,
            cv2.LINE_AA,
        )


def draw_face_identity(frame, face_box, person_id):
    if face_box is None or person_id is None:
        return

    x1, y1, x2, y2 = face_box
    if frame[y1:y2, x1:x2].size == 0:
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 80), 2)
    cv2.putText(
        frame,
        person_id,
        (x1, max(y1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 80, 80),
        2,
        cv2.LINE_AA,
    )


def draw_status(frame, fps, inside_count):
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (40, 255, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"INSIDE: {inside_count}",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (40, 255, 40),
        2,
        cv2.LINE_AA,
    )


def update_fps(previous_time, smoothed_fps):
    now = time.perf_counter()
    instant_fps = 1.0 / max(now - previous_time, 1e-6)
    fps = instant_fps if smoothed_fps == 0.0 else (smoothed_fps * 0.9) + (instant_fps * 0.1)
    return now, fps


def print_track_events(entered_track_ids, exited_track_ids):
    return


class PeopleTrackingApp:
    def __init__(self, args):
        self.args = args
        self.camera = CameraStream(
            rtsp_url=args.rtsp,
            rtsp_transport=args.rtsp_transport,
            open_timeout_ms=args.open_timeout_ms,
            read_timeout_ms=args.read_timeout_ms,
        )
        self.tracker = PersonTracker(
            model_path=args.model,
            confidence=args.conf,
            tracker_config=args.tracker,
            device=args.device,
            max_disappeared=args.max_disappeared,
            max_distance=args.max_distance,
        )
        self.database = PeopleDatabase(args.database)
        self.inside_ids = set()
        self.track_to_person = {}
        self.track_names = {}
        self.person_records = {}
        self.pending_track_confirmations = {}
        self.debug_enabled = not args.no_debug
        self.window_name = "YOLOv8 RTSP Person Tracker"
        self.fps = 0.0
        self.frame_count = 0
        self.previous_time = time.perf_counter()

    def run(self):
        self._open_camera()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        try:
            try:
                while True:
                    if not self._process_frame():
                        break
            except KeyboardInterrupt:
                print("Stopping system safely")
        finally:
            self.camera.release()
            self.database.close()
            cv2.destroyAllWindows()

    def _open_camera(self):
        if self.camera.open():
            return

        masked_url = mask_url_password(self.args.rtsp)
        raise RuntimeError(
            "Could not open RTSP stream. Check that the camera is reachable, "
            f"RTSP is enabled, and the URL is correct: {masked_url}"
        )

    def _process_frame(self):
        ok, frame = self.camera.read()
        if not ok:
            print("Frame read failed. Reconnecting...")
            return True

        frame = resize_frame(frame, self.args.width)
        self.frame_count += 1
        should_run_recognition = self.frame_count % self.args.recognition_interval == 0
        tracking_update = self.tracker.update(frame)
        tracks = tracking_update.tracks
        for track in tracks:
            person_id, face_box = self._identify_person(
                track,
                frame,
                should_run_recognition,
            )
            if person_id is not None:
                self._register_person_entry(person_id)
            draw_person_track(frame, track, name=self.track_names.get(track.track_id))
            draw_face_identity(frame, face_box, person_id)

        self.previous_time, self.fps = update_fps(self.previous_time, self.fps)
        draw_status(frame, self.fps, len(self.inside_ids))

        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF not in (ord("q"), 27)

    def _identify_person(self, track, frame, should_run_recognition):
        if track.track_id in self.track_to_person:
            person_id = self.track_to_person[track.track_id]
            self.track_names[track.track_id] = person_id
            return person_id, None

        if self.args.no_face_recognition or not should_run_recognition:
            return None, None

        face_crop, crop_origin = crop_track_region(frame, track.box)
        encoding, _, normalized_face = extract_face_encoding(face_crop)
        if encoding is None:
            self.pending_track_confirmations.pop(track.track_id, None)
            return None, None

        face_box = get_absolute_face_box(face_crop, crop_origin)
        person_id, _ = match_face_encoding(encoding, threshold=self.args.face_tolerance)
        if person_id is not None:
            remember_person_encoding(person_id, encoding, face_crop=normalized_face)
            self.pending_track_confirmations.pop(track.track_id, None)
            self._assign_track_person(
                track.track_id,
                person_id,
                encoding,
                get_face_image_path(person_id),
            )
            return person_id, face_box

        confirmation_count = self.pending_track_confirmations.get(track.track_id, 0) + 1
        self.pending_track_confirmations[track.track_id] = confirmation_count
        if confirmation_count < 3:
            return None, face_box

        self.pending_track_confirmations.pop(track.track_id, None)
        person_id = create_person_from_encoding(encoding, face_crop=normalized_face)
        self._assign_track_person(
            track.track_id,
            person_id,
            encoding,
            get_face_image_path(person_id),
        )
        return person_id, face_box

    def _assign_track_person(self, track_id, person_id, encoding, image_path):
        self.track_to_person[track_id] = person_id
        self.track_names[track_id] = person_id
        self.person_records[person_id] = {
            "encoding": encoding,
            "image_path": image_path,
        }

    def _register_person_entry(self, person_id):
        if person_id in self.inside_ids:
            return

        print(f"ENTRY: {person_id}")
        self.inside_ids.add(person_id)
        person_record = self.person_records.get(person_id, {})
        self.database.add_person(
            person_id,
            encoding=person_record.get("encoding"),
            image_path=person_record.get("image_path"),
            print_log=False,
        )
        print(f"INSIDE COUNT: {len(self.inside_ids)}")


def crop_track_region(frame, box):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))

    if x2 <= x1 or y2 <= y1:
        return None, None

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None

    return face_crop, (x1, y1)


def get_absolute_face_box(face_crop, crop_origin):
    local_face_box = get_face_location(face_crop)
    if local_face_box is None or crop_origin is None:
        return None

    left, top, right, bottom = local_face_box
    origin_x, origin_y = crop_origin
    padding = 20

    x = origin_x + left
    y = origin_y + top
    w = right - left
    h = bottom - top

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = x + w + padding
    y2 = y + h + padding

    return (
        x1,
        y1,
        x2,
        y2,
    )


def main():
    args = parse_args()
    app = PeopleTrackingApp(args)
    app.run()


if __name__ == "__main__":
    main()
