import argparse
import os
import threading
import time

import cv2
import face_recognition

from api_sender import send_exit_to_api
from camera import DEFAULT_RTSP_URL
from database import DB_PATH, PeopleDatabase
from recognition import extract_face_encoding


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exit system that matches faces against the database and removes them."
    )
    parser.add_argument("--rtsp", default=DEFAULT_RTSP_URL, help="RTSP stream URL.")
    parser.add_argument("--database", default=DB_PATH, help="SQLite database path.")
    parser.add_argument("--width", type=int, default=960, help="Resize frame width.")
    parser.add_argument(
        "--recognition-interval",
        type=int,
        default=3,
        help="Run face recognition every N frames to reduce CPU load.",
    )
    parser.add_argument(
        "--confirmation-frames",
        type=int,
        default=2,
        help="Number of matching frames required before confirming exit.",
    )
    parser.add_argument(
        "--detection-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(480, 270),
        help="Detection frame size used for face search.",
    )
    return parser.parse_args()


def resize_frame(frame, target_width):
    if target_width <= 0 or frame.shape[1] <= target_width:
        return frame

    scale = target_width / frame.shape[1]
    target_height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def crop_face_with_padding(frame, face_location, padding=20):
    top, right, bottom, left = face_location
    x1 = max(0, left - padding)
    y1 = max(0, top - padding)
    x2 = min(frame.shape[1], right + padding)
    y2 = min(frame.shape[0], bottom + padding)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None

    return face_crop, (x1, y1, x2, y2)


def scale_face_location(face_location, from_shape, to_shape):
    from_height, from_width = from_shape
    to_height, to_width = to_shape
    scale_x = to_width / max(from_width, 1)
    scale_y = to_height / max(from_height, 1)
    top, right, bottom, left = face_location
    return (
        int(round(top * scale_y)),
        int(round(right * scale_x)),
        int(round(bottom * scale_y)),
        int(round(left * scale_x)),
    )


def draw_face_box(frame, face_box, label):
    if face_box is None:
        return

    x1, y1, x2, y2 = face_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 80), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 80, 80),
        2,
        cv2.LINE_AA,
    )


def print_summary(total_in, total_out, current_inside):
    print("===== SUMMARY =====")
    print(f"Total IN: {total_in}")
    print(f"Total OUT: {total_out}")
    print(f"Inside: {current_inside}")


def show_matched_person(person_id, person_record):
    if person_record is None:
        return

    image_path = person_record.get("image_path")
    if not image_path:
        return

    image = cv2.imread(image_path)
    if image is None:
        return

    print(f"Matched with {person_id}")
    cv2.imshow("Matched Person", image)
    cv2.waitKey(2000)


def main(rtsp_url=None):
    args = parse_args()
    if rtsp_url is not None:
        args.rtsp = rtsp_url

    database = PeopleDatabase(args.database)
    total_in = len(database.get_inside_people())
    total_out = 0
    current_inside = total_in
    exit_cooldown_seconds = 3
    last_exit_times = {}
    last_unknown_warning_time = 0.0
    exit_confirmation_counts = {}
    frame_count = 0

    capture = cv2.VideoCapture(args.rtsp)
    if not capture.isOpened():
        print("[WARNING] exit camera not connected, retrying...")
        time.sleep(1.0)
        capture.release()
        capture = cv2.VideoCapture(args.rtsp)
        if not capture.isOpened():
            print("[ERROR] Cannot connect to camera")
            return

    window_name = "Exit System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        try:
            while True:
                try:
                    if not capture.isOpened():
                        capture.release()
                        time.sleep(1.0)
                        capture = cv2.VideoCapture(args.rtsp)
                        if not capture.isOpened():
                            print("[WARNING] exit camera reconnect failed")
                            continue

                    ok, frame = capture.read()
                    if not ok or frame is None:
                        print("[WARNING] exit frame read failed")
                        capture.release()
                        time.sleep(1.0)
                        capture = cv2.VideoCapture(args.rtsp)
                        continue

                    original_frame = frame
                    frame_count += 1
                    should_run_recognition = frame_count % args.recognition_interval == 0
                    face_locations = []
                    if should_run_recognition:
                        detection_frame = cv2.resize(
                            original_frame,
                            tuple(args.detection_size),
                            interpolation=cv2.INTER_AREA,
                        )
                        rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                        small_face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                        face_locations = [
                            scale_face_location(
                                face_location,
                                from_shape=detection_frame.shape[:2],
                                to_shape=original_frame.shape[:2],
                            )
                            for face_location in small_face_locations
                        ]

                    for face_location in face_locations:
                        face_crop, face_box = crop_face_with_padding(original_frame, face_location)
                        new_encoding, _, _ = extract_face_encoding(face_crop)
                        if new_encoding is None:
                            continue

                        person_id = database.match_person(new_encoding, threshold=0.47)
                        now = time.time()
                        if person_id is not None:
                            exit_confirmation_counts[person_id] = (
                                exit_confirmation_counts.get(person_id, 0) + 1
                            )
                            if exit_confirmation_counts[person_id] < args.confirmation_frames:
                                draw_face_box(original_frame, face_box, person_id)
                                continue

                            exit_confirmation_counts[person_id] = 0
                            if now - last_exit_times.get(person_id, 0.0) < exit_cooldown_seconds:
                                draw_face_box(original_frame, face_box, person_id)
                                continue

                            person_record = database.get_person(person_id)
                            show_matched_person(person_id, person_record)
                            image_path = person_record.get("image_path") if person_record is not None else None
                            if image_path:
                                api_thread = threading.Thread(
                                    target=send_exit_to_api,
                                    args=(person_id, image_path),
                                    daemon=True,
                                )
                                api_thread.start()
                                api_thread.join(timeout=10)

                            print("Deleting person:", person_id)
                            if image_path and os.path.exists(image_path):
                                os.remove(image_path)
                                print(f"Deleted image: {image_path}")
                            else:
                                print("[ERROR] image not found")

                            if database.remove_person(person_id):
                                last_exit_times[person_id] = now
                                total_out += 1
                                current_inside = max(0, current_inside - 1)
                                print(f"[EXIT] {person_id} removed | Count: {current_inside}")
                            draw_face_box(original_frame, face_box, person_id)
                        else:
                            exit_confirmation_counts.clear()
                            if now - last_unknown_warning_time >= exit_cooldown_seconds:
                                print("[WARNING] unknown person")
                                last_unknown_warning_time = now
                            draw_face_box(original_frame, face_box, "Unknown")

                    display_frame = resize_frame(original_frame, args.width)
                    cv2.imshow(window_name, display_frame)
                    time.sleep(0.01)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                except Exception as exc:
                    print(f"[ERROR] exit frame processing failed: {exc}")
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping system safely")
    finally:
        capture.release()
        database.close()
        cv2.destroyAllWindows()
        current_inside = len(database.get_inside_people())
        print_summary(total_in, total_out, current_inside)


if __name__ == "__main__":
    main()
