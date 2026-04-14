import argparse
import threading
import time

import cv2
import face_recognition

from api_sender import send_exit_to_api
from camera import DEFAULT_RTSP_URL
from database import DB_PATH, PeopleDatabase, get_person, match_person, remove_person
from recognition import extract_face_encoding


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exit system that matches faces against the database and removes them."
    )
    parser.add_argument("--rtsp", default=DEFAULT_RTSP_URL, help="RTSP stream URL.")
    parser.add_argument("--database", default=DB_PATH, help="SQLite database path.")
    parser.add_argument("--width", type=int, default=960, help="Resize frame width.")
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


def main():
    args = parse_args()
    database = PeopleDatabase(args.database)
    total_in = len(database.get_inside_people())
    total_out = 0
    current_inside = total_in
    exit_cooldown_seconds = 3
    last_exit_times = {}
    last_unknown_warning_time = 0.0

    capture = cv2.VideoCapture(args.rtsp)
    if not capture.isOpened():
        print("Error: Cannot connect to camera")
        return

    window_name = "Exit System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("Frame read failed")
                    continue

                frame = resize_frame(frame, args.width)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)

                for face_location in face_locations:
                    face_crop, face_box = crop_face_with_padding(frame, face_location)
                    new_encoding, _, _ = extract_face_encoding(face_crop)
                    if new_encoding is None:
                        continue

                    person_id = match_person(new_encoding, db_path=args.database)
                    now = time.time()
                    if person_id is not None:
                        if now - last_exit_times.get(person_id, 0.0) < exit_cooldown_seconds:
                            draw_face_box(frame, face_box, person_id)
                            continue

                        person_record = get_person(person_id, db_path=args.database)
                        show_matched_person(person_id, person_record)
                        image_path = person_record.get("image_path") if person_record is not None else None
                        if image_path:
                            threading.Thread(
                                target=send_exit_to_api,
                                args=(person_id, image_path),
                                daemon=True,
                            ).start()
                        if remove_person(person_id, db_path=args.database):
                            last_exit_times[person_id] = now
                            total_out += 1
                            current_inside = max(0, current_inside - 1)
                            print(f"EXIT: {person_id}")
                        draw_face_box(frame, face_box, person_id)
                    else:
                        if now - last_unknown_warning_time >= exit_cooldown_seconds:
                            print("WARNING: Unknown person exiting")
                            last_unknown_warning_time = now
                        draw_face_box(frame, face_box, "Unknown")

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
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
