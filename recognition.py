from dataclasses import dataclass
import math
import os

import cv2
import face_recognition
import numpy as np


known_encodings = []
known_ids = []
person_count = 0
FACE_MATCH_THRESHOLD = 0.47
MIN_FACE_SIZE = 60
MIN_FACE_WIDTH_FOR_ENCODING = 60
NORMALIZED_FACE_SIZE = (150, 150)
MAX_ENCODINGS_PER_PERSON = 5
MAX_SAVE_RETRIES = 3
MIN_FACE_SAMPLES = 5
MAX_FACE_SAMPLES = 10
SAVE_FACE_PADDING_RATIO = 0.4
SAVE_BLUR_THRESHOLD = 120.0
CENTER_DISTANCE_THRESHOLD = 0.55
face_save_attempts = {}
face_samples = {}
FACES_DIR = "faces"


@dataclass(frozen=True)
class FaceDetection:
    box: tuple[int, int, int, int]
    name: str
    encoding: object | None = None
    distance: float | None = None


def get_person_id(face_img):
    person_id, _, _ = get_person_data(face_img)
    return person_id


def get_person_data(face_img, create_new=True):
    encoding, _, normalized_face = extract_face_encoding(face_img)
    if encoding is None:
        return None, None, None

    person_id, distance = match_face_encoding(encoding)
    if person_id is not None:
        remember_person_encoding(person_id, encoding, face_crop=normalized_face)
    elif create_new:
        person_id = create_person_from_encoding(encoding, face_crop=normalized_face)
    else:
        return None, encoding, None

    image_path = get_face_image_path(person_id)
    return person_id, encoding, image_path


def extract_face_encoding(face_img):
    if face_img is None or face_img.size == 0:
        return None, None, None

    if face_img.shape[0] < MIN_FACE_SIZE or face_img.shape[1] < MIN_FACE_SIZE:
        return None, None, None

    detection_img = cv2.resize(
        face_img,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC,
    )
    rgb_img = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img, model="hog")
    if not face_locations:
        return None, None, None

    detection_face_location = max(
        face_locations,
        key=lambda location: (location[1] - location[3]) * (location[2] - location[0]),
    )
    face_location = scale_face_location(
        detection_face_location,
        scale_x=face_img.shape[1] / detection_img.shape[1],
        scale_y=face_img.shape[0] / detection_img.shape[0],
        max_width=face_img.shape[1],
        max_height=face_img.shape[0],
    )
    top, right, bottom, left = face_location
    if (right - left) < MIN_FACE_WIDTH_FOR_ENCODING:
        return None, None, None

    normalized_face = crop_face_region(
        face_img,
        face_location,
        padding=20,
        resize_to=NORMALIZED_FACE_SIZE,
    )
    if normalized_face is None:
        return None, None, None

    save_face = prepare_face_sample(face_img, face_location)

    face_img = cv2.cvtColor(normalized_face, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(face_img)
    if len(encodings) == 0:
        print("No encoding found")
        return None, None, None

    encoding = encodings[0]
    print("Encoding shape:", encoding.shape)
    return encoding, face_location, save_face


def get_face_location(face_img):
    if face_img is None or face_img.size == 0:
        return None

    detection_img = cv2.resize(
        face_img,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC,
    )
    rgb_face = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_face, model="hog")
    if not face_locations:
        return None

    top, right, bottom, left = scale_face_location(
        face_locations[0],
        scale_x=face_img.shape[1] / detection_img.shape[1],
        scale_y=face_img.shape[0] / detection_img.shape[0],
        max_width=face_img.shape[1],
        max_height=face_img.shape[0],
    )
    return left, top, right, bottom


def identify_face_encoding(encoding, threshold=FACE_MATCH_THRESHOLD, face_crop=None):
    person_id, distance = match_face_encoding(encoding, threshold=threshold)
    if person_id is not None:
        remember_person_encoding(person_id, encoding, face_crop=face_crop)
        return person_id, distance

    person_id = create_person_from_encoding(encoding, face_crop=face_crop)
    return person_id, distance


def match_face_encoding(encoding, threshold=FACE_MATCH_THRESHOLD):
    if not known_encodings:
        return None, None

    best_index = None
    best_distance = None
    for index, person_encodings in enumerate(known_encodings):
        encoding_matrix = np.asarray(person_encodings, dtype=np.float64)
        distances = np.linalg.norm(encoding_matrix - encoding, axis=1)
        min_distance = float(np.min(distances))
        if best_distance is None or min_distance < best_distance:
            best_index = index
            best_distance = min_distance

    if best_distance is not None and best_distance < threshold:
        print("Matching distance:", best_distance)
        return known_ids[best_index], best_distance

    if best_distance is not None:
        print("Matching distance:", best_distance)
    return None, best_distance


def create_person_from_encoding(encoding, face_crop=None):
    global person_count

    person_count += 1
    person_id = f"Person_{person_count}"
    known_encodings.append([encoding])
    known_ids.append(person_id)
    ensure_face_saved(person_id, face_crop)
    if not has_face_image(person_id) and face_crop is not None and face_crop.size != 0:
        save_face_image(person_id, face_crop)
    return person_id


def remember_person_encoding(person_id, encoding, face_crop=None):
    try:
        person_index = known_ids.index(person_id)
    except ValueError:
        return

    person_encodings = known_encodings[person_index]
    person_encodings.append(encoding)
    if len(person_encodings) > MAX_ENCODINGS_PER_PERSON:
        del person_encodings[:-MAX_ENCODINGS_PER_PERSON]

    ensure_face_saved(person_id, face_crop)


def cache_known_person(person_id, encoding, face_crop=None):
    if person_id in known_ids:
        remember_person_encoding(person_id, encoding, face_crop=face_crop)
        return

    known_ids.append(person_id)
    known_encodings.append([encoding])
    ensure_face_saved(person_id, face_crop)


def normalize_face_crop(face_img, face_location):
    return crop_face_region(
        face_img,
        face_location,
        padding=20,
        resize_to=NORMALIZED_FACE_SIZE,
    )


def crop_face_region(face_img, face_location, padding=20, resize_to=None):
    top, right, bottom, left = face_location
    face_width = right - left
    face_height = bottom - top

    if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
        return None

    x1 = max(0, left - padding)
    y1 = max(0, top - padding)
    x2 = min(face_img.shape[1], right + padding)
    y2 = min(face_img.shape[0], bottom + padding)

    face_crop = face_img[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    if resize_to is not None:
        return cv2.resize(face_crop, resize_to)

    return face_crop


def scale_face_location(face_location, scale_x, scale_y, max_width, max_height):
    top, right, bottom, left = face_location
    scaled_top = max(0, min(int(round(top * scale_y)), max_height - 1))
    scaled_right = max(0, min(int(round(right * scale_x)), max_width))
    scaled_bottom = max(0, min(int(round(bottom * scale_y)), max_height))
    scaled_left = max(0, min(int(round(left * scale_x)), max_width - 1))
    return scaled_top, scaled_right, scaled_bottom, scaled_left


def ensure_face_saved(person_id, face_crop):
    if face_crop is None or face_crop.size == 0:
        return

    if has_face_image(person_id):
        face_save_attempts.pop(person_id, None)
        face_samples.pop(person_id, None)
        return

    attempts = face_save_attempts.get(person_id, 0)
    if attempts >= MAX_SAVE_RETRIES:
        return

    add_face_sample(person_id, face_crop)
    samples = face_samples.get(person_id, [])
    if len(samples) < MIN_FACE_SAMPLES:
        if attempts >= MAX_SAVE_RETRIES - 1:
            save_face_image(person_id, face_crop)
        return

    best_face = max(samples, key=lambda sample: sample["score"])["image"]
    face_save_attempts[person_id] = attempts + 1
    save_face_image(person_id, best_face)

    if has_face_image(person_id):
        face_save_attempts.pop(person_id, None)
        face_samples.pop(person_id, None)


def has_face_image(person_id):
    os.makedirs(FACES_DIR, exist_ok=True)
    image_path = get_face_image_path(person_id)
    return os.path.exists(image_path)


def get_face_image_path(person_id):
    return os.path.join(FACES_DIR, f"{person_id}.jpg")


def save_face_image(person_id, face_crop):
    if face_crop is None or face_crop.size == 0:
        return

    image_path = get_face_image_path(person_id)
    if has_face_image(person_id):
        return

    os.makedirs(FACES_DIR, exist_ok=True)
    print("Saving image for:", person_id)
    resized_face = cv2.resize(face_crop, NORMALIZED_FACE_SIZE)
    if cv2.imwrite(image_path, resized_face):
        print(f"Saved face for {person_id}")
        print(f"Image saved at: {image_path}")
    if not os.path.exists(image_path):
        print("[ERROR] image not saved")


def add_face_sample(person_id, face_crop):
    if face_crop is None or face_crop.size == 0:
        return

    score = calculate_face_quality_score(face_crop)
    if score is None:
        return

    samples = face_samples.setdefault(person_id, [])
    samples.append(
        {
            "image": face_crop.copy(),
            "score": score,
        }
    )
    if len(samples) > MAX_FACE_SAMPLES:
        del samples[:-MAX_FACE_SAMPLES]


def calculate_face_quality_score(face_img):
    height, width = face_img.shape[:2]
    size_score = float(height * width) / 1000.0

    blur_value = calculate_blur_value(face_img)
    if blur_value < SAVE_BLUR_THRESHOLD:
        return None
    blur_score = blur_value / 100.0

    center_score = calculate_center_score(face_img)
    if center_score is None:
        return None

    return size_score + blur_score + center_score


def prepare_face_sample(face_img, face_location):
    if not is_face_near_center(face_img.shape, face_location):
        return None

    top, right, bottom, left = face_location
    dynamic_padding = int(max(right - left, bottom - top) * SAVE_FACE_PADDING_RATIO)
    face_crop = crop_face_region(face_img, face_location, padding=dynamic_padding)
    if face_crop is None:
        return None

    aligned_face = align_face(face_crop)
    normalized_face = normalize_brightness(aligned_face)
    if calculate_blur_value(normalized_face) < SAVE_BLUR_THRESHOLD:
        return None

    return normalized_face


def align_face(face_img):
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    landmarks_list = face_recognition.face_landmarks(rgb_face)
    if not landmarks_list:
        return face_img

    landmarks = landmarks_list[0]
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")
    if not left_eye or not right_eye:
        return face_img

    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    angle = math.degrees(
        math.atan2(
            right_eye_center[1] - left_eye_center[1],
            right_eye_center[0] - left_eye_center[0],
        )
    )
    eyes_center = tuple(np.mean([left_eye_center, right_eye_center], axis=0))
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    return cv2.warpAffine(
        face_img,
        rotation_matrix,
        (face_img.shape[1], face_img.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def normalize_brightness(face_img):
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def calculate_blur_value(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_center_score(face_img):
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_face)
    if not face_locations:
        return None

    height, width = face_img.shape[:2]
    top, right, bottom, left = face_locations[0]
    face_center_x = (left + right) / 2.0
    face_center_y = (top + bottom) / 2.0
    crop_center_x = width / 2.0
    crop_center_y = height / 2.0
    distance = np.hypot(face_center_x - crop_center_x, face_center_y - crop_center_y)
    max_distance = np.hypot(crop_center_x, crop_center_y)
    return max(0.0, 100.0 * (1.0 - (distance / max_distance)))


def is_face_near_center(frame_shape, face_location):
    frame_height, frame_width = frame_shape[:2]
    top, right, bottom, left = face_location
    face_center_x = (left + right) / 2.0
    face_center_y = (top + bottom) / 2.0
    frame_center_x = frame_width / 2.0
    frame_center_y = frame_height / 2.0
    distance = np.hypot(face_center_x - frame_center_x, face_center_y - frame_center_y)
    max_distance = np.hypot(frame_center_x, frame_center_y)
    return (distance / max_distance) <= CENTER_DISTANCE_THRESHOLD


class FaceRecognitionService:
    def __init__(self, scale=0.5, model="hog", tolerance=FACE_MATCH_THRESHOLD):
        self.scale = max(0.1, min(scale, 1.0))
        self.model = model
        self.tolerance = tolerance

    def recognize_faces_in_person(self, frame, person_box):
        person_crop, crop_origin = self._crop_person(frame, person_box)
        if person_crop is None:
            return []

        resized_crop = self._resize_for_detection(person_crop)
        rgb_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_crop, model=self.model)
        face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)

        faces = []
        for location, encoding in zip(face_locations, face_encodings):
            normalized_face = normalize_face_crop(resized_crop, location)
            if normalized_face is None:
                continue

            name, distance = identify_face_encoding(
                encoding,
                threshold=self.tolerance,
                face_crop=normalized_face,
            )
            face_box = self._scale_face_box(location, crop_origin)
            faces.append(
                FaceDetection(
                    box=face_box,
                    name=name,
                    encoding=encoding,
                    distance=distance,
                )
            )

        return faces

    def _crop_person(self, frame, person_box):
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, person_box)

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width))
        y2 = max(0, min(y2, frame_height))

        if x2 <= x1 or y2 <= y1:
            return None, None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        return crop, (x1, y1)

    def _resize_for_detection(self, crop):
        if self.scale >= 1.0:
            return crop

        return cv2.resize(
            crop,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_AREA,
        )

    def _scale_face_box(self, location, crop_origin):
        top, right, bottom, left = location
        origin_x, origin_y = crop_origin

        x1 = origin_x + int(left / self.scale)
        y1 = origin_y + int(top / self.scale)
        x2 = origin_x + int(right / self.scale)
        y2 = origin_y + int(bottom / self.scale)
        return x1, y1, x2, y2


FaceDetector = FaceRecognitionService
