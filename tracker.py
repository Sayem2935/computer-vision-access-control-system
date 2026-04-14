from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class PersonDetection:
    confidence: float
    box: tuple[int, int, int, int]

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


@dataclass(frozen=True)
class PersonTrack:
    track_id: int
    confidence: float
    box: tuple[int, int, int, int]
    centroid: tuple[int, int]


@dataclass(frozen=True)
class TrackingUpdate:
    tracks: list[PersonTrack]
    entered_track_ids: list[int]
    exited_track_ids: list[int]


class CentroidTracker:
    def __init__(self, max_disappeared=20, max_distance=80):
        self.next_track_id = 1
        self.tracks = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        entered_track_ids = []
        exited_track_ids = []

        if not detections:
            exited_track_ids.extend(self._mark_all_missing())
            return self._build_update(entered_track_ids, exited_track_ids)

        if not self.tracks:
            for detection in detections:
                entered_track_ids.append(self._register(detection))
            return self._build_update(entered_track_ids, exited_track_ids)

        object_ids = list(self.tracks.keys())
        object_centroids = np.array([self.tracks[track_id].centroid for track_id in object_ids])
        input_centroids = np.array([detection.centroid for detection in detections])
        distances = np.linalg.norm(object_centroids[:, None] - input_centroids[None, :], axis=2)

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distances[row, col] > self.max_distance:
                continue

            track_id = object_ids[row]
            self._update_track(track_id, detections[col])
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))) - used_rows
        unused_cols = set(range(len(detections))) - used_cols

        for row in unused_rows:
            track_id = object_ids[row]
            self.disappeared[track_id] += 1

            if self.disappeared[track_id] > self.max_disappeared:
                exited_track_ids.append(self._deregister(track_id))

        for col in unused_cols:
            entered_track_ids.append(self._register(detections[col]))

        return self._build_update(entered_track_ids, exited_track_ids)

    def _register(self, detection):
        track_id = self.next_track_id
        self.next_track_id += 1

        self.tracks[track_id] = PersonTrack(
            track_id=track_id,
            confidence=detection.confidence,
            box=detection.box,
            centroid=detection.centroid,
        )
        self.disappeared[track_id] = 0
        return track_id

    def _update_track(self, track_id, detection):
        self.tracks[track_id] = PersonTrack(
            track_id=track_id,
            confidence=detection.confidence,
            box=detection.box,
            centroid=detection.centroid,
        )
        self.disappeared[track_id] = 0

    def _deregister(self, track_id):
        self.tracks.pop(track_id, None)
        self.disappeared.pop(track_id, None)
        return track_id

    def _mark_all_missing(self):
        exited_track_ids = []

        for track_id in list(self.disappeared.keys()):
            self.disappeared[track_id] += 1

            if self.disappeared[track_id] > self.max_disappeared:
                exited_track_ids.append(self._deregister(track_id))

        return exited_track_ids

    def _build_update(self, entered_track_ids, exited_track_ids):
        tracks = sorted(self.tracks.values(), key=lambda track: track.track_id)
        return TrackingUpdate(
            tracks=tracks,
            entered_track_ids=entered_track_ids,
            exited_track_ids=exited_track_ids,
        )


class PersonTracker:
    def __init__(
        self,
        model_path="yolov8n.pt",
        confidence=0.35,
        tracker_config=None,
        device=None,
        max_disappeared=20,
        max_distance=80,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.tracker_config = tracker_config
        self.device = device
        self.centroid_tracker = CentroidTracker(
            max_disappeared=max_disappeared,
            max_distance=max_distance,
        )

    def update(self, frame):
        detections = self._detect_people(frame)
        return self.centroid_tracker.update(detections)

    def track(self, frame):
        return self.update(frame).tracks

    def _detect_people(self, frame):
        results = self.model.predict(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().tolist()

        detections = []
        for box, confidence in zip(xyxy, confidences):
            x1, y1, x2, y2 = map(int, box)
            detections.append(
                PersonDetection(
                    confidence=float(confidence),
                    box=(x1, y1, x2, y2),
                )
            )

        return detections
