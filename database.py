import os
import shutil
import sqlite3
from datetime import datetime

import face_recognition
import numpy as np


DB_PATH = "people.db"
EXITED_FACES_DIR = "exited_faces"
FACES_DIR = "faces"


def current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db(db_path=DB_PATH):
    connection = sqlite3.connect(db_path, timeout=10)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS people_inside (
                person_id TEXT PRIMARY KEY,
                encoding BLOB,
                image_path TEXT,
                entry_time TEXT
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


class PeopleDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.initialize()

    def initialize(self):
        init_db(self.db_path)

    def close(self):
        return None

    def add_person(self, person_id, encoding=None, image_path=None, print_log=True):
        entry_time = current_timestamp()
        if image_path is None:
            image_path = f"{FACES_DIR}/{person_id}.jpg"

        encoding_bytes = None
        if encoding is not None:
            encoding_bytes = encoding.tobytes()

        connection = self._connect()
        try:
            person = connection.execute(
                "SELECT person_id FROM people_inside WHERE person_id = ?",
                (person_id,),
            ).fetchone()

            if person is not None:
                return False

            connection.execute(
                """
                INSERT INTO people_inside (person_id, encoding, image_path, entry_time)
                VALUES (?, ?, ?, ?)
                """,
                (person_id, encoding_bytes, image_path, entry_time),
            )
            connection.commit()
        finally:
            connection.close()

        if encoding_bytes is not None:
            print(f"Encoding saved for {person_id}")
        print(f"Linked image {image_path} with {person_id}")
        if print_log:
            print(f"Saved to DB: {person_id}")
        return True

    def remove_person(self, person_id):
        connection = self._connect()
        try:
            person = connection.execute(
                """
                SELECT image_path
                FROM people_inside
                WHERE person_id = ?
                """,
                (person_id,),
            ).fetchone()

            image_path = person["image_path"] if person is not None else None
            result = connection.execute(
                """
                DELETE FROM people_inside
                WHERE person_id = ?
                """,
                (person_id,),
            )
            connection.commit()
        finally:
            connection.close()

        if result.rowcount == 0:
            return False

        print(f"Removed from DB: {person_id}")
        if image_path and os.path.exists(image_path):
            os.makedirs(EXITED_FACES_DIR, exist_ok=True)
            destination_path = os.path.join(EXITED_FACES_DIR, os.path.basename(image_path))
            shutil.move(image_path, destination_path)
            print(f"Moved {person_id} image to exited_faces")
        return True

    def get_inside_people(self):
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT person_id, encoding, image_path, entry_time
                FROM people_inside
                ORDER BY entry_time
                """
            ).fetchall()
        finally:
            connection.close()

        return [dict(row) for row in rows]

    def get_person(self, person_id):
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT person_id, encoding, image_path, entry_time
                FROM people_inside
                WHERE person_id = ?
                """,
                (person_id,),
            ).fetchone()
        finally:
            connection.close()

        return dict(row) if row is not None else None

    def load_all_people(self):
        encodings = []
        person_ids = []

        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT person_id, encoding
                FROM people_inside
                WHERE encoding IS NOT NULL
                ORDER BY entry_time
                """
            ).fetchall()
        finally:
            connection.close()

        for row in rows:
            encoding_bytes = row["encoding"]
            if encoding_bytes is None:
                continue

            encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
            if encoding.shape != (128,):
                continue

            print("Loaded encoding for:", row["person_id"])
            encodings.append(encoding)
            person_ids.append(row["person_id"])

        return encodings, person_ids

    def match_person(self, new_encoding, threshold=0.45):
        encodings, person_ids = self.load_all_people()
        if not encodings:
            print("DB distances: []")
            return None

        distances = face_recognition.face_distance(encodings, new_encoding)
        debug_values = [f"{person_id}={distance:.3f}" for person_id, distance in zip(person_ids, distances)]
        print(f"DB distances: {debug_values}")

        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])

        if best_distance < threshold:
            return person_ids[best_index]

        return None

    def get_inside_ids(self):
        return {person["person_id"] for person in self.get_inside_people()}

    def get_inside_track_ids(self):
        return self.get_inside_ids()

    def enter_track(self, track_id):
        return self.add_person(format_track_id(track_id))

    def exit_track(self, track_id):
        return False

    def update_track_name(self, track_id, name):
        return False

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection


def add_person(person_id, encoding, image_path, db_path=DB_PATH):
    return PeopleDatabase(db_path).add_person(
        person_id,
        encoding=encoding,
        image_path=image_path,
        print_log=True,
    )


def remove_person(person_id, db_path=DB_PATH):
    return PeopleDatabase(db_path).remove_person(person_id)


def get_inside_people(db_path=DB_PATH):
    return PeopleDatabase(db_path).get_inside_people()


def get_person(person_id, db_path=DB_PATH):
    return PeopleDatabase(db_path).get_person(person_id)


def load_all_people(db_path=DB_PATH):
    return PeopleDatabase(db_path).load_all_people()


def match_person(new_encoding, db_path=DB_PATH):
    return PeopleDatabase(db_path).match_person(new_encoding)


def format_track_id(track_id):
    return f"Track_{track_id}"


init_db()
