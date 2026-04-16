import os
from datetime import datetime

import requests


ENTRY_API_URL = "https://api.khalo66.com/api/entry"
EXIT_API_URL = "https://api.khalo66.com/api/exitt"


def send_entry_to_api(person_id, image_path):
    data = {
        "person_id": person_id,
        "event": "entry",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        if not image_path or not os.path.exists(image_path):
            print("[ERROR] image not found")
            return

        with open(image_path, "rb") as img:
            files = {
                "image": img,
            }
            response = requests.post(
                ENTRY_API_URL,
                data=data,
                files=files,
                timeout=10,
            )
        print(f"ENTRY API status: {response.status_code}")
        print(f"ENTRY API response: {response.text}")
    except Exception as exc:
        print(f"[ERROR] entry api failed: {exc}")


def send_exit_to_api(person_id, image_path):
    data = {
        "person_id": person_id,
        "event": "exit",
        
    }

    try:
        if not image_path or not os.path.exists(image_path):
            print("[ERROR] image not found")
            return

        with open(image_path, "rb") as img:
            files = {
                "image": img,
            }
            response = requests.post(
                EXIT_API_URL,
                data=data,
                files=files,
                timeout=10,
            )
        print("EXIT API status:", response.status_code)
        print("EXIT API response:", response.text)
    except Exception as exc:
        print(f"[ERROR] exit api failed: {exc}")
