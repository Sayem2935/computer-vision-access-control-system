import sys
import threading
from contextlib import contextmanager

from fastapi import FastAPI

import exit_system
import main
from database import load_all_people


app = FastAPI(title="People Tracking API")

state_lock = threading.Lock()
entry_thread = None
exit_thread = None
entry_running = False
exit_running = False
stop_requested = False


@contextmanager
def patched_argv(argv):
    original_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original_argv


def run_entry_system():
    global entry_running
    try:
        with patched_argv(["main.py"]):
            main.main()
    except SystemExit:
        pass
    except Exception as exc:
        print(f"Entry system stopped with error: {exc}")
    finally:
        with state_lock:
            entry_running = False


def run_exit_system():
    global exit_running
    try:
        with patched_argv(["exit_system.py"]):
            exit_system.main()
    except SystemExit:
        pass
    except Exception as exc:
        print(f"Exit system stopped with error: {exc}")
    finally:
        with state_lock:
            exit_running = False


@app.post("/start-entry")
def start_entry():
    global entry_thread, entry_running, stop_requested
    with state_lock:
        if entry_running and entry_thread is not None and entry_thread.is_alive():
            return {"status": "already_running", "system": "entry"}

        stop_requested = False
        entry_running = True
        entry_thread = threading.Thread(target=run_entry_system, daemon=True)
        entry_thread.start()
        return {"status": "started", "system": "entry"}


@app.post("/start-exit")
def start_exit():
    global exit_thread, exit_running, stop_requested
    with state_lock:
        if exit_running and exit_thread is not None and exit_thread.is_alive():
            return {"status": "already_running", "system": "exit"}

        stop_requested = False
        exit_running = True
        exit_thread = threading.Thread(target=run_exit_system, daemon=True)
        exit_thread.start()
        return {"status": "started", "system": "exit"}


@app.get("/people")
def get_people():
    _, person_ids = load_all_people()
    return {"people": person_ids}


@app.get("/count")
def get_count():
    _, person_ids = load_all_people()
    return {"count": len(person_ids)}


@app.post("/stop")
def stop_systems():
    global stop_requested
    with state_lock:
        stop_requested = True
        return {
            "status": "stop_requested",
            "entry_running": bool(entry_thread and entry_thread.is_alive()),
            "exit_running": bool(exit_thread and exit_thread.is_alive()),
        }
