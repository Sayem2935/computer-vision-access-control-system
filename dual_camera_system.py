import multiprocessing


CAMERA_1 = "rtsp://admin:Shuborno%40500@192.168.0.200:554/Streaming/Channels/102"
CAMERA_2 = "rtsp://admin:Shuborno%40500@192.168.0.201:554/Streaming/Channels/102"


def run_entry():
    import main

    print("Entry system started")
    try:
        main.main(rtsp_url=CAMERA_1)
    except KeyboardInterrupt:
        print("Entry system stopped safely")
    except Exception as exc:
        print(f"[ERROR] entry system failed: {exc}")


def run_exit():
    import exit_system

    print("Exit system started")
    try:
        exit_system.main(rtsp_url=CAMERA_2)
    except KeyboardInterrupt:
        print("Exit system stopped safely")
    except Exception as exc:
        print(f"[ERROR] exit system failed: {exc}")


def main_runner():
    p1 = multiprocessing.Process(target=run_entry)
    p2 = multiprocessing.Process(target=run_exit)

    try:
        p1.start()
        p2.start()

        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("Stopping system safely")
        if p1.is_alive():
            p1.terminate()
            p1.join()
        if p2.is_alive():
            p2.terminate()
            p2.join()


if __name__ == "__main__":
    main_runner()
