import cv2
from ultralytics import YOLO
from attendance.attendance_manager import AttendanceManager
from recognition.recognizer import FaceRecognizer


MODEL_PATH = "models/yolov11n-face.pt"

RECOGNITION_INTERVAL = 30
REQUIRED_CONFIRMATIONS = 3

def main():
    attendance_manager = AttendanceManager()
    model = YOLO(MODEL_PATH)
    recognizer = FaceRecognizer()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Starting face tracking + recognition...")
    print("Press Q to quit.")

    # Stores recognition information for each BoT-SORT track.
    track_memory = {}

    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame_count += 1
        active_track_ids = set()

        results = model.track(
            frame,
            tracker="botsort.yaml",
            persist=True,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:

            track_ids = boxes.id.int().cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()

            for track_id, bbox in zip(track_ids, coordinates):

                x1, y1, x2, y2 = map(int, bbox)

                h, w = frame.shape[:2]


                margin = 0.25

                box_width = x2 - x1
                box_height = y2 - y1

                x1 = max(0, int(x1 - box_width * margin))
                y1 = max(0, int(y1 - box_height * margin))
                x2 = min(w, int(x2 + box_width * margin))
                y2 = min(h, int(y2 + box_height * margin))

                face_crop = frame[y1:y2, x1:x2]
                active_track_ids.add(track_id)
                

                if face_crop.size == 0:
                    continue

                # Check whether this track already has a recognition result.
                track_info = track_memory.get(track_id)

                should_recognize = (
                    track_info is None
                    or frame_count - track_info["last_recognition"]
                    >= RECOGNITION_INTERVAL
                )

                if should_recognize:

                    name, similarity = recognizer.recognize(face_crop)

                    if name == "UNKNOWN":
                        confirmation_count = 0

                    elif track_info is not None and name == track_info["name"]:
                        confirmation_count = track_info["confirmation_count"] + 1

                    else:
                        confirmation_count = 1

                    track_memory[track_id] = {
                        "name": name,
                        "similarity": similarity,
                        "last_recognition": frame_count,
                        "confirmation_count": confirmation_count,
                    }

                    if confirmation_count >= REQUIRED_CONFIRMATIONS:
                        attendance_manager.mark_attendance(name)

                else:
                    name = track_info["name"]
                    similarity = track_info["similarity"]

                track_memory = {track_id: info for track_id, info in track_memory.items() if track_id in active_track_ids}

                label = (
                    f"ID: {track_id} | "
                    f"{name} | "
                    f"{similarity:.2f}"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow(
            "Face Attendance - Tracking + Recognition",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()