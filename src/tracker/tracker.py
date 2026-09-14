from ultralytics import YOLO


MODEL_PATH = "models/yolov11n-face.pt"


def main():
    model = YOLO(MODEL_PATH)

    results = model.track(
        source=0,
        show=True,
        tracker="botsort.yaml",
        persist=True,
        stream=True,
    )

    for result in results:
        boxes = result.boxes

        if boxes is None or boxes.id is None:
            continue

        track_ids = boxes.id.int().tolist()
        coordinates = boxes.xyxy.tolist()

        for track_id, bbox in zip(track_ids, coordinates):
            x1, y1, x2, y2 = map(int, bbox)

            print(
                f"Track ID: {track_id} | "
                f"BBox: ({x1}, {y1}, {x2}, {y2})"
            )


if __name__ == "__main__":
    main()