import cv2
from ultralytics import YOLO


MODEL_PATH = "models/yolov11n-face.pt"

IMAGE_PATH = r"D:\projects\face_attendance\data\enrolled_faces\anish\IMG_20260825_192347730_MP.jpg"


def main():

    model = YOLO(MODEL_PATH)

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Test image not found.")
        return

    results = model(image, verbose=False)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("FAIL: No face detected.")
        return

    print("PASS: Face detected successfully.")
    print(f"Faces detected: {len(boxes)}")

    for i, box in enumerate(boxes.xyxy.cpu().tolist(), start=1):
        x1, y1, x2, y2 = map(int, box)
        print(f"Face {i}: ({x1}, {y1}) -> ({x2}, {y2})")


if __name__ == "__main__":
    main()