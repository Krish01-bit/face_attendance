import sys
import os
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.recognition.recognizer import FaceRecognizer

IMAGE_PATH = r"D:\papa ka documents\Rajiv photo.jpg"


def main():
    recognizer = FaceRecognizer()

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("FAIL: Test image not found.")
        return

    name, similarity = recognizer.recognize(image)

    print("\nRecognition Test")
    print("----------------")
    print(f"Name: {name}")
    print(f"Similarity: {similarity:.4f}")

    if name == "Balaji":
        print("\nPASS: Correct person recognized.")
    else:
        print("\nFAIL: Person was not recognized correctly.")


if __name__ == "__main__":
    main()