import sys
import os
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.recognition.recognizer import FaceRecognizer

DATASET_DIR = "data/enrolled_faces"


def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def main():

    recognizer = FaceRecognizer()

    genuine_scores = []
    impostor_scores = []

    print("\n========================================")
    print("   RECOGNITION THRESHOLD EVALUATION")
    print("========================================")

    for student in os.listdir(DATASET_DIR):

        student_dir = os.path.join(DATASET_DIR, student)

        if not os.path.isdir(student_dir):
            continue

        print(f"\nProcessing: {student}")

        for filename in os.listdir(student_dir):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(student_dir, filename)
            image = cv2.imread(image_path)

            if image is None:
                continue

            faces = recognizer.app.get(image)

            if len(faces) == 0:
                print(f"  {filename} -> No face")
                continue

            if len(faces) > 1:
                print(f"  {filename} -> Multiple faces")
                continue

            embedding = faces[0].embedding
            embedding = embedding / np.linalg.norm(embedding)

            for name, enrolled_embedding in recognizer.known_embeddings.items():

                similarity = cosine_similarity(
                    embedding,
                    enrolled_embedding
                )

                if name == student:
                    genuine_scores.append(similarity)
                else:
                    impostor_scores.append(similarity)

    if not genuine_scores or not impostor_scores:
        print("\nNot enough data for evaluation.")
        return

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    print("\n========================================")
    print("              RESULTS")
    print("========================================")

    print(f"Valid genuine pairs  : {len(genuine_scores)}")
    print(f"Impostor pairs       : {len(impostor_scores)}")

    print("\nGenuine Scores")
    print(f"Minimum : {genuine_scores.min():.4f}")
    print(f"Maximum : {genuine_scores.max():.4f}")
    print(f"Average : {genuine_scores.mean():.4f}")

    print("\nImpostor Scores")
    print(f"Minimum : {impostor_scores.min():.4f}")
    print(f"Maximum : {impostor_scores.max():.4f}")
    print(f"Average : {impostor_scores.mean():.4f}")

    # Find threshold with minimum FAR + FRR
    all_scores = np.concatenate(
        [genuine_scores, impostor_scores]
    )

    thresholds = np.linspace(
        all_scores.min(),
        all_scores.max(),
        500
    )

    best_threshold = None
    best_error = float("inf")
    best_far = None
    best_frr = None

    for threshold in thresholds:

        false_accepts = np.sum(
            impostor_scores >= threshold
        )

        false_rejects = np.sum(
            genuine_scores < threshold
        )

        far = false_accepts / len(impostor_scores)
        frr = false_rejects / len(genuine_scores)

        total_error = far + frr

        if total_error < best_error:
            best_error = total_error
            best_threshold = threshold
            best_far = far
            best_frr = frr

    print("\n========================================")
    print("         BEST THRESHOLD")
    print("========================================")

    print(f"Threshold : {best_threshold:.4f}")
    print(f"FAR       : {best_far * 100:.2f}%")
    print(f"FRR       : {best_frr * 100:.2f}%")

    print("========================================")


if __name__ == "__main__":
    main()