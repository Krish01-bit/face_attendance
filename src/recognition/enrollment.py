import os
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


ENROLLED_FACES_DIR = "data/enrolled_faces"
EMBEDDINGS_DIR = "data/embeddings"


class FaceEnrollment:

    def __init__(self):

        # Use the same CUDA setup as recognition.
        ort.preload_dlls()

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640),
        )

    def enroll_student(self, student_name):

        student_dir = os.path.join(
            ENROLLED_FACES_DIR,
            student_name,
        )

        if not os.path.isdir(student_dir):
            print(
                f"Student folder not found: {student_dir}"
            )
            return False

        embeddings = []

        for filename in os.listdir(student_dir):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                student_dir,
                filename,
            )

            image = cv2.imread(image_path)

            if image is None:
                print(
                    f"Could not read: {filename}"
                )
                continue

            faces = self.app.get(image)

            if len(faces) == 0:
                print(
                    f"No face found: {filename}"
                )
                continue

            if len(faces) > 1:
                print(
                    f"Multiple faces found: {filename}"
                )
                continue

            embedding = faces[0].embedding

            embedding = embedding / np.linalg.norm(
                embedding
            )

            embeddings.append(embedding)

            print(
                f"Processed: {filename}"
            )

        if len(embeddings) == 0:
            print(
                f"No valid face images found for "
                f"{student_name}."
            )
            return False

        # For now, create one representative embedding
        # by averaging the valid embeddings.
        mean_embedding = np.mean(
            embeddings,
            axis=0,
        )

        mean_embedding = mean_embedding / np.linalg.norm(
            mean_embedding
        )

        os.makedirs(
            EMBEDDINGS_DIR,
            exist_ok=True,
        )

        output_path = os.path.join(
            EMBEDDINGS_DIR,
            f"{student_name}.npy",
        )

        np.save(
            output_path,
            mean_embedding,
        )

        print()
        print(
            f"Enrollment completed: {student_name}"
        )
        print(
            f"Valid images: {len(embeddings)}"
        )
        print(
            f"Embedding saved: {output_path}"
        )

        return True


if __name__ == "__main__":
    enrollment = FaceEnrollment()

    if not os.path.isdir(ENROLLED_FACES_DIR):
        print(f"Enrollment folder not found: {ENROLLED_FACES_DIR}")
        exit()

    students = [
        name
        for name in os.listdir(ENROLLED_FACES_DIR)
        if os.path.isdir(os.path.join(ENROLLED_FACES_DIR, name))
    ]

    if not students:
        print("No student folders found.")
        exit()

    print("========================================")
    print("        FACE ENROLLMENT")
    print("========================================")

    for student_name in students:
        print(f"\nProcessing: {student_name}")
        enrollment.enroll_student(student_name)

    print("\n========================================")
    print("Enrollment process completed.")
    print(f"Students processed: {len(students)}")
    print("========================================")