import os
import json
import numpy as np
import cv2
import torch
import onnxruntime as ort

from insightface.app import FaceAnalysis


ENROLLED_DIR = "data/enrolled_faces"
EMBEDDING_DIR = "data/embeddings"


def create_face_app():
    """
    Create InsightFace model using GPU.
    """
    ort.preload_dlls()

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    return app


def normalize_embedding(embedding):
    """
    Normalize an embedding vector.
    """
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


def get_image_files(student_dir):
    """
    Get all supported image files from a student's folder.
    """
    valid_extensions = (".jpg", ".jpeg", ".png")

    files = []

    for filename in os.listdir(student_dir):
        if filename.lower().endswith(valid_extensions):
            files.append(os.path.join(student_dir, filename))

    return sorted(files)


def get_metadata_path(student_name):
    return os.path.join(
        EMBEDDING_DIR,
        f"{student_name}.json"
    )


def get_current_metadata(image_files):
    """
    Store filename, file size and modification time
    for detecting enrollment changes.
    """
    metadata = {}

    for image_path in image_files:
        metadata[os.path.basename(image_path)] = {
            "size": os.path.getsize(image_path),
            "modified": os.path.getmtime(image_path)
        }

    return metadata


def embedding_needs_update(student_name, image_files):
    """
    Check whether the student's embedding is missing
    or their enrollment images have changed.
    """
    embedding_path = os.path.join(
        EMBEDDING_DIR,
        f"{student_name}.npy"
    )

    metadata_path = get_metadata_path(student_name)

    # No embedding yet
    if not os.path.exists(embedding_path):
        return True

    # No metadata yet
    if not os.path.exists(metadata_path):
        return True

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            old_metadata = json.load(file)
    except (json.JSONDecodeError, OSError):
        return True

    current_metadata = get_current_metadata(image_files)

    return old_metadata != current_metadata


def enroll_student(app, student_name):
    """
    Generate an embedding for one student.
    """
    student_dir = os.path.join(
        ENROLLED_DIR,
        student_name
    )

    image_files = get_image_files(student_dir)

    if not image_files:
        print(f"[SKIP] No images found for {student_name}")
        return False

    embeddings = []

    print(f"\nProcessing student: {student_name}")

    for image_path in image_files:

        image = cv2.imread(image_path)

        if image is None:
            print(f"[SKIP] Could not read: {image_path}")
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print(f"[SKIP] No face found: {image_path}")
            continue

        if len(faces) > 1:
            print(f"[SKIP] Multiple faces found: {image_path}")
            continue

        embedding = faces[0].embedding
        embedding = normalize_embedding(embedding)

        embeddings.append(embedding)

    if not embeddings:
        print(f"[FAIL] No valid face images for {student_name}")
        return False

    # Average all valid image embeddings
    final_embedding = np.mean(embeddings, axis=0)

    # Normalize final embedding
    final_embedding = normalize_embedding(final_embedding)

    embedding_path = os.path.join(
        EMBEDDING_DIR,
        f"{student_name}.npy"
    )

    metadata_path = get_metadata_path(student_name)

    np.save(embedding_path, final_embedding)

    metadata = get_current_metadata(image_files)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(
        f"[UPDATED] {student_name} "
        f"({len(embeddings)} valid images)"
    )

    return True


def update_embeddings():
    """
    Check all enrolled students and generate embeddings
    only when required.
    """
    os.makedirs(ENROLLED_DIR, exist_ok=True)
    os.makedirs(EMBEDDING_DIR, exist_ok=True)

    app = create_face_app()

    student_names = [
        name
        for name in os.listdir(ENROLLED_DIR)
        if os.path.isdir(os.path.join(ENROLLED_DIR, name))
    ]

    if not student_names:
        print("[WARNING] No enrolled students found.")
        return

    updated_count = 0

    for student_name in sorted(student_names):

        student_dir = os.path.join(
            ENROLLED_DIR,
            student_name
        )

        image_files = get_image_files(student_dir)

        if embedding_needs_update(
            student_name,
            image_files
        ):
            enroll_student(app, student_name)
            updated_count += 1
        else:
            print(f"[UNCHANGED] {student_name}")

    print(
        f"\nEnrollment check complete. "
        f"{updated_count} student(s) updated."
    )


if __name__ == "__main__":
    update_embeddings()