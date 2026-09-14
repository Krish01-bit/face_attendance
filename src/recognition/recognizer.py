import os
import cv2
import numpy as np
import torch
import onnxruntime as ort
from insightface.app import FaceAnalysis


EMBEDDINGS_DIR = "data/embeddings"
RECOGNITION_THRESHOLD = 0.35


class FaceRecognizer:

    def __init__(self):

        # Make sure PyTorch loads CUDA/cuDNN before InsightFace.
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

        self.known_embeddings = self.load_embeddings()

    def load_embeddings(self):

        embeddings = {}

        if not os.path.isdir(EMBEDDINGS_DIR):
            print("Embeddings directory not found.")
            return embeddings

        for file in os.listdir(EMBEDDINGS_DIR):

            if not file.endswith(".npy"):
                continue

            student_name = os.path.splitext(file)[0]

            file_path = os.path.join(
                EMBEDDINGS_DIR,
                file,
            )

            embedding = np.load(file_path)

            embedding = embedding / np.linalg.norm(embedding)

            embeddings[student_name] = embedding

        print(f"Loaded {len(embeddings)} enrolled students.")

        return embeddings

    def get_embedding(self, image):

        faces = self.app.get(image)

        if len(faces) == 0:
            return None

        if len(faces) > 1:
            return None

        embedding = faces[0].embedding

        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def recognize(self, image):

        embedding = self.get_embedding(image)

        if embedding is None:
            return "UNKNOWN", 0.0

        best_name = "UNKNOWN"
        best_score = -1.0

        for student_name, known_embedding in self.known_embeddings.items():

            score = float(
                np.dot(embedding, known_embedding)
            )

            if score > best_score:
                best_score = score
                best_name = student_name

        if best_score < RECOGNITION_THRESHOLD:
            best_name = "UNKNOWN"

        return best_name, best_score