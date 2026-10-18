"""Frozen MobileNet visual features plus a locally trained, portable binary classifier.

No pose detector or TensorFlow training installation is required. The backbone is
pretrained on ImageNet; only the logistic head learns from your human labels.
"""

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from .model import Prediction

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_embedder/"
    "mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite"
)
MODEL_SHA256 = "bbbb4c51a55a53905af1daec995ca1aae355046f8839bb8c9f5ce9271394bc40"
EMBEDDING_SIZE = 1024


def download_image_model(settings):
    path = settings.image_model_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        if not path.is_file():
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != MODEL_SHA256:
                raise ValueError("Image encoder download checksum differs from the verified model")
            temporary.replace(path)
        image_context(settings)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Image encoder: {path}; sha256={MODEL_SHA256}")


def image_context(settings):
    digest = hashlib.sha256(settings.image_model_path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("Use the verified MobileNet encoder from training_workflow.py download-model")
    return {
        "encoder_sha256": digest,
        "preprocessing": "bgr-to-rgb-resize224-div255-l2-v1",
        "embedding_size": EMBEDDING_SIZE,
        "crop": list(settings.image_crop),
        "width": settings.width,
        "height": settings.height,
    }


def preprocess_image(image, crop):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image encoder requires a uint8 BGR image")
    h, w = image.shape[:2]
    left, top, right, bottom = crop
    region = image[int(top * h) : int(bottom * h), int(left * w) : int(right * w)]
    if not region.size:
        raise ValueError("Image crop contains no pixels")
    rgb = cv2.cvtColor(cv2.resize(region, (224, 224)), cv2.COLOR_BGR2RGB)
    # Verified in this model's embedded NormalizationOptions: mean=0, std=255.
    return (rgb.astype(np.float32) / 255.0)[None, ...]


class ImageExtractor:
    def __init__(self, settings):
        from ai_edge_litert.interpreter import Interpreter

        image_context(settings)
        self.crop = settings.image_crop
        self.interpreter = Interpreter(model_path=str(settings.image_model_path), num_threads=2)
        self.interpreter.allocate_tensors()
        inputs, outputs = self.interpreter.get_input_details(), self.interpreter.get_output_details()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or tuple(inputs[0]["shape"]) != (1, 224, 224, 3)
            or inputs[0]["dtype"] != np.float32
            or tuple(outputs[0]["shape"]) != (1, EMBEDDING_SIZE)
            or outputs[0]["dtype"] != np.float32
        ):
            raise ValueError("Image encoder input/output schema differs from the training workflow")
        self.input_index, self.output_index = inputs[0]["index"], outputs[0]["index"]

    def extract(self, image, elapsed_s=0):
        self.interpreter.set_tensor(self.input_index, preprocess_image(image, self.crop))
        self.interpreter.invoke()
        values = self.interpreter.get_tensor(self.output_index)[0].astype(np.float64)
        norm = np.linalg.norm(values)
        if not np.isfinite(values).all() or not np.isfinite(norm) or norm <= 1e-10:
            raise ValueError("Image encoder returned invalid features")
        return values / norm

    def close(self):
        pass


class ImageModel:
    def __init__(self, settings):
        artifact = json.loads(settings.classifier_path.read_text())
        if artifact.get("schema_version") != 1 or artifact.get("kind") != "image_logistic":
            raise ValueError("Expected a trained image classifier JSON; run training_workflow.py train")
        if artifact.get("image_context") != image_context(settings):
            raise ValueError("Image encoder/crop/dimensions differ from training; use the exported config")
        self.weights = np.asarray(artifact["weights"], dtype=np.float64)
        self.bias = float(artifact["bias"])
        self.threshold = settings.local_threshold
        if self.weights.shape != (EMBEDDING_SIZE,) or not np.isfinite(self.weights).all():
            raise ValueError("Invalid image classifier weights")
        if not np.isfinite(self.bias):
            raise ValueError("Invalid image classifier bias")

    def predict(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float64)
        if embedding.shape != self.weights.shape or not np.isfinite(embedding).all():
            raise ValueError("Invalid image embedding")
        logit = float(np.dot(self.weights, embedding) + self.bias)
        score = float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))
        return Prediction(score >= self.threshold, score, "mobilenet_image_classifier")


@dataclass(frozen=True)
class ImageObservation:
    sustained_flag_s: float
    required_flag_s: float
    image_crop: tuple[float, float, float, float]
    kind: str = "image_classification"
