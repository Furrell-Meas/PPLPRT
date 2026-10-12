"""
redaction.py
Core detection + anonymisation engine for the Privacy-Preserving
Licence Plate Redaction Tool (PPLPRT).

Maps directly onto the RedactionEngine class in the UML diagram:
    - preprocessImage(media)
    - detectPlates(media)
    - filterDetections()
    - expandBoundingBox()
    - applyGaussian()
    - generateOutput()
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration (mirrors RedactionEngine attributes: detectionModel,
# confidenceThreshold, paddingPercent, blurKernel)
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

DEFAULT_PADDING_PERCENT = 0.15
DEFAULT_SCALE_FACTOR = 1.05
DEFAULT_MIN_NEIGHBORS = 4
DEFAULT_MIN_SIZE = (25, 25)

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
_plate_cascade = cv2.CascadeClassifier(_CASCADE_PATH)
if _plate_cascade.empty():
    raise RuntimeError(f"Could not load cascade classifier from {_CASCADE_PATH}")

# ---------------------------------------------------------------------------
# Optional YOLOv8 detector. If a trained model exists at MODEL_PATH, it is
# used automatically instead of the Haar cascade fallback above. This lets
# you drop in your CCPD-trained best.pt without touching any other code.
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "best.pt")
YOLO_CONFIDENCE_THRESHOLD = 0.4

_yolo_model = None
_yolo_load_attempted = False


def _get_yolo_model():
    """Lazily loads the YOLOv8 model on first use, if present. Returns None
    (and falls back to Haar cascade) if no trained model has been added yet."""
    global _yolo_model, _yolo_load_attempted
    if _yolo_load_attempted:
        return _yolo_model
    _yolo_load_attempted = True
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(MODEL_PATH)
        print(f"[redaction] Loaded YOLOv8 model from {MODEL_PATH}")
    except Exception as e:
        print(f"[redaction] Failed to load YOLOv8 model ({e}); using Haar cascade fallback.")
        _yolo_model = None
    return _yolo_model


class ValidationError(Exception):
    """Raised when an uploaded file fails validation (Report Invalid File /
    Report Unsupported Format in the functional diagram)."""


class MediaFile:
    """Lightweight equivalent of the MediaFile class in the UML diagram."""

    def __init__(self, file_path: str, original_filename: str):
        self.file_path = file_path
        self.file_type = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""
        self.file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        self.image = None

    def validate(self):
        """Validate Image File step in the functional diagram."""
        if not self.file_type or self.file_type not in ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file format '.{self.file_type}'. Please upload a JPG or PNG image."
            )
        if self.file_size == 0:
            raise ValidationError("The uploaded file is empty.")
        if self.file_size > MAX_FILE_SIZE_BYTES:
            raise ValidationError("File too large. Maximum allowed size is 10MB.")

        img = cv2.imread(self.file_path)
        if img is None:
            raise ValidationError("The file could not be read as a valid image.")
        self.image = img
        return img

    def load(self):
        if self.image is None:
            self.image = cv2.imread(self.file_path)
        return self.image


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Pre-process Image step: grayscale + histogram equalisation improves
    Haar cascade detection reliability under varied lighting."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray


def _detect_plates_yolo(img: np.ndarray, model):
    """Run the trained YOLOv8 detector. Returns list of (x, y, w, h, confidence)."""
    results = model(img, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), conf))
    return detections


def _detect_plates_cascade(img: np.ndarray, scale_factor, min_neighbors, min_size):
    """Fallback detector using OpenCV's bundled Haar cascade."""
    gray = preprocess_image(img)
    boxes, _, weights = _plate_cascade.detectMultiScale3(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        outputRejectLevels=True,
    )
    detections = []
    for (x, y, w, h), weight in zip(boxes, weights):
        detections.append((int(x), int(y), int(w), int(h), float(weight)))
    return detections


def detect_plates(img: np.ndarray, scale_factor=DEFAULT_SCALE_FACTOR,
                   min_neighbors=DEFAULT_MIN_NEIGHBORS, min_size=DEFAULT_MIN_SIZE):
    """Detect Licence Plates step. Uses the trained YOLOv8 model if one has
    been added at model/best.pt, otherwise falls back to the Haar cascade.
    Returns list of (x, y, w, h, confidence)."""
    model = _get_yolo_model()
    if model is not None:
        return _detect_plates_yolo(img, model)
    return _detect_plates_cascade(img, scale_factor, min_neighbors, min_size)


def filter_detections(detections, confidence_threshold=0.0):
    """Filter Low-Confidence Detections step."""
    return [d for d in detections if d[4] >= confidence_threshold]


def expand_bounding_box(x, y, w, h, img_shape, padding_percent=DEFAULT_PADDING_PERCENT):
    """Expand Bounding Box step: pad by X% of box size, clip to image edge."""
    pad_w = int(w * padding_percent)
    pad_h = int(h * padding_percent)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_shape[1], x + w + pad_w)
    y2 = min(img_shape[0], y + h + pad_h)
    return x1, y1, x2, y2


def apply_gaussian_blur(img: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    """Apply Gaussian Blur step: kernel size scaled to box dimensions."""
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img
    h, w = roi.shape[:2]
    k = max(15, (min(w, h) // 2))
    if k % 2 == 0:
        k += 1  # kernel size must be odd
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return img


def redact_image(img: np.ndarray, padding_percent=DEFAULT_PADDING_PERCENT,
                  confidence_threshold=0.0):
    """
    Full pipeline: processMedia(media) in the UML diagram.
    Returns (redacted_image, boxes_used, plates_found: bool)
    """
    output = img.copy()
    detections = detect_plates(img)
    detections = filter_detections(detections, confidence_threshold)

    boxes_used = []
    for (x, y, w, h, conf) in detections:
        x1, y1, x2, y2 = expand_bounding_box(x, y, w, h, img.shape, padding_percent)
        output = apply_gaussian_blur(output, x1, y1, x2, y2)
        boxes_used.append((x1, y1, x2, y2))

    plates_found = len(boxes_used) > 0
    return output, boxes_used, plates_found
