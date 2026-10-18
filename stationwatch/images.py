"""Prepare JPEG/PNG/WebP images for the Jev image API."""

import cv2
import numpy as np
from fastapi import HTTPException

MAX_IMAGE_PIXELS = 200_000_000


def prepare_image(data):
    """Rotate using EXIF, fit Jev's limits, and export a baseline JPEG."""
    is_jpeg = data.startswith(b"\xff\xd8\xff")
    is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    if not (is_jpeg or is_png or is_webp):
        raise HTTPException(415, "Choose a JPEG, PNG, or WebP image.")
    try:
        # IMREAD_COLOR applies EXIF orientation and converts to three channels.
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unreadable image")
        height, width = image.shape[:2]
        if height * width > MAX_IMAGE_PIXELS:
            raise HTTPException(413, "Image is too large. Use a photo at or below 200 megapixels.")
        scale = min(1, 1568 / max(height, width))
        if scale < 1:
            image = cv2.resize(
                image, (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        height, width = image.shape[:2]
        pad_y, pad_x = max(0, 280 - height), max(0, 280 - width)
        image = cv2.copyMakeBorder(
            image, pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2,
            cv2.BORDER_CONSTANT, value=(24, 24, 24),
        )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Encoding failed")
        return encoded.tobytes()
    except (cv2.error, ValueError):
        raise HTTPException(422, "This image could not be decoded. Try a different photo.") from None

