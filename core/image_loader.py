from __future__ import annotations

from pathlib import Path

import cv2


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageLoadError(RuntimeError):
    pass


def load_image(path: Path):
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(f"Unsupported image extension: {image_path.suffix}")
    if not image_path.exists():
        raise ImageLoadError(f"Image does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"OpenCV failed to read image: {image_path}")
    return image
