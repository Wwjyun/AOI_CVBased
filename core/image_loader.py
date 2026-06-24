from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from core.logging_system import LogMixin


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageLoadError(RuntimeError):
    pass


class ImageLoader(LogMixin):
    def __init__(self, supported_extensions: set[str] | None = None):
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS
        Image.MAX_IMAGE_PIXELS = None

    def load_bgr(self, path: Path):
        image = self.load_rgb(path)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def load_rgb(self, path: Path):
        image_path = self._validate_path(path)
        try:
            self.logger.debug("Reading image with Pillow: %s", image_path)
            with Image.open(image_path) as pil_image:
                image = ImageOps.exif_transpose(pil_image).convert("RGB")
                array = np.array(image)
                self.logger.debug("Image read completed: %s shape=%s", image_path, array.shape)
                return array
        except Exception as exc:
            self.logger.exception("Image read failed: %s", image_path)
            raise ImageLoadError(f"Pillow failed to read image: {image_path}") from exc

    def _validate_path(self, path: Path) -> Path:
        image_path = Path(path)
        if image_path.suffix.lower() not in self.supported_extensions:
            raise ImageLoadError(f"Unsupported image extension: {image_path.suffix}")
        if not image_path.exists():
            raise ImageLoadError(f"Image does not exist: {image_path}")
        return image_path


def load_image(path: Path):
    return ImageLoader().load_bgr(path)
