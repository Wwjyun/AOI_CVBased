from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class Tile:
    tile_id: str
    x: int
    y: int
    width: int
    height: int
    row: int
    col: int
    image: object
    metadata: dict | None = None


@dataclass(frozen=True)
class BinaryThresholdConfig:
    method: str = "global"
    threshold: int = 128
    max_value: int = 255
    invert: bool = False
    adaptive_block_size: int = 31
    adaptive_c: float = 5.0
    blur_size: int = 0
    morph_open_kernel: int = 0
    morph_open_iterations: int = 1
    morph_close_kernel: int = 0
    morph_close_iterations: int = 1

    @classmethod
    def from_dict(cls, config: dict | None) -> "BinaryThresholdConfig":
        config = config or {}
        return cls(
            method=str(config.get("method", "global")),
            threshold=int(config.get("threshold", 128)),
            max_value=int(config.get("max_value", 255)),
            invert=bool(config.get("invert", False)),
            adaptive_block_size=int(config.get("adaptive_block_size", 31)),
            adaptive_c=float(config.get("adaptive_c", 5.0)),
            blur_size=int(config.get("blur_size", 0)),
            morph_open_kernel=int(config.get("morph_open_kernel", 0)),
            morph_open_iterations=int(config.get("morph_open_iterations", 1)),
            morph_close_kernel=int(config.get("morph_close_kernel", 0)),
            morph_close_iterations=int(config.get("morph_close_iterations", 1)),
        )


@dataclass(frozen=True)
class ShapeFilterConfig:
    enabled_shapes: tuple[str, ...] = ("rectangle", "circle", "polygon")
    min_area: float = 1.0
    max_area: float = 0.0
    min_width: float = 0.0
    max_width: float = 0.0
    min_height: float = 0.0
    max_height: float = 0.0
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = 0.0
    min_radius: float = 0.0
    max_radius: float = 0.0
    min_circularity: float = 0.75
    polygon_min_vertices: int = 3
    polygon_max_vertices: int = 99
    approx_epsilon_ratio: float = 0.02
    subpixel_enabled: bool = True
    subpixel_window: int = 5
    crop_padding: int = 0

    @classmethod
    def from_dict(cls, config: dict | None) -> "ShapeFilterConfig":
        config = config or {}
        enabled_shapes = tuple(str(shape) for shape in config.get("enabled_shapes", ["rectangle", "circle", "polygon"]))
        return cls(
            enabled_shapes=enabled_shapes,
            min_area=float(config.get("min_area", 1.0)),
            max_area=float(config.get("max_area", 0.0)),
            min_width=float(config.get("min_width", 0.0)),
            max_width=float(config.get("max_width", 0.0)),
            min_height=float(config.get("min_height", 0.0)),
            max_height=float(config.get("max_height", 0.0)),
            min_aspect_ratio=float(config.get("min_aspect_ratio", 0.0)),
            max_aspect_ratio=float(config.get("max_aspect_ratio", 0.0)),
            min_radius=float(config.get("min_radius", 0.0)),
            max_radius=float(config.get("max_radius", 0.0)),
            min_circularity=float(config.get("min_circularity", 0.75)),
            polygon_min_vertices=int(config.get("polygon_min_vertices", 3)),
            polygon_max_vertices=int(config.get("polygon_max_vertices", 99)),
            approx_epsilon_ratio=float(config.get("approx_epsilon_ratio", 0.02)),
            subpixel_enabled=bool(config.get("subpixel_enabled", True)),
            subpixel_window=int(config.get("subpixel_window", 5)),
            crop_padding=int(config.get("crop_padding", 0)),
        )


class BinarySegmenter:
    def __init__(self, config: BinaryThresholdConfig):
        self.config = config

    def make_mask(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        if self.config.blur_size > 1:
            blur_size = self._odd_at_least(self.config.blur_size, 3)
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        method = self.config.method.lower()
        threshold_type = cv2.THRESH_BINARY_INV if self.config.invert else cv2.THRESH_BINARY
        if method == "otsu":
            _, mask = cv2.threshold(gray, 0, self.config.max_value, threshold_type | cv2.THRESH_OTSU)
        elif method in {"adaptive_mean", "adaptive_gaussian"}:
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if method == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            block_size = self._odd_at_least(self.config.adaptive_block_size, 3)
            mask = cv2.adaptiveThreshold(
                gray,
                self.config.max_value,
                adaptive_method,
                threshold_type,
                block_size,
                self.config.adaptive_c,
            )
        elif method == "global":
            _, mask = cv2.threshold(gray, self.config.threshold, self.config.max_value, threshold_type)
        else:
            raise ValueError(f"Unsupported threshold method: {self.config.method}")

        mask = self._morph(mask, cv2.MORPH_OPEN, self.config.morph_open_kernel, self.config.morph_open_iterations)
        mask = self._morph(mask, cv2.MORPH_CLOSE, self.config.morph_close_kernel, self.config.morph_close_iterations)
        return mask

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1

    @staticmethod
    def _morph(mask, operation: int, kernel_size: int, iterations: int):
        if kernel_size <= 1 or iterations <= 0:
            return mask
        kernel_size = BinarySegmenter._odd_at_least(kernel_size, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(mask, operation, kernel, iterations=iterations)


class ContourShapeAnalyzer:
    def __init__(self, config: ShapeFilterConfig):
        self.config = config

    def analyze(self, contour, gray_image) -> dict | None:
        area = float(cv2.contourArea(contour))
        if not self._within(area, self.config.min_area, self.config.max_area):
            return None

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            return None

        epsilon = self.config.approx_epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, width, height = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (rect_width, rect_height), angle = rect
        radius_center, radius = cv2.minEnclosingCircle(contour)
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        shape = self._classify(approx, circularity)
        if shape is None:
            return None

        normalized_width = float(max(rect_width, rect_height))
        normalized_height = float(min(rect_width, rect_height))
        aspect_ratio = normalized_width / normalized_height if normalized_height > 0 else 0.0
        if shape == "rectangle" and not self._passes_rectangle(normalized_width, normalized_height, aspect_ratio):
            return None
        if shape == "circle" and not self._passes_circle(float(radius), circularity):
            return None
        if shape == "polygon" and not self._passes_polygon(len(approx), float(width), float(height)):
            return None

        vertices = approx.reshape(-1, 2).astype(np.float32)
        if self.config.subpixel_enabled and len(vertices):
            vertices = self._refine_vertices(gray_image, vertices)

        return {
            "shape": shape,
            "area": area,
            "perimeter": perimeter,
            "bbox": [int(x), int(y), int(width), int(height)],
            "min_area_rect": {
                "center": [float(center_x), float(center_y)],
                "width": float(rect_width),
                "height": float(rect_height),
                "angle": float(angle),
                "aspect_ratio": float(aspect_ratio),
            },
            "circle": {
                "center": [float(radius_center[0]), float(radius_center[1])],
                "radius": float(radius),
                "circularity": circularity,
            },
            "vertices": [[float(point[0]), float(point[1])] for point in vertices],
        }

    def _classify(self, approx, circularity: float) -> str | None:
        vertex_count = len(approx)
        enabled = set(self.config.enabled_shapes)
        if "rectangle" in enabled and vertex_count == 4:
            return "rectangle"
        if "circle" in enabled and circularity >= self.config.min_circularity and vertex_count >= 6:
            return "circle"
        if "polygon" in enabled and self.config.polygon_min_vertices <= vertex_count <= self.config.polygon_max_vertices:
            return "polygon"
        return None

    def _passes_rectangle(self, width: float, height: float, aspect_ratio: float) -> bool:
        return (
            self._within(width, self.config.min_width, self.config.max_width)
            and self._within(height, self.config.min_height, self.config.max_height)
            and self._within(aspect_ratio, self.config.min_aspect_ratio, self.config.max_aspect_ratio)
        )

    def _passes_circle(self, radius: float, circularity: float) -> bool:
        return (
            self._within(radius, self.config.min_radius, self.config.max_radius)
            and circularity >= self.config.min_circularity
        )

    def _passes_polygon(self, vertex_count: int, width: float, height: float) -> bool:
        return (
            self.config.polygon_min_vertices <= vertex_count <= self.config.polygon_max_vertices
            and self._within(width, self.config.min_width, self.config.max_width)
            and self._within(height, self.config.min_height, self.config.max_height)
        )

    def _refine_vertices(self, gray_image, vertices):
        if gray_image is None:
            return vertices
        window = max(1, int(self.config.subpixel_window))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners = vertices.reshape(-1, 1, 2).copy()
        try:
            return cv2.cornerSubPix(gray_image, corners, (window, window), (-1, -1), criteria).reshape(-1, 2)
        except cv2.error:
            return vertices

    @staticmethod
    def _within(value: float, minimum: float, maximum: float) -> bool:
        if minimum and value < minimum:
            return False
        if maximum and value > maximum:
            return False
        return True


class Tiler:
    def __init__(self, width: int, height: int, overlap_x: int = 0, overlap_y: int = 0):
        if width <= 0 or height <= 0:
            raise ValueError("Tile width and height must be positive.")
        if overlap_x < 0 or overlap_y < 0:
            raise ValueError("Tile overlap cannot be negative.")
        if overlap_x >= width or overlap_y >= height:
            raise ValueError("Tile overlap must be smaller than tile size.")

        self.width = width
        self.height = height
        self.step_x = width - overlap_x
        self.step_y = height - overlap_y

    def iter_tiles(self, image) -> Iterator[Tile]:
        image_height, image_width = image.shape[:2]
        y_positions = self._positions(image_height, self.height, self.step_y)
        x_positions = self._positions(image_width, self.width, self.step_x)

        for row, y in enumerate(y_positions):
            for col, x in enumerate(x_positions):
                x2 = min(x + self.width, image_width)
                y2 = min(y + self.height, image_height)
                tile_image = image[y:y2, x:x2].copy()
                yield Tile(
                    tile_id=f"r{row:04d}_c{col:04d}",
                    x=x,
                    y=y,
                    width=x2 - x,
                    height=y2 - y,
                    row=row,
                    col=col,
                    image=tile_image,
                    metadata={"mode": "grid"},
                )

    @staticmethod
    def _positions(total: int, size: int, step: int) -> list[int]:
        if total <= size:
            return [0]

        positions = list(range(0, total - size + 1, step))
        last = total - size
        if positions[-1] != last:
            positions.append(last)
        return positions


class ContourTiler:
    def __init__(self, threshold: BinaryThresholdConfig, shapes: ShapeFilterConfig):
        self.segmenter = BinarySegmenter(threshold)
        self.analyzer = ContourShapeAnalyzer(shapes)
        self.shape_config = shapes

    @classmethod
    def from_config(cls, config: dict) -> "ContourTiler":
        return cls(
            threshold=BinaryThresholdConfig.from_dict(config.get("threshold")),
            shapes=ShapeFilterConfig.from_dict(config.get("shapes")),
        )

    def iter_tiles(self, image) -> Iterator[Tile]:
        mask = self.segmenter.make_mask(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_height, image_width = image.shape[:2]
        accepted_index = 0

        for contour_index, contour in enumerate(contours):
            metadata = self.analyzer.analyze(contour, gray)
            if metadata is None:
                continue

            x, y, width, height = metadata["bbox"]
            padding = self.shape_config.crop_padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image_width, x + width + padding)
            y2 = min(image_height, y + height + padding)
            if x2 <= x1 or y2 <= y1:
                continue

            tile_image = image[y1:y2, x1:x2].copy()
            shape = metadata["shape"]
            yield Tile(
                tile_id=f"{shape}_{accepted_index:04d}",
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1,
                row=accepted_index,
                col=0,
                image=tile_image,
                metadata={
                    "mode": "contour",
                    "contour_index": int(contour_index),
                    **metadata,
                },
            )
            accepted_index += 1


def create_tiler(tile_config: dict):
    mode = str(tile_config.get("mode", "grid")).lower()
    if mode == "grid":
        return Tiler(
            width=int(tile_config["width"]),
            height=int(tile_config["height"]),
            overlap_x=int(tile_config.get("overlap_x", 0)),
            overlap_y=int(tile_config.get("overlap_y", 0)),
        )
    if mode == "contour":
        return ContourTiler.from_config(tile_config)
    raise ValueError(f"Unsupported tile mode: {mode}")
