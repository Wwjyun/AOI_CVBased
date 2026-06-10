from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


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
