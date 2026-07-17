from __future__ import annotations

import cv2


class TilePreprocessCache:
    """Tile-scoped reusable CPU preprocessing values shared by detectors."""

    def __init__(self, source) -> None:
        self.source = source
        self._gray = None

    def gray(self):
        if self._gray is None:
            self._gray = (
                cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)
                if self.source.ndim == 3 else self.source
            )
        return self._gray
