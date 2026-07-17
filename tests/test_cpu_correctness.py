from __future__ import annotations

import unittest

import cv2
import numpy as np

from core.preprocess_plan import (
    CpuPreprocessExecutor,
    Gaussian,
    Gray,
    Morphology,
    PreprocessPlan,
    Resize,
    Threshold,
)


class CpuStructuredInputTests(unittest.TestCase):
    executor = CpuPreprocessExecutor()
    reference_plan = PreprocessPlan(
        (
            Gray(),
            Gaussian(3),
            Threshold(110, invert=True),
            Morphology("open", kernel_size=3, iterations=1),
        ),
        name="structured_cpu_reference",
    )

    @staticmethod
    def _reference(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    @staticmethod
    def _structured_cases() -> dict[str, np.ndarray]:
        rng = np.random.default_rng(20260717)
        checker = ((np.indices((31, 47)).sum(axis=0) % 2) * 255).astype(np.uint8)
        boundary = np.zeros((31, 47), dtype=np.uint8)
        boundary[0, 0] = 255
        boundary[0, -1] = 255
        boundary[-1, 0] = 255
        boundary[-1, -1] = 255
        return {
            "random_bgr": rng.integers(0, 256, size=(31, 47, 3), dtype=np.uint8),
            "random_gray": rng.integers(0, 256, size=(31, 47), dtype=np.uint8),
            "black": np.zeros((31, 47, 3), dtype=np.uint8),
            "white": np.full((31, 47, 3), 255, dtype=np.uint8),
            "checkerboard": checker,
            "boundary_pixels": boundary,
        }

    def test_fixed_seed_structured_inputs_match_direct_opencv(self):
        for name, image in self._structured_cases().items():
            with self.subTest(name=name):
                original = image.copy()

                first = self.executor.execute(image, self.reference_plan)
                second = self.executor.execute(image, self.reference_plan)

                np.testing.assert_array_equal(first, self._reference(image))
                np.testing.assert_array_equal(second, first)
                np.testing.assert_array_equal(image, original)

    def test_odd_tiny_4k_stride_channels_and_roi_sizes(self):
        rng = np.random.default_rng(20260717)
        odd_bgr = rng.integers(0, 256, size=(9, 13, 3), dtype=np.uint8)
        odd_result = self.executor.execute(odd_bgr, PreprocessPlan((Gray(), Gaussian(3), Threshold(127))))
        self.assertEqual(odd_result.shape, (9, 13))

        tiny_bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
        tiny_result = self.executor.execute(tiny_bgr, PreprocessPlan((Gray(), Resize(5, 3, "nearest"))))
        self.assertEqual(tiny_result.shape, (3, 5))
        self.assertTrue(np.all(tiny_result == cv2.cvtColor(tiny_bgr, cv2.COLOR_BGR2GRAY)[0, 0]))

        image_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
        image_4k[0, 0] = (255, 255, 255)
        gray_4k = self.executor.execute(image_4k, PreprocessPlan((Gray(),)))
        self.assertEqual(gray_4k.shape, (2160, 3840))
        self.assertEqual(int(gray_4k[0, 0]), 255)

        base = rng.integers(0, 256, size=(33, 54, 3), dtype=np.uint8)
        non_contiguous = base[:, ::2, :]
        self.assertFalse(non_contiguous.flags.c_contiguous)
        stride_result = self.executor.execute(non_contiguous, self.reference_plan)
        np.testing.assert_array_equal(stride_result, self._reference(non_contiguous))

        gray_input = rng.integers(0, 256, size=(13, 17), dtype=np.uint8)
        gray_result = self.executor.execute(gray_input, PreprocessPlan((Gray(), Threshold(127))))
        self.assertEqual(gray_result.shape, gray_input.shape)

        roi_source = rng.integers(0, 256, size=(40, 50, 3), dtype=np.uint8)
        for height, width in ((1, 1), (3, 7), (15, 11), (32, 29)):
            with self.subTest(roi=(height, width)):
                roi = roi_source[:height, :width]
                result = self.executor.execute(roi, PreprocessPlan((Gray(), Resize(8, 6, "nearest"))))
                self.assertEqual(result.shape, (6, 8))


if __name__ == "__main__":
    unittest.main()
