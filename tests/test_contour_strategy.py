from __future__ import annotations

import unittest

import cv2
import numpy as np


class ConnectedComponentsSemanticTests(unittest.TestCase):
    def test_component_pixel_area_does_not_match_contour_geometry_area(self):
        binary = np.zeros((120, 160), dtype=np.uint8)
        cv2.rectangle(binary, (20, 30), (59, 69), 255, thickness=cv2.FILLED)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        self.assertEqual(len(contours), 1)
        self.assertEqual(component_count - 1, 1)
        self.assertEqual(cv2.boundingRect(contours[0]), tuple(stats[1, :4]))
        self.assertEqual(cv2.contourArea(contours[0]), 1521.0)
        self.assertEqual(int(stats[1, cv2.CC_STAT_AREA]), 1600)

    def test_list_mode_hole_contour_cannot_be_represented_by_components(self):
        binary = np.zeros((120, 160), dtype=np.uint8)
        cv2.circle(binary, (80, 60), 35, 255, thickness=cv2.FILLED)
        cv2.circle(binary, (80, 60), 15, 0, thickness=cv2.FILLED)

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        component_count, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        self.assertEqual(len(contours), 2)
        self.assertEqual(component_count - 1, 1)


if __name__ == "__main__":
    unittest.main()
