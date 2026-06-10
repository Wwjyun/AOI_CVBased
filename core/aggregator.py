from __future__ import annotations


class Aggregator:
    def __init__(self, decision_config: dict):
        self.decision_config = decision_config or {}

    def aggregate(self, tile_results: list[dict]) -> dict:
        ng_tiles = []
        defect_count = 0
        detector_ng_counts: dict[str, int] = {}

        for tile_result in tile_results:
            tile_pass = True
            for detector_result in tile_result["detectors"]:
                defects = detector_result.get("defects", [])
                defect_count += len(defects)
                if not detector_result.get("pass", True):
                    tile_pass = False
                    detector_id = detector_result["detector_id"]
                    detector_ng_counts[detector_id] = detector_ng_counts.get(detector_id, 0) + 1
            tile_result["result"] = "PASS" if tile_pass else "NG"
            if not tile_pass:
                ng_tiles.append(tile_result)

        max_ng_count = int(self.decision_config.get("max_ng_count", 0))
        final_result = "PASS" if len(ng_tiles) <= max_ng_count else "NG"

        return {
            "final_result": final_result,
            "ng_tiles": ng_tiles,
            "summary": {
                "tile_count": len(tile_results),
                "ng_count": len(ng_tiles),
                "defect_count": defect_count,
                "detector_ng_counts": detector_ng_counts,
            },
        }
