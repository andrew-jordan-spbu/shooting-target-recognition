from typing import Any, Dict, List

from calibration.zone_store import ZoneStore

_IOU_THRESHOLD = 0.40


class ShotScorer:
    """
    Determines which holes are *new* compared to the previous frame
    and resolves the score: calibration zones take priority over the model class.
    """

    def __init__(self) -> None:
        self._zone_store = ZoneStore()
        self._prev_holes: List[Dict[str, Any]] = []

    def find_new_holes(
        self, current_holes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        new_holes = []
        for hole in current_holes:
            if not self._is_duplicate(hole, self._prev_holes):
                resolved = dict(hole)
                resolved["score"] = self._resolve_score(hole)
                new_holes.append(resolved)
        self._prev_holes = current_holes
        return new_holes

    def _resolve_score(self, hole: Dict[str, Any]) -> int:
        """
        Score resolution priority:
          1. Calibration zones (always preferred — works with any model).
          2. Model class score (only if model was trained with 10 zone classes).
          3. 0 — hole outside all zones or no calibration data yet.
        """
        if self._zone_store.is_ready():
            cx, cy = hole["center"]
            zone_score = self._zone_store.score_for_point(cx, cy)
            if zone_score > 0:
                return zone_score
            # Hole detected but outside all calibration zones (e.g. miss)
            return 0
        # No calibration zones → fall back to model class score
        return hole.get("score", 0)

    def _is_duplicate(
        self, hole: Dict[str, Any], existing: List[Dict[str, Any]]
    ) -> bool:
        for ex in existing:
            if _iou(hole["box"], ex["box"]) > _IOU_THRESHOLD:
                return True
        return False

    def reset(self) -> None:
        self._prev_holes = []


def _iou(a: List[int], b: List[int]) -> float:
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
