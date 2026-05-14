import json
from pathlib import Path
from typing import Optional

from calibration.auto_detect import get_score_from_zones, load_zones, save_zones
from config.settings import CALIBRATION_PATH


class ZoneStore:
    """Thread-safe in-memory cache of calibration zone data backed by a JSON file."""

    def __init__(self, path: Path = CALIBRATION_PATH) -> None:
        self.path = path
        self._data: Optional[dict] = None

    def load(self) -> Optional[dict]:
        self._data = load_zones(self.path)
        return self._data

    def save(self, data: dict) -> None:
        self._data = data
        save_zones(data, self.path)

    def set(self, data: dict) -> None:
        """Update in-memory zones without touching the file (used by the pipeline)."""
        self._data = data

    def get_zones(self) -> list:
        if self._data is None:
            self.load()
        return self._data.get("zones", []) if self._data else []

    def score_for_point(self, x: float, y: float) -> int:
        return get_score_from_zones(x, y, self.get_zones())

    def is_ready(self) -> bool:
        return bool(self.get_zones())
