"""
src/dataset.py
──────────────
Legacy single-dataset utilities.  Kept for backward compatibility and
diagnostics; the production pipeline now uses src/merge.py to combine
multiple Roboflow datasets into one.

The functions here can still be used to inspect or validate a single
downloaded dataset (e.g. for ad-hoc debugging).
"""
import sys
from pathlib import Path

import yaml


# ── Expected scored classes (subset of dataset classes) ──────────────────────
SCORE_CLASSES = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}


def download(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    data_dir: Path,
) -> str:
    """Download dataset in YOLOv8 format. Returns path to extracted directory."""
    if not api_key:
        print("[ERROR] Roboflow API key is empty.")
        print("  Get a free key at https://roboflow.com → Account settings → API keys")
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow not installed. Run: pip install roboflow")
        sys.exit(1)

    data_dir = Path(data_dir)
    location = data_dir / "raw"

    # Skip download if already present
    if (location / "data.yaml").exists():
        print(f"[INFO] Dataset already present at {location} — skipping download.")
        return str(location)

    print(f"[INFO] Downloading {workspace}/{project} v{version} …")
    rf      = Roboflow(api_key=api_key)
    proj    = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(
        model_format="yolov8",
        location=str(location),
        overwrite=False,
    )
    print(f"[OK]  Downloaded → {dataset.location}")
    return dataset.location


def inspect(location: str) -> dict:
    """Return dataset info: class names, split sizes."""
    p         = Path(location)
    yaml_path = p / "data.yaml"

    if not yaml_path.exists():
        return {"nc": 0, "names": [], "splits": {}}

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    info: dict = {
        "nc":     cfg.get("nc", 0),
        "names":  cfg.get("names", []),
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        img_dir = p / split / "images"
        n = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        info["splits"][split] = n

    return info


def validate_classes(info: dict) -> None:
    """Warn if expected scored classes are missing from the dataset."""
    names   = set(str(n) for n in info.get("names", []))
    found   = SCORE_CLASSES & names
    missing = SCORE_CLASSES - names

    print(f"[INFO] Scored zone classes found  : {sorted(found)}")
    if missing:
        print(f"[WARN] Scored zone classes missing: {sorted(missing)}")
        print("       Scoring for missing classes will fall back to calibration zones.")
    else:
        print("[OK]  All 10 scoring zone classes present.")

    # ── Split size sanity check ───────────────────────────────────────────────
    MIN_IMAGES = {"train": 200, "valid": 50, "test": 50}
    splits = info.get("splits", {})
    print("\n[CHECK] Dataset split sizes:")
    any_small = False
    for split, min_n in MIN_IMAGES.items():
        n = splits.get(split, 0)
        ok = "✓" if n >= min_n else "⚠ "
        print(f"  {ok}  {split:6s}: {n:4d} images  (recommended ≥ {min_n})")
        if n < min_n:
            any_small = True
    if any_small:
        print()
        print("  ⚠  Some splits are too small. Consequences:")
        print("     • val/test metrics will have high variance")
        print("     • overfitting risk increases with large models (yolov8m/l)")
        print("     Recommendation: collect more labelled images, or use yolov8s.")


def patch_yaml_paths(location: str) -> str:
    """
    Rewrite data.yaml with absolute paths so training works from any CWD.
    Roboflow datasets use varying relative path formats (e.g. '../train/images',
    'train/images', 'train') — we normalise all of them to the canonical
    absolute path: {location}/{split}/images
    Saves as data_abs.yaml and returns its path.
    """
    yaml_path = Path(location) / "data.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    abs_loc = Path(location).resolve()
    for key in ("train", "valid", "test"):
        if key in cfg:
            cfg[key] = str(abs_loc / key / "images")

    patched = abs_loc / "data_abs.yaml"
    with open(patched, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    print(f"[OK]  Absolute-path YAML → {patched}")
    return str(patched)
