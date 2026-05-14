"""
src/merge.py
────────────
Download multiple Roboflow shooting-target datasets and merge them into a
single unified dataset with one class ("bulletholes").

Each source dataset has its own class taxonomy (KNSA: 1 class, BAT: 13 classes,
Andrew Crook: 1 class, …).  We use a per-source "remap" table to:
  • collapse every hole-like class to "bulletholes" (class_id = 0)
  • drop classes that are not bullet holes (Target, black_contour, etc.)
The merged dataset has the canonical YOLOv8 layout:

    {output_dir}/
        train/{images,labels}/
        valid/{images,labels}/
        test/{images,labels}/
        data.yaml         # nc: 1, names: ['bulletholes']

File names are prefixed with the source name (e.g. "knsa_xxx.jpg") to avoid
collisions across datasets.
"""
from __future__ import annotations

import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


# ── Roboflow split directory aliases ──────────────────────────────────────────
# Some datasets call the validation split "valid", others "val".  We handle both.
_SPLITS = ("train", "valid", "test")
_VALID_ALIASES = ("valid", "val")


# ── Public API ────────────────────────────────────────────────────────────────

def merge_datasets(
    sources_cfg: list[dict[str, Any]],
    output_dir: Path,
    api_key: str,
    raw_dir: Path,
) -> str:
    """
    Download and merge all sources listed in `sources_cfg` into one dataset.

    Args:
        sources_cfg: list of dicts, each with keys
            name, workspace, project, version, remap (dict[str, str|None])
        output_dir:  where the merged dataset will be written.
        api_key:     Roboflow API key.
        raw_dir:     directory for individual raw downloads.

    Returns:
        Absolute path to the merged data.yaml.
    """
    if (output_dir / "data.yaml").exists():
        print(f"[INFO] Merged dataset already present at {output_dir} — skipping merge.")
        return str((output_dir.resolve() / "data.yaml"))

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create output split directories
    for split in _SPLITS:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # stats[source_name][split] = number of images contributed

    print("\n═══ Merging datasets ════════════════════════════════════════")
    for src in sources_cfg:
        name = src["name"]
        print(f"\n── Source: {name} ─────────────────────────────────")
        loc = _download_one(api_key, src, raw_dir)
        if loc is None:
            stats[name] = {sp: 0 for sp in _SPLITS}
            continue
        per_split = _ingest_source(
            src_root=Path(loc),
            src_yaml=Path(loc) / "data.yaml",
            remap=src["remap"],
            prefix=name,
            output_dir=output_dir,
        )
        for split, n in per_split.items():
            stats[name][split] += n

    # ── Write merged data.yaml ────────────────────────────────────────────────
    abs_out = output_dir.resolve()
    merged_yaml_path = abs_out / "data.yaml"
    merged_cfg = {
        "train":   str(abs_out / "train" / "images"),
        "val":     str(abs_out / "valid" / "images"),
        "test":    str(abs_out / "test"  / "images"),
        "nc":      1,
        "names":   ["bulletholes"],
    }
    with open(merged_yaml_path, "w") as f:
        yaml.dump(merged_cfg, f, allow_unicode=True, sort_keys=False)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n═══ Merge summary ═══════════════════════════════════════════")
    header = f"  {'Source':<20s} {'train':>8s} {'valid':>8s} {'test':>8s} {'total':>8s}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    totals = defaultdict(int)
    for src in sources_cfg:
        name = src["name"]
        s = stats[name]
        row_total = s["train"] + s["valid"] + s["test"]
        for sp in _SPLITS:
            totals[sp] += s[sp]
        totals["total"] += row_total
        print(f"  {name:<20s} {s['train']:>8d} {s['valid']:>8d} {s['test']:>8d} {row_total:>8d}")
    print("  " + "─" * (len(header) - 2))
    print(f"  {'MERGED':<20s} {totals['train']:>8d} {totals['valid']:>8d} "
          f"{totals['test']:>8d} {totals['total']:>8d}")
    print(f"\n✓ Merged dataset → {merged_yaml_path}")
    return str(merged_yaml_path)


def verify_merged(merged_yaml: str) -> dict[str, int]:
    """Sanity-check the merged dataset and return per-split counts."""
    p = Path(merged_yaml)
    with open(p) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("nc") != 1:
        print(f"[ERROR] Merged dataset must have exactly 1 class, got {cfg.get('nc')}")
        sys.exit(1)
    if cfg.get("names") != ["bulletholes"]:
        print(f"[ERROR] Class name must be 'bulletholes', got {cfg.get('names')}")
        sys.exit(1)

    counts = {}
    base = p.parent
    for split, alias in (("train", "train"), ("valid", "valid"), ("test", "test")):
        img_dir = base / alias / "images"
        lbl_dir = base / alias / "labels"
        n_img = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        counts[split] = n_img
        print(f"  {split:6s}: {n_img:4d} images, {n_lbl:4d} label files")
        if n_img != n_lbl:
            print(f"  [WARN] Image/label mismatch in {split} ({n_img} vs {n_lbl})")
    return counts


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_version(proj, requested) -> int:
    """
    Resolve `requested` (int | None | 'latest') into a concrete version number.
    On failure, lists all available versions for the user to inspect.
    """
    versions = proj.versions()
    if not versions:
        raise RuntimeError("No versions available for this project")
    available = sorted(int(v.version.split("/")[-1]) for v in versions)
    latest = available[-1]

    if requested in (None, "latest"):
        return latest

    if int(requested) in available:
        return int(requested)

    print(f"  [WARN] Requested version {requested} not found.")
    print(f"  [WARN] Available versions: {available}. Falling back to latest: {latest}")
    return latest


def _download_one(api_key: str, src: dict[str, Any], raw_dir: Path) -> str | None:
    """
    Download a single Roboflow dataset; return its location on disk, or None
    on failure (so the caller can skip this source instead of aborting).
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow not installed. Run: pip install roboflow")
        sys.exit(1)

    name      = src["name"]
    workspace = src["workspace"]
    project   = src["project"]
    requested = src.get("version", "latest")
    location  = raw_dir / name

    if (location / "data.yaml").exists():
        print(f"  [INFO] Already downloaded → {location}")
        return str(location)

    try:
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        version = _resolve_version(proj, requested)
        print(f"  [INFO] Downloading {workspace}/{project} v{version} …")
        ds = proj.version(version).download(
            model_format="yolov8",
            location=str(location),
            overwrite=False,
        )
        return ds.location
    except Exception as exc:
        print(f"  [ERROR] Failed to download {name}: {exc}")
        print(f"  [INFO]  Skipping this source — pipeline continues with others.")
        return None


def _ingest_source(
    src_root: Path,
    src_yaml: Path,
    remap: dict[str, Any],
    prefix: str,
    output_dir: Path,
) -> dict[str, int]:
    """
    Copy images from src_root into the merged dataset, rewriting label files
    according to the remap table.  Returns per-split image counts.
    """
    with open(src_yaml) as f:
        cfg = yaml.safe_load(f)
    src_class_names: list[str] = cfg.get("names", [])

    # Build src_class_id → output_class_id (always 0) or None to skip
    keep_ids: dict[int, int] = {}
    for idx, cls_name in enumerate(src_class_names):
        if cls_name in remap and remap[cls_name] is not None:
            keep_ids[idx] = 0   # all classes collapse to "bulletholes" (id 0)
    if not keep_ids:
        print(f"  ⚠⚠⚠  CRITICAL: no classes from '{prefix}' matched the remap table!")
        print(f"        Source classes: {src_class_names}")
        print(f"        Remap keys:     {list(remap.keys())}")
        print(f"        ALL annotations from this source will be DROPPED.")

    counts: dict[str, int] = {sp: 0 for sp in _SPLITS}
    annotations_kept = 0
    for split in _SPLITS:
        # Roboflow may name the validation split "valid" or "val" – find both
        src_split_dir = None
        for alias in (_VALID_ALIASES if split == "valid" else (split,)):
            if (src_root / alias / "images").exists():
                src_split_dir = src_root / alias
                break
        if src_split_dir is None:
            continue

        src_img_dir = src_split_dir / "images"
        src_lbl_dir = src_split_dir / "labels"
        out_img_dir = output_dir / split / "images"
        out_lbl_dir = output_dir / split / "labels"

        for img_path in src_img_dir.iterdir():
            if not img_path.is_file():
                continue

            # Copy image with prefixed name
            new_name = f"{prefix}_{img_path.name}"
            shutil.copy2(img_path, out_img_dir / new_name)

            # Rewrite label file (drop unkept classes, remap kept ones to 0)
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            new_lbl_name = f"{prefix}_{img_path.stem}.txt"
            new_lines = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        try:
                            cls_id = int(parts[0])
                        except (ValueError, IndexError):
                            continue
                        if cls_id in keep_ids:
                            parts[0] = str(keep_ids[cls_id])
                            new_lines.append(" ".join(parts))
            # Always write a label file (empty if no kept annotations)
            with open(out_lbl_dir / new_lbl_name, "w") as f:
                f.write("\n".join(new_lines))

            counts[split] += 1
            annotations_kept += len(new_lines)

    print(f"  [OK] {prefix}: {counts['train']}+{counts['valid']}+{counts['test']} "
          f"= {sum(counts.values())} images, {annotations_kept} annotations kept")
    if annotations_kept == 0 and sum(counts.values()) > 0:
        print(f"  ⚠⚠⚠  ZERO annotations kept — training will not work!")
        print(f"        Inspect remap table for '{prefix}' and update class names.")
    return counts
