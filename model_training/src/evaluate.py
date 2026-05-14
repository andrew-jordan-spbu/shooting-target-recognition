"""
src/evaluate.py
───────────────
Evaluate a trained model and save metrics to eval_metrics.json.

Control checks performed automatically:
  1. Dataset split size  — warns if test < 50 images (metrics unreliable)
  2. Val / Test gap      — warns if val_mAP > test_mAP + 0.15 (overfitting)
  3. Recall gate         — warns if test recall < 0.70 (priority metric)
  4. Regression guard    — warns if new model is worse than previous best
"""
import json
from pathlib import Path

import torch

# ── Quality targets ──────────────────────────────────────────────────────────
THRESHOLDS = {
    "mAP@50":    (0.80, 0.60),
    "mAP@50-95": (0.55, 0.40),
    "Precision":  (0.80, 0.65),
    "Recall":     (0.75, 0.55),
}

# Gap above which we flag overfitting (val_mAP − test_mAP)
_OVERFIT_GAP = 0.15

# Minimum test-set images for reliable metrics
_MIN_TEST_IMAGES = 50


def _status(value: float, good: float, ok: float) -> str:
    if value >= good: return "✓ GOOD"
    if value >= ok:   return "~ OK"
    return "✗ LOW"


def _bar(v: float, width: int = 20) -> str:
    return "█" * int(v * width)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_test_images(data_yaml: str) -> int:
    """Count images in the test split (or 0 if not found)."""
    import yaml
    try:
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        test_path = Path(cfg.get("test", ""))
        if test_path.exists():
            return len(list(test_path.glob("*.*")))
    except Exception:
        pass
    return 0


def _load_best_metrics(weights: str) -> dict | None:
    """Load the previously saved best metrics (baseline for regression guard)."""
    baseline = Path(weights).parent.parent / "best_metrics.json"
    if baseline.exists():
        try:
            return json.loads(baseline.read_text())
        except Exception:
            pass
    return None


def _save_best_metrics(weights: str, metrics: dict) -> None:
    """Persist metrics as the new baseline."""
    baseline = Path(weights).parent.parent / "best_metrics.json"
    baseline.write_text(json.dumps(metrics, indent=2))


def _eval_split(model, data_yaml: str, split: str, cfg: dict, device: str) -> dict:
    """Run YOLO validation on a given split and return metric dict."""
    ev = cfg["evaluate"]
    m  = model.val(
        data=data_yaml,
        imgsz=cfg["train"].get("imgsz", 640),
        split=split,
        device=device,
        conf=ev.get("conf", 0.5),
        iou=ev.get("iou", 0.5),
        plots=(split == "test"),
        save_json=(split == "test"),
        verbose=False,
    )
    return {
        "map50":     float(m.box.map50),
        "map5095":   float(m.box.map),
        "precision": float(m.box.mp),
        "recall":    float(m.box.mr),
        "per_class": {
            model.names.get(i, str(i)): float(ap)
            for i, ap in enumerate(m.box.maps or [])
        },
    }


# ── Main entry ────────────────────────────────────────────────────────────────

def run(cfg: dict, weights: str, data_yaml: str) -> dict:
    """Run validation, perform all control checks, return metrics dict."""
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Weights : {weights}")
    print(f"[INFO] Device  : {device}")

    model = YOLO(weights)

    # ── 1. Dataset size check ────────────────────────────────────────────────
    n_test = _count_test_images(data_yaml)
    print(f"\n[CHECK 1/4] Test-set size: {n_test} images")
    if n_test < _MIN_TEST_IMAGES:
        print(f"  ⚠  WARNING: only {n_test} test images (< {_MIN_TEST_IMAGES}).")
        print(f"     Metrics have high variance — collect more labelled data for")
        print(f"     reliable evaluation.")
    else:
        print(f"  ✓  Sufficient test images.")

    # ── 2. Evaluate on TEST split ────────────────────────────────────────────
    print(f"\n═══ Test split  (conf={cfg['evaluate'].get('conf', 0.5)}) ═══════════════════")
    test_m = _eval_split(model, data_yaml, "test", cfg, device)
    _print_metrics(test_m, model.names)

    # ── 3. Evaluate on VAL split (for overfitting check) ─────────────────────
    print(f"\n═══ Val split  (overfitting check) ═══════════════════════════")
    val_m = _eval_split(model, data_yaml, "val", cfg, device)
    _print_metrics(val_m, model.names)

    gap = val_m["map50"] - test_m["map50"]
    print(f"\n[CHECK 2/4] Val / Test mAP@50 gap: {gap:+.3f}")
    if gap > _OVERFIT_GAP:
        print(f"  ⚠  OVERFITTING DETECTED (gap={gap:.3f} > threshold={_OVERFIT_GAP})")
        print(f"     Val mAP={val_m['map50']:.3f}  Test mAP={test_m['map50']:.3f}")
        print(f"     Recommendations:")
        print(f"       • Use a smaller model (e.g. yolov8s instead of yolov8m)")
        print(f"       • Add dropout: dropout: 0.1 in config.yaml → train section")
        print(f"       • Collect more training images (target ≥ 500 per split)")
        print(f"       • Reduce imgsz to 640 if dataset was annotated at low resolution")
    elif gap > 0.05:
        print(f"  ~ Mild gap ({gap:.3f}) — monitor if it grows in future runs.")
    else:
        print(f"  ✓  No overfitting detected.")

    # ── 4. Recall gate ────────────────────────────────────────────────────────
    recall = test_m["recall"]
    print(f"\n[CHECK 3/4] Test Recall: {recall:.3f}")
    if recall < 0.70:
        print(f"  ⚠  LOW RECALL ({recall:.3f} < 0.70) — model misses too many holes.")
        print(f"     Recommendations:")
        print(f"       • Lower inference conf threshold to 0.25–0.30")
        print(f"       • Increase training epochs (patience=50)")
        print(f"       • Add more images with holes in outer (white) zones")
    elif recall < 0.75:
        print(f"  ~ Recall borderline ({recall:.3f}) — consider more data or lower conf.")
    else:
        print(f"  ✓  Recall meets target.")

    # ── 5. Regression guard ───────────────────────────────────────────────────
    prev = _load_best_metrics(weights)
    print(f"\n[CHECK 4/4] Regression vs previous best model")
    if prev is None:
        print(f"  (no baseline found — this is the first evaluated run)")
        _save_best_metrics(weights, test_m)
        print(f"  ✓  Saved as new baseline.")
    else:
        prev_map   = prev.get("map50", 0)
        prev_recall = prev.get("recall", 0)
        delta_map   = test_m["map50"] - prev_map
        delta_recall = test_m["recall"] - prev_recall
        print(f"  mAP@50 :  {prev_map:.3f} → {test_m['map50']:.3f}  ({delta_map:+.3f})")
        print(f"  Recall :  {prev_recall:.3f} → {test_m['recall']:.3f}  ({delta_recall:+.3f})")
        if test_m["map50"] >= prev_map and test_m["recall"] >= prev_recall - 0.02:
            print(f"  ✓  New model is better or equal — baseline updated.")
            _save_best_metrics(weights, test_m)
        elif test_m["map50"] >= prev_map - 0.02:
            print(f"  ~ Marginal change — baseline updated.")
            _save_best_metrics(weights, test_m)
        else:
            print(f"  ✗  REGRESSION DETECTED — new model is WORSE than previous best!")
            print(f"     Previous best.pt was NOT replaced in best_metrics.json.")
            print(f"     Action: revert config.yaml to previous settings and retrain,")
            print(f"     OR keep the previous best.pt for inference.")

    # ── Save full metrics ─────────────────────────────────────────────────────
    result = {
        "test":  test_m,
        "val":   val_m,
        "gap_val_test": round(gap, 4),
        # flat keys for backward compatibility
        "map50":     test_m["map50"],
        "map5095":   test_m["map5095"],
        "precision": test_m["precision"],
        "recall":    test_m["recall"],
        "per_class": test_m["per_class"],
    }
    out_path = Path(weights).parent.parent / "eval_metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Metrics saved → {out_path}")
    return result


def _print_metrics(m: dict, names: dict) -> None:
    rows = [
        ("mAP@50",    m["map50"]),
        ("mAP@50-95", m["map5095"]),
        ("Precision",  m["precision"]),
        ("Recall",     m["recall"]),
    ]
    for name, val in rows:
        good, ok = THRESHOLDS[name]
        print(f"  {name:<14} {val:.4f}  {_bar(val):<20}  {_status(val, good, ok)}")
    if m["per_class"]:
        print("  Per-class mAP@50:")
        for cls, ap in m["per_class"].items():
            print(f"    {cls:<12} {ap:.4f}  {_bar(ap)}")
