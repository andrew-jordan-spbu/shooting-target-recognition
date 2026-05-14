"""
src/train.py
────────────
YOLOv8 training with heavy augmentation for angle invariance.
Supports CUDA (Colab), MPS (Mac M4) and CPU fallback.

Pre-training control checks:
  • Model vs dataset size compatibility (warns if large model + small data)
  • dropout recommendation for yolov8m/l on small datasets
"""
from pathlib import Path

import torch

# Thresholds for model-size warning
_MODEL_SIZE_WARN = {
    "yolov8n": 0,     # always fine
    "yolov8s": 0,     # fine up to any size
    "yolov8m": 300,   # need ≥ 300 train images
    "yolov8l": 500,
    "yolov8x": 800,
}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _count_train_images(data_yaml: str) -> int:
    import yaml
    try:
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        train_path = Path(cfg.get("train", ""))
        if train_path.exists():
            return len(list(train_path.glob("*.*")))
    except Exception:
        pass
    return 0


def _pre_train_checks(t: dict, data_yaml: str) -> None:
    """Print warnings before training starts."""
    model_name = t.get("model", "yolov8s.pt").replace(".pt", "")
    n_train    = _count_train_images(data_yaml)
    min_needed = _MODEL_SIZE_WARN.get(model_name, 0)

    print("\n[PRE-TRAIN CHECKS]")
    print(f"  Model     : {model_name}")
    print(f"  Train imgs: {n_train}")

    if min_needed > 0 and n_train < min_needed:
        print(f"\n  ⚠  WARNING: {model_name} needs ≥ {min_needed} training images,")
        print(f"     but only {n_train} found.")
        print(f"     Risk: overfitting — val metrics may look good but test will degrade.")
        print(f"     Options:")
        print(f"       • Downgrade to yolov8s.pt (no minimum requirement)")
        print(f"       • Add dropout: 0.1 in config.yaml → train section")
        print(f"       • Collect more labelled images")
    else:
        print(f"  ✓  Model-to-dataset size ratio is acceptable.")

    dropout = t.get("dropout", 0.0)
    if model_name in ("yolov8m", "yolov8l", "yolov8x") and dropout == 0.0:
        print(f"\n  ℹ  Tip: consider adding 'dropout: 0.1' to config.yaml → train")
        print(f"     section when using {model_name} on small datasets.")
    print("─" * 60)


def run(cfg: dict, data_yaml: str) -> str:
    """
    Train the model according to config.
    Returns the path to best.pt.
    """
    from ultralytics import YOLO

    t      = cfg["train"]
    out    = cfg["output"]
    device = get_device()

    print(f"[INFO] Device  : {device}")
    print(f"[INFO] Model   : {t.get('model', 'yolov8s.pt')}")
    print(f"[INFO] Epochs  : {t.get('epochs', 100)}")
    print(f"[INFO] imgsz   : {t.get('imgsz', 640)}")
    print(f"[INFO] Batch   : {t.get('batch', 16)}")
    print(f"[INFO] Data    : {data_yaml}")

    _pre_train_checks(t, data_yaml)
    print("─" * 60)

    model = YOLO(t.get("model", "yolov8s.pt"))

    model.train(
        data=data_yaml,
        epochs=t.get("epochs", 100),
        imgsz=t.get("imgsz", 640),
        batch=t.get("batch", 16),
        device=device,
        project=out["project"],
        name=out["run_name"],
        exist_ok=True,
        # Early stopping
        patience=t.get("patience", 25),
        save=True,
        save_period=10,
        plots=True,
        # Regularisation
        dropout=t.get("dropout", 0.0),
        weight_decay=t.get("weight_decay", 0.0005),
        # Augmentation
        degrees=t.get("degrees", 30.0),
        translate=t.get("translate", 0.15),
        scale=t.get("scale", 0.6),
        shear=t.get("shear", 5.0),
        perspective=t.get("perspective", 0.0005),
        flipud=t.get("flipud", 0.1),
        fliplr=t.get("fliplr", 0.5),
        mosaic=t.get("mosaic", 1.0),
        mixup=t.get("mixup", 0.15),
        copy_paste=t.get("copy_paste", 0.1),
        hsv_h=t.get("hsv_h", 0.015),
        hsv_s=t.get("hsv_s", 0.7),
        hsv_v=t.get("hsv_v", 0.4),
        # Optimiser
        optimizer=t.get("optimizer", "AdamW"),
        lr0=t.get("lr0", 0.001),
        lrf=t.get("lrf", 0.01),
        warmup_epochs=t.get("warmup_epochs", 3),
    )

    best = Path(out["project"]) / out["run_name"] / "weights" / "best.pt"
    print(f"\n[OK]  Best weights → {best}")
    return str(best)
