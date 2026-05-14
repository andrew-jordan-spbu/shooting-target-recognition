"""
src/inference.py
────────────────
Batch inference on a folder (or single file) of photos and videos.
Outputs: annotated images/video + detections.csv + detections.json
"""
import csv
import json
from pathlib import Path

import cv2
import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _device() -> str:
    if torch.cuda.is_available():   return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def _process_image(model, path: Path, out_dir: Path, conf: float) -> list:
    frame   = cv2.imread(str(path))
    results = model(frame, conf=conf, verbose=False)
    dets    = []
    for result in results:
        for box in result.boxes:
            cls_id   = int(box.cls[0])
            cls_name = result.names[cls_id]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            dets.append({
                "file": path.name, "frame": 1,
                "class": cls_name,
                "score": int(cls_name) if cls_name.isdigit() else 0,
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [x1, y1, x2, y2],
            })
        cv2.imwrite(str(out_dir / f"ann_{path.name}"), result.plot())
    return dets


def _process_video(model, path: Path, out_dir: Path, conf: float) -> list:
    cap    = cv2.VideoCapture(str(path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h   = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(
        str(out_dir / f"ann_{path.stem}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
    )
    dets, frame_num = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1
        results = model(frame, conf=conf, verbose=False)
        for result in results:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = result.names[cls_id]
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                dets.append({
                    "file": path.name, "frame": frame_num,
                    "class": cls_name,
                    "score": int(cls_name) if cls_name.isdigit() else 0,
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": [x1, y1, x2, y2],
                })
            writer.write(result.plot())
    cap.release(); writer.release()
    return dets


def run(weights: str, input_path: str, output_dir: str, conf: float = 0.5) -> None:
    from ultralytics import YOLO

    device  = _device()
    model   = YOLO(weights)
    model.to(device)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp   = Path(input_path)
    files = list(inp.iterdir()) if inp.is_dir() else [inp]
    files = [f for f in files if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS]
    print(f"[INFO] {len(files)} file(s) | device={device} | conf={conf}")

    all_dets: list = []
    for f in sorted(files):
        print(f"  → {f.name}")
        if f.suffix.lower() in IMAGE_EXTS:
            all_dets.extend(_process_image(model, f, out_dir, conf))
        else:
            all_dets.extend(_process_video(model, f, out_dir, conf))

    # Save results
    (out_dir / "detections.json").write_text(json.dumps(all_dets, indent=2))
    if all_dets:
        with open(out_dir / "detections.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=all_dets[0].keys())
            w.writeheader(); w.writerows(all_dets)

    total = sum(d.get("score", 0) for d in all_dets)
    print(f"[OK]  Detections : {len(all_dets)}")
    print(f"[OK]  Total score: {total}")
    print(f"[OK]  Results    : {out_dir}")
