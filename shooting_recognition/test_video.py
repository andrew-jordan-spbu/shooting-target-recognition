"""
test_video.py — standalone CLI test for the video → motion → detection pipeline.
─────────────────────────────────────────────────────────────────────────────
Runs the same pipeline as the Flask app, but without Flask, PostgreSQL or the
calibration step.  Useful for verifying model + motion detector on real
video streams (USB camera, IP/RTSP, or a video file).

Usage:
    python test_video.py --source 0                        # USB camera index 0
    python test_video.py --source rtsp://192.168.1.100/    # IP camera
    python test_video.py --source ~/Downloads/range.mp4    # video file
    python test_video.py --source 0 --conf 0.30 --save     # save annotated frames
    python test_video.py --source 0 --no-motion            # disable motion gate

Press Ctrl+C to stop.  At exit, prints aggregate statistics.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np


# ── Source helpers ───────────────────────────────────────────────────────────

def _open_source(source: str) -> tuple[cv2.VideoCapture, str]:
    """
    Resolve `source` to (cv2.VideoCapture, kind):
      • integer string  → USB camera index
      • starts with rtsp/http → IP stream
      • path to existing file → local file
    """
    src = source.strip()
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        kind = f"USB camera #{src}"
    elif src.startswith(("rtsp://", "http://", "https://")):
        cap = cv2.VideoCapture(src)
        kind = f"IP stream {src}"
    else:
        if not Path(src).exists():
            print(f"[ERROR] File not found: {src}")
            sys.exit(1)
        cap = cv2.VideoCapture(src)
        kind = f"file {src}"

    if not cap.isOpened():
        print(f"[ERROR] Failed to open source: {src}")
        sys.exit(1)
    return cap, kind


def _annotate(frame: np.ndarray, holes: list) -> np.ndarray:
    """Draw bounding boxes + confidence on a copy of `frame`."""
    out = frame.copy()
    for h in holes:
        x1, y1, x2, y2 = h["box"]
        cx, cy = h["center"]
        conf = h.get("confidence", 0.0)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(out, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(
            out, f"{conf:.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )
    return out


# ── Main loop ────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    from config.settings import MODEL_PATH, DEVICE, OUTPUT_FOLDER
    from motion.detector import MotionDetector
    from recognition.model import TargetModel

    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("  Copy your trained best.pt to recognition/weights/best.pt")
        sys.exit(1)

    save_dir: Path | None = None
    if args.save:
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_dir = OUTPUT_FOLDER.parent / "detection_frames" / f"test_video_{ts}"
        save_dir.mkdir(parents=True, exist_ok=True)

    cap, kind = _open_source(args.source)
    print(f"[INFO] Source     : {kind}")
    print(f"[INFO] Model      : {MODEL_PATH}")
    print(f"[INFO] Device     : {DEVICE}")
    print(f"[INFO] Conf thresh: {args.conf}")
    print(f"[INFO] Motion gate: {'OFF' if args.no_motion else 'ON'}")
    if save_dir:
        print(f"[INFO] Save frames: {save_dir}")
    print("─" * 60)
    print("Press Ctrl+C to stop.")
    print("─" * 60)

    model = TargetModel()
    motion = MotionDetector()

    # Per-frame inference uses MODEL_CONFIDENCE from settings; override at call-time
    # by monkey-patching the env (the model reads the constant once at load).
    # We re-call YOLO directly to honour the CLI --conf.
    from ultralytics import YOLO
    yolo = YOLO(str(MODEL_PATH))
    yolo.to(DEVICE)

    stats = {
        "frames":          0,   # total frames pulled from source
        "motion_frames":   0,   # frames where motion fired
        "infer_frames":    0,   # frames sent to YOLO
        "holes_detected":  0,   # total holes across frames
        "saved_frames":    0,
    }
    started_at = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("\n[INFO] End of stream.")
                break

            stats["frames"] += 1

            # Motion gate (optional)
            if not args.no_motion:
                if not motion.has_changed(frame):
                    continue
                stats["motion_frames"] += 1

            # Inference
            stats["infer_frames"] += 1
            results = yolo(frame, conf=args.conf, device=DEVICE, verbose=False)
            holes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    holes.append({
                        "class": result.names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]), 3),
                        "box": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                    })

            if holes:
                stats["holes_detected"] += len(holes)
                ts = time.strftime("%H:%M:%S")
                confs = ", ".join(f"{h['confidence']:.2f}" for h in holes)
                print(f"  [{ts}] frame #{stats['frames']:>5d}  "
                      f"→ {len(holes)} hole(s)  conf=[{confs}]")

                # Persist annotated frame
                if save_dir:
                    annotated = _annotate(frame, holes)
                    name = f"frame_{stats['frames']:06d}_{len(holes)}holes.jpg"
                    cv2.imwrite(str(save_dir / name), annotated)
                    stats["saved_frames"] += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()

    elapsed = time.monotonic() - started_at
    fps = stats["frames"] / elapsed if elapsed > 0 else 0.0

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n═══ TEST VIDEO STATISTICS ══════════════════════════════════")
    print(f"  Elapsed time      : {elapsed:7.1f} s")
    print(f"  Frames processed  : {stats['frames']:7d}    ({fps:.1f} FPS avg)")
    print(f"  Motion-changed    : {stats['motion_frames']:7d}")
    print(f"  Sent to YOLO      : {stats['infer_frames']:7d}")
    print(f"  Holes detected    : {stats['holes_detected']:7d}")
    if save_dir:
        print(f"  Frames saved      : {stats['saved_frames']:7d}  → {save_dir}")
    print("════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone test for the video → motion → detection pipeline.",
    )
    parser.add_argument(
        "--source", required=True,
        help="USB cam index ('0'), RTSP URL ('rtsp://…'), or path to a video file.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Confidence threshold for YOLO (default: 0.35).",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated frames where holes were detected.",
    )
    parser.add_argument(
        "--no-motion", action="store_true",
        help="Disable motion gating (run YOLO on every frame).",
    )
    args = parser.parse_args()

    run(args)
    os._exit(0)   # force-kill background torch threads
