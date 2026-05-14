import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Blueprint, Response, jsonify, redirect,
    render_template, request, send_file, url_for,
)
from werkzeug.utils import secure_filename

from calibration.auto_detect import detect_target, save_zones
from calibration.zone_store import ZoneStore
from capture.camera import CameraSource
from capture.file_source import FileSource
from capture.ip_stream import IPStreamSource
from config.settings import (
    ALLOWED_IMAGE_EXT, ALLOWED_VIDEO_EXT,
    CALIBRATION_FRAMES, OUTPUT_FOLDER, UPLOAD_FOLDER,
)
from database.models import (
    add_shot, create_session, finish_session,
    get_session, get_session_shots, update_session_zones,
)
from motion.detector import MotionDetector
from recognition.model import TargetModel
from recognition.scorer import ShotScorer
from reporting.csv_report import export_csv
from reporting.pdf_report import export_pdf

bp = Blueprint("main", __name__)

# ---------------------------------------------------------------------------
# Global pipeline state (single-user prototype)
# ---------------------------------------------------------------------------
_state: dict = {
    "running": False,
    "calibration_done": False,
    "session_id": None,
    "player_name": "",
    "shot_count": 0,
    "total_score": 0,
    "events": [],
    "latest_frame": None,
    "calibration_frame": None,
    "calibration_zones": None,
}
_lock = threading.Lock()


def _allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/start", methods=["POST"])
def start():
    source_type = request.form.get("source_type", "usb")
    player_name = (request.form.get("player_name") or "Player").strip()

    with _lock:
        if _state["running"]:
            return jsonify({"error": "Session already running"}), 400

    source: tuple
    if source_type == "usb":
        source = ("usb", int(request.form.get("camera_index", 0)))
    elif source_type == "ip":
        source = ("ip", request.form.get("rtsp_url", ""))
    elif source_type == "file":
        f = request.files.get("video_file")
        if not f or not _allowed_file(f.filename):
            return jsonify({"error": "Invalid or missing file"}), 400
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        fname = secure_filename(f.filename)
        fpath = UPLOAD_FOLDER / fname
        f.save(str(fpath))
        source = ("file", str(fpath))
    else:
        return jsonify({"error": "Unknown source type"}), 400

    session_id = create_session(player_name)
    with _lock:
        _state.update({
            "running": True,
            "calibration_done": False,
            "session_id": session_id,
            "player_name": player_name,
            "shot_count": 0,
            "total_score": 0,
            "events": [],
            "latest_frame": None,
            "calibration_frame": None,
            "calibration_zones": None,
        })

    t = threading.Thread(target=_run_pipeline, args=(source,), daemon=True)
    t.start()
    return redirect(url_for("main.calibrate_view"))


# ---------------------------------------------------------------------------
# Calibration routes
# ---------------------------------------------------------------------------

@bp.route("/calibrate")
def calibrate_view():
    return render_template("calibrate.html")


@bp.route("/calibrate/frame")
def calibration_frame():
    with _lock:
        frame = _state.get("calibration_frame")
    if frame is None:
        return "", 204
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@bp.route("/calibrate/zones", methods=["GET"])
def get_calibration_zones():
    with _lock:
        zones = _state.get("calibration_zones")
    return jsonify(zones or {})


@bp.route("/calibrate/zones", methods=["POST"])
def save_calibration_zones():
    data = request.get_json(force=True)
    if not data or "zones" not in data:
        return jsonify({"error": "Invalid zone data"}), 400

    store = ZoneStore()
    store.save(data)

    with _lock:
        session_id = _state["session_id"]
        _state["calibration_zones"] = data
        _state["calibration_done"] = True

    if session_id:
        update_session_zones(session_id, data)

    return jsonify({"ok": True})


@bp.route("/calibrate/confirm", methods=["POST"])
def confirm_calibration():
    """Confirm auto-detected zones without editing."""
    with _lock:
        zones = _state.get("calibration_zones")
        session_id = _state["session_id"]

    if zones:
        store = ZoneStore()
        store.save(zones)
        if session_id:
            update_session_zones(session_id, zones)

    with _lock:
        _state["calibration_done"] = True

    return redirect(url_for("main.session_view"))


# ---------------------------------------------------------------------------
# Session routes
# ---------------------------------------------------------------------------

@bp.route("/session")
def session_view():
    with _lock:
        if not _state["calibration_done"]:
            return redirect(url_for("main.calibrate_view"))
    return render_template("session.html")


@bp.route("/stream")
def video_stream():
    def generate():
        while True:
            with _lock:
                frame = _state.get("latest_frame")
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
            time.sleep(0.04)  # ~25 fps cap
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.route("/events")
def events():
    def generate():
        idx = 0
        while True:
            with _lock:
                new_evs = _state["events"][idx:]
                idx = len(_state["events"])
                running = _state["running"]
            for ev in new_evs:
                yield f"data: {json.dumps(ev)}\n\n"
            if not running:
                yield 'data: {"type":"done"}\n\n'
                break
            time.sleep(0.1)
    return Response(generate(), mimetype="text/event-stream")


@bp.route("/status")
def status():
    with _lock:
        return jsonify({
            "running": _state["running"],
            "calibration_done": _state["calibration_done"],
            "session_id": _state["session_id"],
            "shot_count": _state["shot_count"],
            "total_score": _state["total_score"],
        })


@bp.route("/finish", methods=["POST"])
def finish():
    with _lock:
        session_id = _state["session_id"]
        _state["running"] = False

    if session_id:
        finish_session(session_id)
        export_csv(session_id)
        export_pdf(session_id)

    return redirect(url_for("main.result_view", session_id=session_id))


@bp.route("/result/<int:session_id>")
def result_view(session_id: int):
    session = get_session(session_id)
    shots = get_session_shots(session_id)
    total = sum(s["score"] for s in shots)
    safe_name = (session["player_name"] if session else "player").replace(" ", "_")
    return render_template(
        "result.html",
        session_id=session_id,
        session=session,
        shots=shots,
        total=total,
        safe_name=safe_name,
    )


@bp.route("/download/<filename>")
def download(filename: str):
    path = OUTPUT_FOLDER / filename
    if not path.exists():
        return "File not found", 404
    return send_file(str(path), as_attachment=True)


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(source_tuple: tuple) -> None:
    src_type, src_val = source_tuple
    model = TargetModel()
    scorer = ShotScorer()
    motion = MotionDetector()
    zone_store = ZoneStore()
    frame_count = 0

    try:
        if src_type == "usb":
            ctx = CameraSource(index=src_val)
        elif src_type == "ip":
            ctx = IPStreamSource(url=src_val)
        else:
            ctx = FileSource(path=src_val)

        with ctx:
            for frame in ctx.get_frames():
                with _lock:
                    if not _state["running"]:
                        break

                frame_count += 1

                # ── Calibration phase ────────────────────────────────────
                if frame_count <= CALIBRATION_FRAMES:
                    with _lock:
                        _state["calibration_frame"] = frame.copy()
                        _state["latest_frame"] = frame.copy()

                    if frame_count == CALIBRATION_FRAMES:
                        zones_data = detect_target(frame)
                        if zones_data:
                            zone_store.set(zones_data)
                            with _lock:
                                _state["calibration_zones"] = zones_data
                    continue

                # ── Wait until user confirms calibration ─────────────────
                with _lock:
                    calib_done = _state["calibration_done"]
                    calib_zones = _state["calibration_zones"]

                if not calib_done:
                    time.sleep(0.05)
                    continue

                # Sync zone store with whatever the user confirmed
                if calib_zones and not zone_store.is_ready():
                    zone_store.set(calib_zones)
                    scorer._zone_store = zone_store

                # ── Detection phase ───────────────────────────────────────
                display = _draw_zones(frame.copy(), zone_store.get_zones())
                with _lock:
                    _state["latest_frame"] = display

                if not motion.has_changed(frame):
                    continue

                holes = model.detect(frame)
                new_holes = scorer.find_new_holes(holes)

                for hole in new_holes:
                    with _lock:
                        _state["shot_count"] += 1
                        _state["total_score"] += hole["score"]
                        shot_num = _state["shot_count"]
                        session_id = _state["session_id"]
                        total = _state["total_score"]

                    # ── Persist annotated frame for later analysis ──────────
                    frame_path = _save_shot_frame(
                        frame=frame,
                        zones=zone_store.get_zones(),
                        hole=hole,
                        session_id=session_id,
                        shot_num=shot_num,
                    )

                    add_shot(
                        session_id=session_id,
                        shot_number=shot_num,
                        score=hole["score"],
                        confidence=hole.get("confidence", 0.0),
                        bbox=hole.get("box", []),
                        frame_path=frame_path,
                    )

                    with _lock:
                        _state["events"].append({
                            "type": "shot",
                            "shot_number": shot_num,
                            "score": hole["score"],
                            "total": total,
                        })
    finally:
        with _lock:
            _state["running"] = False


def _draw_zones(frame: np.ndarray, zones: list) -> np.ndarray:
    for zone in zones:
        color = (0, 220, 80) if zone["score"] >= 7 else (0, 165, 255)
        cv2.circle(frame, (zone["cx"], zone["cy"]), zone["radius"], color, 1)
        label_x = zone["cx"] - zone["radius"] + 4
        label_y = zone["cy"] - zone["radius"] + 14
        cv2.putText(
            frame, str(zone["score"]),
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
    return frame


def _save_shot_frame(
    frame: np.ndarray,
    zones: list,
    hole: dict,
    session_id: int,
    shot_num: int,
) -> str:
    """
    Persist an annotated frame on disk for later analysis. Returns the absolute
    path written. Saved layout:
        outputs/detection_frames/session_{id}/shot_{N:03d}_score{S}_conf{C}.jpg
    """
    out_dir = OUTPUT_FOLDER.parent / "detection_frames" / f"session_{session_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = _draw_zones(frame.copy(), zones)
    x1, y1, x2, y2 = hole["box"]
    cx, cy = hole["center"]
    score = hole["score"]
    conf = hole.get("confidence", 0.0)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    cv2.putText(
        annotated, f"#{shot_num} score={score} conf={conf:.2f}",
        (x1, max(y1 - 8, 14)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
    )

    fname = f"shot_{shot_num:03d}_score{score}_conf{conf:.2f}.jpg"
    fpath = out_dir / fname
    cv2.imwrite(str(fpath), annotated)
    return str(fpath)
