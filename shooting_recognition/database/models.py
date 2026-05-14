import json
from typing import Any, Dict, List, Optional
from database.db import get_connection, release_connection


def create_session(player_name: str, calibration_zones: Optional[dict] = None) -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (player_name, calibration_zones) "
                "VALUES (%s, %s) RETURNING id",
                (player_name, json.dumps(calibration_zones) if calibration_zones else None),
            )
            session_id: int = cur.fetchone()[0]
        conn.commit()
        return session_id
    finally:
        release_connection(conn)


def finish_session(session_id: int) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET finished_at = NOW() WHERE id = %s",
                (session_id,),
            )
        conn.commit()
    finally:
        release_connection(conn)


def update_session_zones(session_id: int, calibration_zones: dict) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET calibration_zones = %s WHERE id = %s",
                (json.dumps(calibration_zones), session_id),
            )
        conn.commit()
    finally:
        release_connection(conn)


def add_shot(
    session_id: int,
    shot_number: int,
    score: int,
    confidence: float,
    bbox: List[int],
    frame_path: Optional[str] = None,
) -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO shots (session_id, shot_number, score, confidence, bbox, frame_path) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (session_id, shot_number, score, confidence, json.dumps(bbox), frame_path),
            )
            shot_id: int = cur.fetchone()[0]
        conn.commit()
        return shot_id
    finally:
        release_connection(conn)


def get_session(session_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, player_name, started_at, finished_at, calibration_zones "
                "FROM sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "player_name": row[1],
            "started_at": row[2].isoformat() if row[2] else None,
            "finished_at": row[3].isoformat() if row[3] else None,
            "calibration_zones": row[4],
        }
    finally:
        release_connection(conn)


def get_session_shots(session_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, shot_number, score, confidence, bbox, timestamp, frame_path "
                "FROM shots WHERE session_id = %s ORDER BY shot_number",
                (session_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "shot_number": r[1],
                "score": r[2],
                "confidence": r[3],
                "bbox": r[4],
                "timestamp": r[5].isoformat() if r[5] else None,
                "frame_path": r[6],
            }
            for r in rows
        ]
    finally:
        release_connection(conn)
