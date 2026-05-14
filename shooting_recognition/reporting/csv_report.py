from pathlib import Path

import pandas as pd

from config.settings import OUTPUT_FOLDER
from database.models import get_session, get_session_shots


def export_csv(session_id: int) -> Path:
    session = get_session(session_id)
    shots = get_session_shots(session_id)

    cumulative = 0
    rows = []
    for shot in shots:
        cumulative += shot["score"]
        rows.append({
            "player": session["player_name"],
            "shot_number": shot["shot_number"],
            "score": shot["score"],
            "cumulative": cumulative,
            "confidence": round(shot["confidence"] or 0.0, 3),
            "timestamp": shot["timestamp"],
        })

    df = pd.DataFrame(rows)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    safe_name = session["player_name"].replace(" ", "_")
    path = OUTPUT_FOLDER / f"session_{session_id}_{safe_name}.csv"
    df.to_csv(path, index=False)
    return path
