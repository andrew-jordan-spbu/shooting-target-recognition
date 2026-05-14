CREATE TABLE IF NOT EXISTS sessions (
    id                SERIAL      PRIMARY KEY,
    player_name       TEXT        NOT NULL,
    started_at        TIMESTAMPTZ DEFAULT NOW(),
    finished_at       TIMESTAMPTZ,
    calibration_zones JSONB
);

CREATE TABLE IF NOT EXISTS shots (
    id          SERIAL      PRIMARY KEY,
    session_id  INTEGER     NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    shot_number INTEGER     NOT NULL,
    score       INTEGER     NOT NULL CHECK (score BETWEEN 0 AND 10),
    confidence  REAL,
    bbox        JSONB,
    timestamp   TIMESTAMPTZ DEFAULT NOW(),
    frame_path  TEXT
);

CREATE INDEX IF NOT EXISTS idx_shots_session_id ON shots(session_id);
