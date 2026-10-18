-- Run once against the hub database, using a role permitted to create tables.
-- The runtime role needs SELECT on both tables and UPDATE on stationwatch_alerts.
-- These ordinary PostgreSQL tables also work in a Timescale database; no extension is required.
CREATE TABLE IF NOT EXISTS stationwatch_alerts (
    id TEXT PRIMARY KEY,
    captured_at TIMESTAMPTZ NOT NULL,
    reviewed BOOLEAN NOT NULL DEFAULT FALSE,
    -- Same event JSON shape emitted by stationwatch.storage.make_event().
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object')
);
CREATE INDEX IF NOT EXISTS stationwatch_alerts_captured_at_idx ON stationwatch_alerts (captured_at DESC);

-- Optional evidence supplied by the alert producer. The cache is disposable.
-- Store only confirmed-alert JPEGs/MP4s here, never the continuous camera feed.
CREATE TABLE IF NOT EXISTS stationwatch_evidence (
    event_id TEXT NOT NULL REFERENCES stationwatch_alerts(id),
    kind TEXT NOT NULL CHECK (kind IN ('snapshot', 'clip')),
    content BYTEA NOT NULL,
    PRIMARY KEY (event_id, kind)
);
