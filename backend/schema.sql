CREATE TABLE IF NOT EXISTS maps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    width_cells INTEGER NOT NULL,
    height_cells INTEGER NOT NULL,
    cell_size_m REAL NOT NULL DEFAULT 0.1,
    grid_data BLOB,
    scene_glb_path TEXT,
    scene_origin_x REAL NOT NULL DEFAULT 0.0,
    scene_origin_y REAL NOT NULL DEFAULT 0.0,
    scene_scale REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS spaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    map_id INTEGER NOT NULL REFERENCES maps(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    vertices_json TEXT NOT NULL,
    is_home INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (map_id, name)
);

CREATE INDEX IF NOT EXISTS idx_spaces_map_id ON spaces(map_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_spaces_one_home_per_map
    ON spaces(map_id) WHERE is_home = 1;

CREATE TABLE IF NOT EXISTS command_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    map_id INTEGER REFERENCES maps(id) ON DELETE SET NULL,
    audio_path TEXT,
    transcription TEXT,
    intent TEXT NOT NULL,
    params_json TEXT,
    success INTEGER NOT NULL,
    error_message TEXT,
    latency_ms INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS command_corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command_log_id INTEGER NOT NULL REFERENCES command_log(id) ON DELETE CASCADE,
    transcription TEXT NOT NULL,
    predicted_intent TEXT NOT NULL,
    predicted_confidence REAL NOT NULL,
    predicted_params_json TEXT,
    was_correct INTEGER NOT NULL,
    corrected_intent TEXT,
    corrected_params_json TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_command_corrections_log_id ON command_corrections(command_log_id);
CREATE INDEX IF NOT EXISTS idx_command_corrections_created_at ON command_corrections(created_at);
