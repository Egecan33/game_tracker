#!/usr/bin/env python3
"""Rebuild local artifacts from the JSON table dumps in data/.

Produces, next to this script:
  - game_tracker.sqlite  : SQLite database with all tables and rows
  - data.sql             : Postgres INSERT statements (pairs with schema.sql)
  - csv/<table>.csv      : one CSV per table

Run: python3 build_local_db.py
"""

import csv
import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# (table, ordered column list, {column: pg_type}) — column order matches schema.sql
TABLES = {
    "players": (
        ["id", "name", "nickname", "joined_on", "pin_code", "emoji_lock", "bag"],
        {"bag": "jsonb"},
    ),
    "games": (
        ["id", "name", "bgg_slug", "min_players", "max_players", "notes", "mechanics",
         "release_year", "bgg_weight", "game_type", "supports_ffa", "supports_team",
         "supports_coop", "supports_solo"],
        {"mechanics": "text[]", "supports_ffa": "bool", "supports_team": "bool",
         "supports_coop": "bool", "supports_solo": "bool"},
    ),
    "sessions": (
        ["id", "played_at", "game_id", "location", "notes"],
        {},
    ),
    "session_players": (
        ["id", "session_id", "player_id", "team", "position", "points", "is_winner"],
        {"is_winner": "bool"},
    ),
    "config": (
        ["key", "value"],
        {},
    ),
    "session_requests": (
        ["id", "payload", "created_by", "status", "created_at"],
        {"payload": "jsonb"},
    ),
    "minigame_scores": (
        ["id", "date_key", "player_id", "score", "rounds_played", "duration_s",
         "started_at", "finished_at", "is_best_for_day", "meta"],
        {"is_best_for_day": "bool", "meta": "jsonb"},
    ),
    "minigame_daily_awards": (
        ["id", "date_key", "player_id", "placement", "reward_code", "awarded_at"],
        {},
    ),
    "survey_responses": (
        ["id", "ts", "user", "answers_json", "ranking_json"],
        {"answers_json": "jsonb", "ranking_json": "jsonb"},
    ),
}

# Insert parents before children so foreign keys resolve
INSERT_ORDER = [
    "players", "games", "config", "sessions", "session_players",
    "minigame_scores", "minigame_daily_awards", "session_requests", "survey_responses",
]


def pg_literal(value, pg_type):
    if value is None:
        return "NULL"
    if pg_type == "bool":
        return "TRUE" if value else "FALSE"
    if pg_type == "jsonb":
        return "'" + json.dumps(value, ensure_ascii=False).replace("'", "''") + "'::jsonb"
    if pg_type == "text[]":
        items = ", ".join("'" + str(v).replace("'", "''") + "'" for v in value)
        return f"ARRAY[{items}]::text[]" if value else "'{}'::text[]"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def sqlite_value(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def main():
    dumps = {t: json.loads((DATA / f"{t}.json").read_text(encoding="utf-8")) for t in TABLES}

    # --- SQLite -------------------------------------------------------------
    db_path = HERE / "game_tracker.sqlite"
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript("""
        PRAGMA journal_mode = WAL;
        CREATE TABLE players (
            id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, nickname TEXT,
            joined_on TEXT, pin_code TEXT, emoji_lock TEXT, bag TEXT DEFAULT '{}'
        );
        CREATE TABLE games (
            id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, bgg_slug TEXT,
            min_players INTEGER, max_players INTEGER, notes TEXT, mechanics TEXT,
            release_year INTEGER, bgg_weight REAL,
            game_type TEXT CHECK (game_type IN ('board', 'digital', 'sport', 'other')),
            supports_ffa INTEGER, supports_team INTEGER,
            supports_coop INTEGER, supports_solo INTEGER
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, played_at TEXT NOT NULL,
            game_id TEXT REFERENCES games (id), location TEXT, notes TEXT
        );
        CREATE TABLE session_players (
            id TEXT PRIMARY KEY,
            session_id TEXT REFERENCES sessions (id),
            player_id TEXT REFERENCES players (id),
            team TEXT, position INTEGER, points REAL, is_winner INTEGER,
            UNIQUE (session_id, player_id)
        );
        CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE session_requests (
            id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_by TEXT,
            status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL
        );
        CREATE TABLE minigame_scores (
            id TEXT PRIMARY KEY, date_key TEXT NOT NULL,
            player_id TEXT NOT NULL REFERENCES players (id),
            score INTEGER NOT NULL DEFAULT 0, rounds_played INTEGER NOT NULL DEFAULT 0,
            duration_s INTEGER NOT NULL DEFAULT 60, started_at TEXT NOT NULL,
            finished_at TEXT, is_best_for_day INTEGER NOT NULL DEFAULT 0,
            meta TEXT DEFAULT '{}'
        );
        CREATE TABLE minigame_daily_awards (
            id TEXT PRIMARY KEY, date_key TEXT NOT NULL,
            player_id TEXT NOT NULL REFERENCES players (id),
            placement INTEGER, reward_code TEXT NOT NULL, awarded_at TEXT NOT NULL
        );
        CREATE TABLE survey_responses (
            id TEXT PRIMARY KEY, ts TEXT NOT NULL, user TEXT,
            answers_json TEXT, ranking_json TEXT
        );
        CREATE INDEX idx_sessions_game ON sessions (game_id);
        CREATE INDEX idx_session_players_session ON session_players (session_id);
        CREATE INDEX idx_session_players_player ON session_players (player_id);
        CREATE INDEX idx_minigame_scores_day_best ON minigame_scores (date_key, is_best_for_day);
        CREATE INDEX idx_minigame_scores_day_player ON minigame_scores (date_key, player_id);
        CREATE INDEX idx_minigame_awards_day ON minigame_daily_awards (date_key);
    """)
    for table in INSERT_ORDER:
        cols, _ = TABLES[table]
        placeholders = ", ".join("?" for _ in cols)
        quoted = ", ".join(f'"{c}"' for c in cols)
        con.executemany(
            f'INSERT INTO {table} ({quoted}) VALUES ({placeholders})',
            [[sqlite_value(row.get(c)) for c in cols] for row in dumps[table]],
        )
    con.commit()

    # --- Postgres INSERT dump ------------------------------------------------
    lines = [
        "-- Data dump of Supabase project \"board-game responses\" (wsbwrytlkonmnzksqrqb)",
        "-- Snapshot: 2026-07-07. Apply after schema.sql.",
        "BEGIN;",
    ]
    for table in INSERT_ORDER:
        cols, types = TABLES[table]
        rows = dumps[table]
        lines.append(f"\n-- {table}: {len(rows)} rows")
        quoted = ", ".join(f'"{c}"' for c in cols)
        for row in rows:
            values = ", ".join(pg_literal(row.get(c), types.get(c)) for c in cols)
            lines.append(f'INSERT INTO public.{table} ({quoted}) VALUES ({values});')
    lines.append("\nCOMMIT;")
    (HERE / "data.sql").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- CSV ------------------------------------------------------------------
    csv_dir = HERE / "csv"
    csv_dir.mkdir(exist_ok=True)
    for table, (cols, _) in TABLES.items():
        with open(csv_dir / f"{table}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for row in dumps[table]:
                writer.writerow(["" if row.get(c) is None else sqlite_value(row.get(c)) for c in cols])

    # --- Verify ----------------------------------------------------------------
    print(f"{'table':<24} {'json':>5} {'sqlite':>6}")
    all_ok = True
    for table in INSERT_ORDER:
        n_json = len(dumps[table])
        n_db = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        mark = "OK" if n_json == n_db else "MISMATCH"
        all_ok &= n_json == n_db
        print(f"{table:<24} {n_json:>5} {n_db:>6}  {mark}")
    con.close()
    print("\nDone." if all_ok else "\nMISMATCH DETECTED")


if __name__ == "__main__":
    main()
