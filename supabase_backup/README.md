# Supabase backup — "board-game responses"

Complete snapshot of the Supabase project backing the game tracker app, taken
**2026-07-07** so the cloud project can be closed and the app moved to local
storage.

- Project ref: `wsbwrytlkonmnzksqrqb` (https://wsbwrytlkonmnzksqrqb.supabase.co)
- Postgres 17.4, region eu-north-1
- No storage buckets, edge functions, custom Postgres functions, views, or
  triggers existed — the tables below are the entire project state.
- Row counts were verified against `SELECT count(*)` on the live database at
  snapshot time, and the REST export was cross-checked row-by-row against an
  independent SQL dump.

| table                 | rows |
| --------------------- | ---- |
| players               | 18   |
| games                 | 18   |
| sessions              | 109  |
| session_players       | 323  |
| config                | 19   |
| minigame_scores       | 19   |
| minigame_daily_awards | 26   |
| session_requests      | 0    |
| survey_responses      | 0    |
| **total**             | 532  |

## Contents

| path                  | what it is |
| --------------------- | ---------- |
| `data/*.json`         | Source of truth: raw rows per table, exactly as returned by the Supabase REST API |
| `schema.sql`          | Postgres DDL — tables, constraints, indexes, RLS policies |
| `data.sql`            | Postgres INSERT statements (apply after `schema.sql`) |
| `game_tracker.sqlite` | Ready-to-use SQLite database with all tables and rows |
| `csv/*.csv`           | One CSV per table, for spreadsheets or pandas |
| `build_local_db.py`   | Regenerates the sqlite/data.sql/csv artifacts from `data/*.json` |

## Restore options

**Local SQLite (planned direction for the app):**

```python
import sqlite3
con = sqlite3.connect("supabase_backup/game_tracker.sqlite")
```

JSONB columns (`players.bag`, `minigame_scores.meta`, …) are stored as JSON
text; booleans as 0/1; timestamps as ISO-8601 text with timezone.

**Any Postgres:**

```sh
psql "$DATABASE_URL" -f schema.sql
psql "$DATABASE_URL" -f data.sql
```

## Notes

- `players.pin_code` values are plaintext PINs — keep this repo private.
- Type mapping in SQLite: `uuid`→TEXT, `timestamptz`→TEXT (ISO-8601),
  `jsonb`→TEXT (JSON), `numeric`→REAL, `boolean`→INTEGER, `text[]`→TEXT (JSON array).
- In the live project, RLS was disabled (or effectively permissive `USING (true)`)
  on every table, so the anon key had full read/write access — one of the
  reasons to move off the public cloud endpoint.
