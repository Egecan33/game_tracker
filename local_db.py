"""Local SQLite drop-in replacement for the Supabase Python client.

The app was written against `supabase.table(name).select(...).eq(...).execute()`
chains. This module provides the same chainable interface backed by a local
SQLite file, so the rest of the code runs unchanged without any cloud project.

The live database lives in local_data/game_tracker.sqlite and is seeded on
first run from the immutable snapshot in supabase_backup/game_tracker.sqlite.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_REPO = Path(__file__).resolve().parent
DB_PATH = _REPO / "local_data" / "game_tracker.sqlite"
SEED_PATH = _REPO / "supabase_backup" / "game_tracker.sqlite"

# Columns stored as JSON text in SQLite but exposed as dict/list (jsonb/text[] in Postgres)
_JSON_COLUMNS: Dict[str, set] = {
    "players": {"bag"},
    "games": {"mechanics"},
    "session_requests": {"payload"},
    "minigame_scores": {"meta"},
    "survey_responses": {"answers_json", "ranking_json"},
}

# Columns stored as 0/1 in SQLite but exposed as bool (boolean in Postgres)
_BOOL_COLUMNS: Dict[str, set] = {
    "games": {"supports_ffa", "supports_team", "supports_coop", "supports_solo"},
    "session_players": {"is_winner"},
    "minigame_scores": {"is_best_for_day"},
}

_PRIMARY_KEY: Dict[str, str] = {"config": "key"}  # every other table: "id"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_uuid() -> str:
    return str(uuid.uuid4())


# Server-side defaults from the original Postgres schema, applied on insert
# when the column is absent from the payload.
_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "players": {"id": _new_uuid, "joined_on": _now_iso, "bag": dict},
    "games": {"id": _new_uuid},
    "sessions": {"id": _new_uuid},
    "session_players": {"id": _new_uuid},
    "config": {},
    "session_requests": {"id": _new_uuid, "status": "pending", "created_at": _now_iso},
    "minigame_scores": {
        "id": _new_uuid,
        "score": 0,
        "rounds_played": 0,
        "duration_s": 60,
        "started_at": _now_iso,
        "is_best_for_day": False,
        "meta": dict,
    },
    "minigame_daily_awards": {"id": _new_uuid, "awarded_at": _now_iso},
    "survey_responses": {"id": _new_uuid, "ts": _now_iso},
}


def _to_db(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _from_db(table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    json_cols = _JSON_COLUMNS.get(table, set())
    bool_cols = _BOOL_COLUMNS.get(table, set())
    out: Dict[str, Any] = {}
    for col, value in row.items():
        if value is not None and col in json_cols and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        elif value is not None and col in bool_cols:
            value = bool(value)
        out[col] = value
    return out


class APIResponse:
    """Mimics supabase-py's response object: result rows live in `.data`."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __repr__(self) -> str:
        return f"APIResponse(data={self.data!r})"


class QueryBuilder:
    def __init__(self, client: "LocalSupabaseClient", table: str):
        self._client = client
        self._table = table
        self._op: Optional[str] = None
        self._payload: Optional[List[Dict[str, Any]]] = None
        self._columns = "*"
        self._filters: List[tuple] = []  # (sql_fragment, params)
        self._order: List[str] = []
        self._limit: Optional[int] = None

    # -- operations ----------------------------------------------------------
    def select(self, columns: str = "*") -> "QueryBuilder":
        self._op = "select"
        self._columns = columns
        return self

    def insert(self, payload: Union[Dict, List[Dict]]) -> "QueryBuilder":
        self._op = "insert"
        self._payload = payload if isinstance(payload, list) else [payload]
        return self

    def upsert(self, payload: Union[Dict, List[Dict]]) -> "QueryBuilder":
        self._op = "upsert"
        self._payload = payload if isinstance(payload, list) else [payload]
        return self

    def update(self, payload: Dict) -> "QueryBuilder":
        self._op = "update"
        self._payload = [payload]
        return self

    def delete(self) -> "QueryBuilder":
        self._op = "delete"
        return self

    # -- filters ---------------------------------------------------------------
    def _cmp(self, column: str, op: str, value: Any) -> "QueryBuilder":
        self._filters.append((f'"{column}" {op} ?', [_to_db(value)]))
        return self

    def eq(self, column: str, value: Any) -> "QueryBuilder":
        return self._cmp(column, "=", value)

    def neq(self, column: str, value: Any) -> "QueryBuilder":
        return self._cmp(column, "!=", value)

    def gt(self, column: str, value: Any) -> "QueryBuilder":
        return self._cmp(column, ">", value)

    def gte(self, column: str, value: Any) -> "QueryBuilder":
        return self._cmp(column, ">=", value)

    def lt(self, column: str, value: Any) -> "QueryBuilder":
        return self._cmp(column, "<", value)

    def lte(self, column: str, value: Any) -> "QueryBuilder":
        return self._cmp(column, "<=", value)

    def in_(self, column: str, values: List[Any]) -> "QueryBuilder":
        placeholders = ", ".join("?" for _ in values)
        self._filters.append(
            (f'"{column}" IN ({placeholders})', [_to_db(v) for v in values])
        )
        return self

    def is_(self, column: str, value: Any) -> "QueryBuilder":
        if value is None or (isinstance(value, str) and value.lower() == "null"):
            self._filters.append((f'"{column}" IS NULL', []))
        else:
            self._cmp(column, "=", value)
        return self

    # -- modifiers ---------------------------------------------------------------
    def order(self, column: str, desc: bool = False) -> "QueryBuilder":
        # Match Postgres null ordering (ASC NULLS LAST / DESC NULLS FIRST);
        # SQLite defaults are the opposite.
        direction = "DESC NULLS FIRST" if desc else "ASC NULLS LAST"
        self._order.append(f'"{column}" {direction}')
        return self

    def limit(self, count: int) -> "QueryBuilder":
        self._limit = int(count)
        return self

    # -- execution -----------------------------------------------------------------
    def execute(self) -> APIResponse:
        return self._client._run(self)

    def _where_clause(self) -> tuple:
        if not self._filters:
            return "", []
        fragments, params = zip(*self._filters)
        return " WHERE " + " AND ".join(fragments), [p for ps in params for p in ps]


class LocalSupabaseClient:
    """SQLite-backed stand-in for supabase-py's Client (table API only)."""

    def __init__(self, db_path: Union[str, Path] = DB_PATH):
        db_path = Path(db_path)
        if not db_path.exists():
            if not SEED_PATH.exists():
                raise FileNotFoundError(
                    f"No database at {db_path} and no seed at {SEED_PATH}"
                )
            db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(SEED_PATH, db_path)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.RLock()
        self._tables = {
            r[0]
            for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    def table(self, name: str) -> QueryBuilder:
        if name not in self._tables:
            raise ValueError(f"Unknown table: {name}")
        return QueryBuilder(self, name)

    # -- internals ------------------------------------------------------------
    def _run(self, q: QueryBuilder) -> APIResponse:
        with self._lock:
            if q._op == "select":
                return self._select(q)
            if q._op == "insert":
                return self._insert(q, upsert=False)
            if q._op == "upsert":
                return self._insert(q, upsert=True)
            if q._op == "update":
                return self._update(q)
            if q._op == "delete":
                return self._delete(q)
            raise ValueError("No operation set; call select/insert/update/upsert/delete first")

    def _select(self, q: QueryBuilder) -> APIResponse:
        if q._columns.strip() == "*":
            cols_sql = "*"
        else:
            cols = [c.strip() for c in q._columns.split(",") if c.strip()]
            cols_sql = ", ".join(f'"{c}"' for c in cols)
        sql = f'SELECT {cols_sql} FROM "{q._table}"'
        where, params = q._where_clause()
        sql += where
        if q._order:
            sql += " ORDER BY " + ", ".join(q._order)
        if q._limit is not None:
            sql += f" LIMIT {q._limit}"
        rows = self._conn.execute(sql, params).fetchall()
        return APIResponse([_from_db(q._table, dict(r)) for r in rows])

    def _insert(self, q: QueryBuilder, upsert: bool) -> APIResponse:
        defaults = _DEFAULTS.get(q._table, {})
        pk = _PRIMARY_KEY.get(q._table, "id")
        inserted: List[Dict[str, Any]] = []
        for payload in q._payload:
            row = dict(payload)
            for col, default in defaults.items():
                if col not in row or row[col] is None:
                    row[col] = default() if callable(default) else default
            cols = list(row.keys())
            cols_sql = ", ".join(f'"{c}"' for c in cols)
            placeholders = ", ".join("?" for _ in cols)
            sql = f'INSERT INTO "{q._table}" ({cols_sql}) VALUES ({placeholders})'
            if upsert:
                updates = ", ".join(
                    f'"{c}" = excluded."{c}"' for c in cols if c != pk
                )
                sql += f' ON CONFLICT ("{pk}") DO UPDATE SET {updates}' if updates else (
                    f' ON CONFLICT ("{pk}") DO NOTHING'
                )
            self._conn.execute(sql, [_to_db(row[c]) for c in cols])
            inserted.append(row)
        self._conn.commit()
        return APIResponse(inserted)

    def _update(self, q: QueryBuilder) -> APIResponse:
        payload = q._payload[0]
        set_sql = ", ".join(f'"{c}" = ?' for c in payload)
        set_params = [_to_db(v) for v in payload.values()]
        where, where_params = q._where_clause()
        self._conn.execute(
            f'UPDATE "{q._table}" SET {set_sql}{where}', set_params + where_params
        )
        self._conn.commit()
        return APIResponse([])

    def _delete(self, q: QueryBuilder) -> APIResponse:
        where, params = q._where_clause()
        self._conn.execute(f'DELETE FROM "{q._table}"{where}', params)
        self._conn.commit()
        return APIResponse([])


_client: Optional[LocalSupabaseClient] = None
_client_lock = threading.Lock()


def get_client(db_path: Union[str, Path] = DB_PATH) -> LocalSupabaseClient:
    """Process-wide singleton, shared across Streamlit sessions/threads."""
    global _client
    with _client_lock:
        if _client is None:
            _client = LocalSupabaseClient(db_path)
        return _client
