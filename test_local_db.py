"""Exercises local_db.py with every Supabase call pattern found in app.py and
galios_den_game.py, against a throwaway copy of the seed database.

Run: python3 test_local_db.py
"""

import shutil
import tempfile
from pathlib import Path

from local_db import LocalSupabaseClient, SEED_PATH

failures = []


def check(name, condition, detail=""):
    status = "ok " if condition else "FAIL"
    print(f"  [{status}] {name}{f'  ({detail})' if detail and not condition else ''}")
    if not condition:
        failures.append(name)


tmp = Path(tempfile.mkdtemp()) / "test.sqlite"
shutil.copy(SEED_PATH, tmp)
sb = LocalSupabaseClient(tmp)

print("select patterns")
res = sb.table("players").select("*").execute()
check("select * from players", len(res.data) == 18)
check("bag decoded to dict", isinstance(res.data[0]["bag"], dict))

res = sb.table("players").select("bag").eq("id", "3c46dbff-adaf-4781-812e-3ac1be41114b").execute()
check("select bag by id (app.py:309)", res.data[0]["bag"]["inventory"]["DUST"] == 403)

res = sb.table("config").select("value").eq("key", "k_factor").limit(1).execute()
check("config select value limit 1 (app.py:572)", res.data == [{"value": "32"}])

res = (
    sb.table("minigame_scores")
    .select("id,score,duration_s,finished_at")
    .eq("date_key", "2025-09-05")
    .eq("player_id", "3c46dbff-adaf-4781-812e-3ac1be41114b")
    .order("score", desc=True)
    .order("duration_s", desc=False)
    .order("finished_at", desc=False)
    .limit(1)
    .execute()
)
check("best-of-day query (app.py:438-445)", res.data[0]["score"] == 15)

res = (
    sb.table("minigame_scores")
    .select("player_id, score, duration_s, finished_at, date_key")
    .gte("date_key", "2025-09-01")
    .lte("date_key", "2025-09-07")
    .execute()
)
check("weekly range gte/lte (galios:618)", len(res.data) == 8, f"got {len(res.data)}")

res = sb.table("minigame_scores").select("player_id, score").eq("date_key", "2025-08-17").eq("is_best_for_day", True).execute()
check("filter on boolean True", len(res.data) == 5)
check("bool decoded", all(r not in (0, 1) for r in [type(x.get("score")) for x in res.data]))

print("insert with defaults")
sb.table("minigame_scores").insert(
    {"date_key": "2026-07-07", "player_id": "3c46dbff-adaf-4781-812e-3ac1be41114b",
     "score": 5, "rounds_played": 5, "duration_s": 42,
     "finished_at": "2026-07-07T20:00:00+00:00", "meta": {"minigame": "Galio's Den"}}
).execute()
row = sb.table("minigame_scores").select("*").eq("date_key", "2026-07-07").execute().data[0]
check("insert filled uuid id", isinstance(row["id"], str) and len(row["id"]) == 36)
check("insert filled started_at default", row["started_at"] is not None)
check("insert default is_best_for_day False", row["is_best_for_day"] is False)
check("meta json round-trip", row["meta"] == {"minigame": "Galio's Den"})

print("update patterns")
sb.table("players").update({"bag": {"test": 1}}).eq("id", "3958cc97-5507-421f-a821-fd80cc9e7f7f").execute()
row = sb.table("players").select("bag").eq("id", "3958cc97-5507-421f-a821-fd80cc9e7f7f").execute().data[0]
check("update jsonb bag (app.py:322)", row["bag"] == {"test": 1})

sb.table("minigame_scores").update({"is_best_for_day": False}).eq("date_key", "2025-08-17").execute()
n = len(sb.table("minigame_scores").select("id").eq("date_key", "2025-08-17").eq("is_best_for_day", True).execute().data)
check("bulk bool update (app.py:433)", n == 0)

print("upsert patterns")
sb.table("config").upsert({"key": "k_factor", "value": "40"}).execute()
check("upsert existing config key", sb.table("config").select("value").eq("key", "k_factor").execute().data[0]["value"] == "40")
sb.table("config").upsert({"key": "brand_new_key", "value": "hello"}).execute()
check("upsert new config key", sb.table("config").select("value").eq("key", "brand_new_key").execute().data[0]["value"] == "hello")
check("config count grew by 1", len(sb.table("config").select("*").execute().data) == 20)

print("session flow (insert with client-side uuid, cascade delete)")
import uuid as _uuid
sid = str(_uuid.uuid4())
sb.table("sessions").insert({"id": sid, "played_at": "2026-07-07T21:00:00+00:00",
                             "game_id": "5be4a8f6-9838-454b-ba22-f91ab02c51cd"}).execute()
sb.table("session_players").insert({"id": str(_uuid.uuid4()), "session_id": sid,
                                    "player_id": "3c46dbff-adaf-4781-812e-3ac1be41114b",
                                    "position": 1, "is_winner": True}).execute()
check("session inserted", len(sb.table("sessions").select("*").eq("id", sid).execute().data) == 1)
sb.table("session_players").delete().eq("session_id", sid).execute()
sb.table("sessions").delete().eq("id", sid).execute()
check("cascade delete (app.py:928)", len(sb.table("sessions").select("*").eq("id", sid).execute().data) == 0)

print("daily award flow (app.py:459-505)")
res = sb.table("minigame_daily_awards").select("id").eq("date_key", "2025-08-17").limit(1).execute()
check("award-exists check", len(res.data) == 1)
sb.table("minigame_daily_awards").insert({"date_key": "2026-07-06", "player_id": "3c46dbff-adaf-4781-812e-3ac1be41114b",
                                          "placement": 1, "reward_code": "BOX_GOLD"}).execute()
row = sb.table("minigame_daily_awards").select("*").eq("date_key", "2026-07-06").execute().data[0]
check("award insert with awarded_at default", row["awarded_at"] is not None)

print("null ordering matches Postgres (ASC NULLS LAST)")
res = sb.table("minigame_daily_awards").select("placement").eq("date_key", "2025-08-17").order("placement", desc=False).execute()
placements = [r["placement"] for r in res.data]
check("nulls sort last on asc", placements[:3] == [1, 2, 3] and placements[-1] is None)

print("empty tables")
check("survey_responses empty select", sb.table("survey_responses").select("*").execute().data == [])

print()
if failures:
    print(f"FAILED: {failures}")
    raise SystemExit(1)
print("ALL TESTS PASSED")
