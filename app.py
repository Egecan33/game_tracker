# app.py — Streamlit Game Sessions Tracker (Supabase)
# ---------------------------------------------------
# Track any games (board, video, sports), sessions, players, winners/teams,
# and show detailed leaderboards. Includes FFA, Team, Co-op, Solo.
# ELO configuration lives in Admin (General is view-only).
# ---------------------------------------------------

from __future__ import annotations
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import plotly.graph_objects as go  # for H2H heatmap

# ─────────────────────────────────────────────────────────────────────────────
# Environment & Supabase
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

st.set_page_config(page_title="🎯 Game Sessions Tracker", page_icon="🏆", layout="wide")


def _sec(key: str, default: str = "") -> str:
    """Prefer st.secrets over env if present."""
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


SUPABASE_URL = _sec("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = _sec("SUPABASE_ANON_KEY", "").strip()
ADMIN_PASSWORD = _sec("ADMIN_PASSWORD", "change-me")
MODERATOR_PASSWORD = _sec("MODERATOR_PASSWORD", "mod-me")

# Supabase v2 client
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = Any  # type: ignore

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_ANON_KEY and create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as e:
        supabase = None
        st.sidebar.error(f"Supabase init failed: {e}")
else:
    st.sidebar.warning(
        "Supabase credentials missing. Set SUPABASE_URL and SUPABASE_ANON_KEY."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Daily mini-game constants, fonts, helpers
# ─────────────────────────────────────────────────────────────────────────────

DAY_ROLLOVER_HOUR_UTC = 3  # 03:00 UTC = new day
MINIGAME_NAME = "Odd Emoji Out"
OEO_DURATION_S = 60
OEO_START_GRID = 3
OEO_MAX_GRID = 7

# Cosmetic fonts (stored as cosmetic code in bag.equipped.font)
# ───────────────── Fonts (more styles) ─────────────────
FONT_CATALOG = {
    "font_arcade": {
        "label": "Arcade",
        "import": "https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap",
        "css_family": "'Press Start 2P', cursive",
    },
    "font_clean": {
        "label": "Clean",
        "import": "https://fonts.googleapis.com/css2?family=Rubik:wght@400;600&display=swap",
        "css_family": "'Rubik', sans-serif",
    },
    "font_serif": {
        "label": "Serif",
        "import": "https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&display=swap",
        "css_family": "'Merriweather', serif",
    },
    "font_hand": {
        "label": "Handwritten",
        "import": "https://fonts.googleapis.com/css2?family=Kalam:wght@300;400;700&display=swap",
        "css_family": "'Kalam', cursive",
    },
    "font_display": {
        "label": "Display",
        "import": "https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap",
        "css_family": "'Bebas Neue', sans-serif",
    },
    # NEW:
    "font_mono": {
        "label": "Mono",
        "import": "https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap",
        "css_family": "'Roboto Mono', monospace",
    },
    "font_round": {
        "label": "Rounded",
        "import": "https://fonts.googleapis.com/css2?family=Fredoka:wght@400..700&display=swap",
        "css_family": "'Fredoka', sans-serif",
    },
    "font_neon": {
        "label": "Neon",
        "import": "https://fonts.googleapis.com/css2?family=Monoton&display=swap",
        "css_family": "'Monoton', cursive",
    },
    "font_script": {
        "label": "Script",
        "import": "https://fonts.googleapis.com/css2?family=Pacifico&display=swap",
        "css_family": "'Pacifico', cursive",
    },
    "font_geo": {
        "label": "Geo",
        "import": "https://fonts.googleapis.com/css2?family=Righteous&display=swap",
        "css_family": "'Righteous', sans-serif",
    },
}

# ───────────────── Items & cosmetics (expanded) ─────────────────
ITEM_DB = {
    # boxes
    "BOX_BRONZE": {"label": "Bronze Box", "icon": "📦"},
    "BOX_SILVER": {"label": "Silver Box", "icon": "🎁"},
    "BOX_GOLD": {"label": "Gold Box", "icon": "🗃️"},
    # currency
    "DUST": {"label": "Emoji Dust", "icon": "✨"},
    # emoji items
    "EMOJI_STAR": {"label": "Star", "icon": "⭐"},
    "EMOJI_FIRE": {"label": "Fire", "icon": "🔥"},
    "EMOJI_APPLE": {"label": "Apple", "icon": "🍎"},
    "EMOJI_CROWN": {"label": "Crown", "icon": "👑"},
    # voucher
    "V_PICK_NEXT_GAME": {"label": "Voucher: Pick Next Game", "icon": "🎟️"},
    # cosmetics — fonts
    "font_arcade": {"label": "Font: Arcade", "icon": "🅰️"},
    "font_clean": {"label": "Font: Clean", "icon": "🔡"},
    "font_serif": {"label": "Font: Serif", "icon": "🔠"},
    "font_hand": {"label": "Font: Handwritten", "icon": "✍️"},
    "font_display": {"label": "Font: Display", "icon": "🆒"},
    "font_mono": {"label": "Font: Mono", "icon": "💻"},
    "font_round": {"label": "Font: Rounded", "icon": "🫧"},
    "font_neon": {"label": "Font: Neon", "icon": "💡"},
    "font_script": {"label": "Font: Script", "icon": "🖋️"},
    "font_geo": {"label": "Font: Geo", "icon": "🔷"},
    # cosmetics — badges/titles
    "badge_star": {"label": "Badge: Star", "icon": "🌟"},
    "badge_crown": {"label": "Badge: Crown", "icon": "👑"},
    "badge_bolt": {"label": "Badge: Lightning", "icon": "⚡"},
    "badge_skull": {"label": "Badge: Skull", "icon": "☠️"},
    "title_emoji_hunter": {"label": "Title: Emoji Hunter", "icon": "🏹"},
    "title_box_breaker": {"label": "Title: Box Breaker", "icon": "📦"},
    "title_speedrunner": {"label": "Title: Speedrunner", "icon": "🏃‍♂️"},
    "title_sharpshooter": {"label": "Title: Sharpshooter", "icon": "🎯"},
}
EMOJI_ITEM_POOL = ["EMOJI_STAR", "EMOJI_FIRE", "EMOJI_APPLE", "EMOJI_CROWN"]

# ───────────────── Drop tables (heavier + cosmetics across all tiers) ─────────────────
BOX_TABLES = {
    "BOX_BRONZE": [
        {"type": "inventory", "code": EMOJI_ITEM_POOL, "weight": 60, "qty": (1, 3)},
        {"type": "inventory", "code": "DUST", "weight": 20, "qty": (6, 10)},
        {"type": "cosmetic", "code": "font_clean", "weight": 7, "qty": 1},
        {"type": "cosmetic", "code": "font_serif", "weight": 5, "qty": 1},
        {
            "type": "cosmetic",
            "code": ["badge_star", "title_emoji_hunter"],
            "weight": 8,
            "qty": 1,
        },
    ],
    "BOX_SILVER": [
        {"type": "inventory", "code": EMOJI_ITEM_POOL, "weight": 45, "qty": (2, 5)},
        {"type": "inventory", "code": "DUST", "weight": 25, "qty": (14, 20)},
        {
            "type": "cosmetic",
            "code": ["font_serif", "font_hand", "badge_star", "title_emoji_hunter"],
            "weight": 18,
            "qty": 1,
        },
        {
            "type": "cosmetic",
            "code": ["font_arcade", "font_mono", "font_round"],
            "weight": 12,
            "qty": 1,
        },
    ],
    "BOX_GOLD": [
        {"type": "inventory", "code": "DUST", "weight": 30, "qty": (26, 36)},
        {"type": "inventory", "code": EMOJI_ITEM_POOL, "weight": 30, "qty": (5, 8)},
        {
            "type": "cosmetic",
            "code": [
                "font_arcade",
                "font_display",
                "font_neon",
                "font_geo",
                "badge_crown",
                "badge_bolt",
                "title_box_breaker",
                "title_speedrunner",
                "title_sharpshooter",
            ],
            "weight": 30,
            "qty": 1,
        },
        {"type": "inventory", "code": "V_PICK_NEXT_GAME", "weight": 10, "qty": 1},
    ],
}

# ───────────────── Crafting (more routes to cosmetics) ─────────────────
CRAFT_RECIPES: Dict[str, Dict[str, int]] = {
    # badges/titles
    "badge_star": {"EMOJI_STAR": 20},
    "badge_bolt": {"EMOJI_FIRE": 25, "DUST": 30},
    "badge_crown": {"EMOJI_CROWN": 15, "DUST": 40},
    "title_emoji_hunter": {"EMOJI_STAR": 10, "EMOJI_APPLE": 10},
    "title_box_breaker": {"EMOJI_APPLE": 20, "EMOJI_FIRE": 20},
    "title_speedrunner": {"DUST": 120},
    "title_sharpshooter": {"EMOJI_STAR": 15, "DUST": 60},
    # fonts (let dust buy style)
    "font_hand": {"DUST": 50},
    "font_mono": {"DUST": 70},
    "font_round": {"DUST": 70},
    "font_script": {"DUST": 90},
    "font_geo": {"DUST": 100},
    "font_neon": {"DUST": 120},
    "font_display": {"DUST": 140},
    "font_arcade": {"DUST": 150},
}


def _utc_now() -> datetime:
    return datetime.utcnow()


def _today_key_utc() -> str:
    now = _utc_now()
    if now.hour < DAY_ROLLOVER_HOUR_UTC:
        day = now.date() - timedelta(days=1)
    else:
        day = now.date()
    return day.strftime("%Y-%m-%d")


def _yesterday_key_utc() -> str:
    now = _utc_now()
    # "Yesterday" relative to rollover window:
    if now.hour < DAY_ROLLOVER_HOUR_UTC:
        day = now.date() - timedelta(days=2)
    else:
        day = now.date() - timedelta(days=1)
    return day.strftime("%Y-%m-%d")


def _inject_font_css(font_code: Optional[str], scope_class: Optional[str] = None):
    """Inject a font only for elements within .{scope_class}. Does nothing if no font."""
    if not font_code:
        return
    font = FONT_CATALOG.get(font_code)
    if not font:
        return
    st.markdown(
        f"<link href='{font['import']}' rel='stylesheet'>", unsafe_allow_html=True
    )
    selector = f".{scope_class}" if scope_class else ".cosmetic-font"
    st.markdown(
        f"<style>{selector}{{font-family:{font['css_family']} !important;}}</style>",
        unsafe_allow_html=True,
    )


def _equipped_for_player(pid: str) -> Dict[str, Optional[str]]:
    bag = _get_player_bag(pid)
    eq = bag.get("cosmetics", {}).get("equipped", {})
    return {"font": eq.get("font"), "badge": eq.get("badge"), "title": eq.get("title")}


def _title_text(code: Optional[str]) -> str:
    if not code:
        return ""
    lab = ITEM_DB.get(code, {}).get("label", "")
    # ITEM_DB labels are like "Title: Emoji Hunter" — show just the title text
    return lab.replace("Title: ", "") if lab.startswith("Title: ") else lab


def _row_to_bag(bag_any: Any) -> Dict[str, Any]:
    bag = _as_dict(bag_any)
    bag.setdefault(
        "inventory", {}
    )  # flat counts (e.g., "BOX_GOLD": 2, "EMOJI_APPLE": 5)
    bag.setdefault("cosmetics", {"owned": [], "equipped": {}})
    bag["cosmetics"].setdefault("equipped", {})
    return bag


def _get_player_bag(pid: str) -> Dict[str, Any]:
    if not supabase:
        return _row_to_bag({})
    try:
        res = supabase.table("players").select("bag").eq("id", pid).execute()
        rows = res.data or []
        if rows:
            return _row_to_bag(rows[0].get("bag"))
    except Exception:
        pass
    return _row_to_bag({})


def _save_player_bag(pid: str, bag: Dict[str, Any]) -> bool:
    if not supabase:
        return False
    try:
        supabase.table("players").update({"bag": bag}).eq("id", pid).execute()
        return True
    except Exception as e:
        st.error(f"Failed to save bag: {e}")
        return False


def _bag_add(pid: str, code: str, qty: int = 1) -> bool:
    bag = _get_player_bag(pid)
    inv = bag["inventory"]
    inv[code] = int(inv.get(code, 0)) + int(qty)
    bag["inventory"] = inv
    return _save_player_bag(pid, bag)


def _equip_cosmetic(pid: str, slot: str, code: Optional[str]) -> bool:
    bag = _get_player_bag(pid)
    eq = bag["cosmetics"]["equipped"]
    if code:
        # Only allow equipping owned cosmetics
        if code not in bag["cosmetics"].get("owned", []):
            st.error("You don't own this cosmetic.")
            return False
        eq[slot] = code
    else:
        eq.pop(slot, None)
    bag["cosmetics"]["equipped"] = eq
    return _save_player_bag(pid, bag)


# Simple login gate: PIN + Emoji
def _login_gate(players_df: pd.DataFrame, key_prefix: str) -> Optional[str]:
    st.markdown("### 🔐 Quick Login")
    if players_df.empty:
        st.info("No players yet.")
        return None

    pname = st.selectbox(
        "Player", options=players_df["name"].tolist(), key=f"{key_prefix}_pname"
    )
    pin = st.text_input(
        "4-digit PIN", type="password", max_chars=4, key=f"{key_prefix}_pin"
    )
    emojis = [
        "😀",
        "😎",
        "🤖",
        "🐸",
        "🐼",
        "🦊",
        "🐵",
        "🐱",
        "🐶",
        "🐯",
        "🐻",
        "🐨",
        "🐷",
        "🐮",
        "",
        "🐤",
        "🐙",
        "🐳",
        "🦄",
        "🐲",
        "🍀",
        "🍕",
        "🍩",
        "⚽",
        "🏀",
        "🎲",
        "⭐",
        "🔥",
        "❄️",
        "🌙",
    ]
    chosen = st.selectbox("Emoji", emojis, key=f"{key_prefix}_emoji")

    verified_pid = st.session_state.get(f"{key_prefix}_verified_pid")
    verified_until = st.session_state.get(f"{key_prefix}_verified_until")

    still_valid = bool(verified_pid and verified_until and _utc_now() < verified_until)
    if still_valid:
        st.success("Session verified")
        return verified_pid

    if st.button("Verify", key=f"{key_prefix}_verify"):
        row = players_df.loc[players_df["name"] == pname].head(1)
        if row.empty:
            st.error("Unknown player.")
            return None
        row = row.iloc[0]
        ok_pin = str(row.get("pin_code") or "") == str(pin or "")
        ok_emo = (row.get("emoji_lock") or "") == chosen
        if ok_pin and ok_emo:
            st.session_state[f"{key_prefix}_verified_pid"] = row["id"]
            st.session_state[f"{key_prefix}_verified_until"] = _utc_now() + timedelta(
                minutes=5
            )
            st.success("Verified for 5 minutes")
            return row["id"]
        else:
            st.error("Wrong PIN or emoji.")
            return None
    return None


# Update "best of day" convenience flag after inserting a run
def _set_best_of_day(date_key: str, player_id: str):
    if not supabase:
        return
    try:
        # reset flags for that player/day
        supabase.table("minigame_scores").update({"is_best_for_day": False}).eq(
            "date_key", date_key
        ).eq("player_id", player_id).execute()
        # pick best score (tie: earliest finish wins)
        best = (
            supabase.table("minigame_scores")
            .select("id,score,finished_at")
            .eq("date_key", date_key)
            .eq("player_id", player_id)
            .order("score", desc=True)
            .order("finished_at", desc=False)
            .limit(1)
            .execute()
        )
        rows = best.data or []
        if rows:
            supabase.table("minigame_scores").update({"is_best_for_day": True}).eq(
                "id", rows[0]["id"]
            ).execute()
    except Exception as e:
        st.error(f"Set best-of-day failed: {e}")


# Award yesterday once (on first visit of the day)
def _run_daily_awards_if_needed():
    if not supabase:
        return
    ykey = _yesterday_key_utc()
    try:
        done = (
            supabase.table("minigame_daily_awards")
            .select("id")
            .eq("date_key", ykey)
            .limit(1)
            .execute()
        )
        if done.data or []:
            return  # already awarded
        # best scores yesterday
        rows = (
            supabase.table("minigame_scores")
            .select("player_id, score")
            .eq("date_key", ykey)
            .eq("is_best_for_day", True)
            .order("score", desc=True)
            .execute()
            .data
            or []
        )
        if not rows:
            return
        # rank & buckets
        # Top1 → BOX_GOLD or V_PICK_NEXT_GAME (20% chance)
        # 2–3 → BOX_SILVER
        # Top 10% (min 10) → BOX_BRONZE
        n = len(rows)
        top1 = rows[0:1]
        top23 = rows[1:3]
        topk = max(10, int(np.ceil(0.10 * n)))
        top10p = rows[:topk]

        # award helper
        def _aw(player_id: str, code: str, placement: Optional[int]):
            supabase.table("minigame_daily_awards").insert(
                {
                    "date_key": ykey,
                    "player_id": player_id,
                    "placement": placement,
                    "reward_code": code,
                }
            ).execute()
            _bag_add(player_id, code, 1)

        # top1
        import random

        for i, r in enumerate(top1):
            code = "V_PICK_NEXT_GAME" if random.random() < 0.20 else "BOX_GOLD"
            _aw(r["player_id"], code, 1)
        # 2–3
        for idx, r in enumerate(top23, start=2):
            _aw(r["player_id"], "BOX_SILVER", idx)
        # top10%/top10
        for r in top10p:
            _aw(r["player_id"], "BOX_BRONZE", None)
        st.toast("Daily rewards granted for yesterday 🎁")
    except Exception as e:
        st.warning(f"Daily award check failed (non-blocking): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Locker: items, drops, crafting
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Economy helpers: pity, dup refund, JSON config overrides
# ─────────────────────────────────────────────────────────────────────────────
PITY_RULES = {  # cosmetic is guaranteed on or before these counts
    "BOX_BRONZE": 10,
    "BOX_SILVER": 7,
    "BOX_GOLD": 4,
}
DUPLICATE_REFUND_DUST = {  # dust for dup cosmetics by category
    "font": 60,
    "badge": 40,
    "title": 35,
}


def _cosmetic_kind(code: str) -> str:
    if str(code).startswith("font_"):
        return "font"
    if str(code).startswith("badge_"):
        return "badge"
    if str(code).startswith("title_"):
        return "title"
    return "other"


# ensure pity bucket in bag shape
def _bag_get_pity(pid: str) -> Dict[str, int]:
    bag = _get_player_bag(pid)
    pit = bag.get("pity", {})
    if not isinstance(pit, dict):
        pit = {}
    return {str(k): int(v) for k, v in pit.items()}


def _bag_set_pity(pid: str, pity: Dict[str, int]) -> bool:
    bag = _get_player_bag(pid)
    bag["pity"] = pity
    return _save_player_bag(pid, bag)


# JSON config plumbing (stored in 'config' table)
def _load_json_config(key: str, default_obj):
    if not supabase:
        return default_obj
    try:
        res = supabase.table("config").select("value").eq("key", key).limit(1).execute()
        rows = res.data or []
        if rows and rows[0].get("value"):
            import json

            return json.loads(rows[0]["value"])
    except Exception:
        pass
    return default_obj


def _save_json_config(key: str, obj) -> bool:
    if not supabase:
        st.error("Supabase not configured.")
        return False
    try:
        import json

        supabase.table("config").upsert(
            {"key": key, "value": json.dumps(obj)}
        ).execute()
        return True
    except Exception as e:
        st.error(f"Failed to save '{key}': {e}")
        return False


EMOJI_ITEM_POOL = ["EMOJI_STAR", "EMOJI_FIRE", "EMOJI_APPLE", "EMOJI_CROWN"]


def _item_label(code: str) -> str:
    meta = ITEM_DB.get(code, {"label": code, "icon": "•"})
    return f"{meta.get('icon','•')} {meta.get('label',code)}"


def _bag_count(pid: str, code: str) -> int:
    bag = _get_player_bag(pid)
    return int(bag.get("inventory", {}).get(code, 0))


def _bag_spend(pid: str, code: str, qty: int) -> bool:
    bag = _get_player_bag(pid)
    inv = bag.get("inventory", {})
    have = int(inv.get(code, 0))
    if have < qty:
        return False
    inv[code] = have - qty
    if inv[code] <= 0:
        inv.pop(code, None)
    bag["inventory"] = inv
    return _save_player_bag(pid, bag)


def _bag_add_owned_cosmetic(pid: str, cos_code: str) -> bool:
    bag = _get_player_bag(pid)
    cos = bag.get("cosmetics", {})
    owned = cos.get("owned", [])
    if cos_code not in owned:
        owned.append(cos_code)
    cos["owned"] = owned
    bag["cosmetics"] = cos
    return _save_player_bag(pid, bag)


def _weighted_pick(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    weights = np.array([e["weight"] for e in entries], dtype=float)
    weights = weights / weights.sum()
    idx = np.random.choice(len(entries), p=weights)
    return entries[int(idx)]


def _normalize_qty(q) -> int:
    if isinstance(q, (list, tuple)) and len(q) == 2:
        return int(np.random.randint(int(q[0]), int(q[1]) + 1))
    return int(q)


def _resolve_code(code):
    if isinstance(code, list):
        return str(np.random.choice(code))
    return str(code)


def _give_reward(pid: str, item_code: str, qty: int, is_cosmetic: bool) -> None:
    if is_cosmetic:
        _bag_add_owned_cosmetic(pid, item_code)
    else:
        _bag_add(pid, item_code, qty)


def _open_box_once(pid: str, box_code: str) -> Tuple[str, int, bool]:
    """Single open with pity + duplicate→dust refund. Returns (code, qty, is_cosmetic)."""
    table = BOX_TABLES.get(box_code, [])
    if not table:
        return ("DUST", 1, False)

    # Pity check
    pity = _bag_get_pity(pid)
    pity_count = int(pity.get(box_code, 0))
    force_cosmetic = False
    if box_code in PITY_RULES:
        force_cosmetic = (pity_count + 1) >= int(PITY_RULES[box_code])

    # pick prize
    entries = (
        [e for e in table if e.get("type") == "cosmetic"] if force_cosmetic else table
    )
    prize = _weighted_pick(entries)
    code = _resolve_code(prize["code"])
    qty = _normalize_qty(prize.get("qty", 1))
    is_cosmetic = prize["type"] == "cosmetic"

    # Duplicate conversion
    if is_cosmetic:
        bag = _get_player_bag(pid)
        owned = bag.get("cosmetics", {}).get("owned", [])
        if code in owned:
            kind = _cosmetic_kind(code)
            refund = int(DUPLICATE_REFUND_DUST.get(kind, 25))
            code, qty, is_cosmetic = "DUST", refund, False
            pity[box_code] = pity_count + 1  # still counts toward pity
            _bag_set_pity(pid, pity)
            _bag_add(pid, code, qty)
            return (code, qty, False)
        else:
            # success → reset pity and grant cosmetic
            pity[box_code] = 0
            _bag_set_pity(pid, pity)
            _bag_add_owned_cosmetic(pid, code)
            return (code, qty, True)

    # Non-cosmetic → increment pity and grant item
    pity[box_code] = pity_count + 1
    _bag_set_pity(pid, pity)
    _bag_add(pid, code, qty)
    return (code, qty, False)


def _open_boxes(pid: str, box_code: str, n: int) -> List[Tuple[str, int, bool]]:
    results = []
    for _ in range(int(n)):
        results.append(_open_box_once(pid, box_code))
    _bag_spend(pid, box_code, int(n))  # spend after successful opens
    return results


def _can_craft(pid: str, cos_code: str) -> bool:
    need = CRAFT_RECIPES.get(cos_code, {})
    return all(_bag_count(pid, k) >= v for k, v in need.items())


def _craft(pid: str, cos_code: str) -> bool:
    if cos_code not in CRAFT_RECIPES:
        return False
    need = CRAFT_RECIPES[cos_code]
    if not _can_craft(pid, cos_code):
        return False
    # spend
    for k, v in need.items():
        if not _bag_spend(pid, k, v):
            return False
    # grant cosmetic
    return _bag_add_owned_cosmetic(pid, cos_code)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults & Config Helpers
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    "starting_elo": 1000.0,
    "k_ffa_base": 32.0,
    "k_team": 28.0,
    "coop_drift": 0.0,
    "solo_drift": 2.0,
    "use_points_weight": True,
    "points_alpha": 0.5,
    "gap_bonus": True,
    "gap_bonus_coeff": 0.15,
    "exceptional_bonus": True,
    "exceptional_threshold": 0.33,  # normalized points margin
    "exceptional_points": 5.0,
}


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")


def load_config() -> Dict[str, Any]:
    """Load configuration from Supabase 'config' table; fall back to defaults if missing."""
    cfg = DEFAULT_CONFIG.copy()
    if not supabase:
        return cfg
    try:
        res = supabase.table("config").select("*").execute()
        rows = res.data or []
        for r in rows:
            k = r.get("key")
            v = r.get("value")
            if k in cfg:
                # Cast by key
                if k in ("use_points_weight", "gap_bonus", "exceptional_bonus"):
                    cfg[k] = _parse_bool(v)
                elif k in (
                    "starting_elo",
                    "k_ffa_base",
                    "k_team",
                    "coop_drift",
                    "solo_drift",
                    "points_alpha",
                    "gap_bonus_coeff",
                    "exceptional_threshold",
                    "exceptional_points",
                ):
                    try:
                        cfg[k] = float(v)
                    except Exception:
                        pass
                else:
                    cfg[k] = v
    except Exception:
        # If table missing, just use defaults (no crash)
        pass
    return cfg


def save_config(cfg: Dict[str, Any]) -> bool:
    """Upsert key/value pairs into config table."""
    if not supabase:
        st.error("Supabase not configured.")
        return False
    payload = [{"key": k, "value": str(v)} for k, v in cfg.items()]
    try:
        supabase.table("config").upsert(payload).execute()
        return True
    except Exception as e:
        st.error(f"Failed to save config: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Data Utilities
# ─────────────────────────────────────────────────────────────────────────────


def _uuid() -> str:
    return str(uuid.uuid4())


@st.cache_data(ttl=30)
def sb_select(table: str) -> pd.DataFrame:
    """Fetch all rows from a table as DataFrame (cached)."""
    if not supabase:
        return pd.DataFrame()
    try:
        res = supabase.table(table).select("*").execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        st.error(f"Failed to fetch {table}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=5)
def sb_select_sessions_joined() -> pd.DataFrame:
    """Join session_players × sessions × players × games into one convenient frame."""
    sessions = sb_select("sessions")
    session_players = sb_select("session_players")
    players = sb_select("players")
    games = sb_select("games")

    if sessions.empty or players.empty or games.empty or session_players.empty:
        return pd.DataFrame()

    # Normalize date columns
    if "played_at" in sessions.columns:
        sessions["played_at"] = pd.to_datetime(sessions["played_at"], errors="coerce")

    # Merge chain
    joined = (
        session_players.merge(
            sessions.rename(columns={"id": "session_id_"}),
            left_on="session_id",
            right_on="session_id_",
            how="left",
            suffixes=("", "_s"),
        )
        .merge(
            # include nickname for display
            players.rename(
                columns={
                    "id": "player_id_",
                    "name": "player_name",
                    "nickname": "player_nick",
                }
            ),
            left_on="player_id",
            right_on="player_id_",
            how="left",
        )
        .merge(
            games.rename(columns={"id": "game_id_", "name": "game_name"}),
            left_on="game_id",
            right_on="game_id_",
            how="left",
        )
    )

    # Clean columns
    drop_cols = [
        c for c in ["session_id_", "player_id_", "game_id_"] if c in joined.columns
    ]
    if drop_cols:
        joined = joined.drop(columns=drop_cols)

    # Types
    if "position" in joined.columns:
        joined["position"] = pd.to_numeric(joined["position"], errors="coerce")
    if "points" in joined.columns:
        joined["points"] = pd.to_numeric(joined["points"], errors="coerce")
    if "is_winner" in joined.columns:
        joined["is_winner"] = joined["is_winner"].fillna(False).astype(bool)
    if "team" in joined.columns:
        joined["team"] = joined["team"].fillna("").astype(str).str.strip().str.upper()
    if "player_nick" in joined.columns:
        joined["player_nick"] = joined["player_nick"].fillna("").astype(str)

    return joined


def sb_insert(table: str, data: Dict[str, Any]) -> bool:
    if not supabase:
        st.error("Supabase not configured.")
        return False
    try:
        supabase.table(table).insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Insert into {table} failed: {e}")
        return False


def sb_delete_by_id(table: str, _id: str) -> bool:
    if not supabase:
        st.error("Supabase not configured.")
        return False
    try:
        supabase.table(table).delete().eq("id", _id).execute()
        return True
    except Exception as e:
        st.error(f"Delete from {table} failed: {e}")
        return False


def sb_delete_session(session_id: str) -> bool:
    """Delete a session and all its participant rows (no DB cascade required)."""
    if not supabase:
        st.error("Supabase not configured.")
        return False
    try:
        # 1) Remove participants for this session (safe even if none)
        supabase.table("session_players").delete().eq(
            "session_id", session_id
        ).execute()
        # 2) Remove the session itself
        supabase.table("sessions").delete().eq("id", session_id).execute()
        return True
    except Exception as e:
        st.error(f"Delete session failed: {e}")
        return False


def sb_update_by_id(table: str, _id: str, data: Dict[str, Any]) -> bool:
    if not supabase:
        st.error("Supabase not configured.")
        return False
    try:
        supabase.table(table).update(data).eq("id", _id).execute()
        return True
    except Exception as e:
        st.error(f"Update {table} failed: {e}")
        return False


def _as_dict(x):
    """Ensure JSON payload is a dict (Supabase returns dict; safe-guard strings)."""
    if isinstance(x, dict):
        return x
    try:
        import json

        return json.loads(x or "{}")
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Winner resolution
# ─────────────────────────────────────────────────────────────────────────────


def resolve_winners(df_session_participants: pd.DataFrame) -> pd.Series:
    """
    Return a boolean Series (index aligned) for winners of that session.
    Priority: explicit is_winner → (for competitive sessions) lowest position → highest points → none.
    For co-op or solo sessions with no explicit winners, no one wins.
    """
    df = df_session_participants
    if df.empty:
        return pd.Series(dtype=bool)

    idx = df.index

    # 1) Respect explicit flags if any exist
    if "is_winner" in df.columns and df["is_winner"].fillna(False).any():
        return df["is_winner"].fillna(False)

    n = len(df)

    # 2) Guard: solo or co-op (single non-empty team label for multiple players) → no winners if none flagged
    teams_clean = (
        df["team"].fillna("").astype(str).str.strip().str.upper()
        if "team" in df.columns
        else pd.Series([""] * n, index=idx)
    )
    non_empty_labels = [t for t in teams_clean.unique().tolist() if t != ""]

    if n == 1:
        # Solo with no explicit winner = loss → no winner
        return pd.Series([False], index=idx)

    if len(non_empty_labels) == 1:
        # Everyone on the same non-empty team (e.g., COOP) and no explicit winners → no winner
        return pd.Series([False] * n, index=idx)

    # 3) Competitive fallback: positions, then points
    if "position" in df.columns and df["position"].notna().any():
        min_pos = df["position"].min()
        return df["position"].eq(min_pos)

    if "points" in df.columns and df["points"].notna().any():
        max_pts = df["points"].max()
        return df["points"].eq(max_pts)

    # 4) Nothing else → no winners
    return pd.Series([False] * n, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# ELO v2.7 (FFA, Team, Co-op, Solo)
# ─────────────────────────────────────────────────────────────────────────────


def _elo_expected(r_a: float, r_b: float) -> float:
    qa = 10 ** (r_a / 400.0)
    qb = 10 ** (r_b / 400.0)
    return qa / (qa + qb)


def _norm_margin(a: float, b: float, pmin: float, pmax: float) -> float:
    """Normalized margin in [-1, 1] from points a-b using session range."""
    if pd.isna(a) or pd.isna(b) or pd.isna(pmin) or pd.isna(pmax) or pmax <= pmin:
        return 0.0
    return float(np.clip((a - b) / max(1e-9, (pmax - pmin)), -1.0, 1.0))


def compute_leaderboard(
    joined: pd.DataFrame,
    game_filter: Optional[List[str]],
    date_range: Optional[Tuple[date, date]],
    *,
    base_rating: float,
    k_ffa_base: float,
    k_team: float,
    coop_drift: float,
    solo_drift: float,
    use_points_weight: bool,
    points_alpha: float,
    gap_bonus: bool,
    gap_bonus_coeff: float,
    exceptional_bonus: bool,
    exceptional_threshold: float,
    exceptional_points: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (leaderboard_df, h2h_df, recent_df, ratings_over_time_df) using the provided ELO configuration.
    """
    EMPTY = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if joined.empty:
        return EMPTY

    df = joined.copy()

    if game_filter is not None:
        if len(game_filter) == 0:
            # Return empty tables if no games selected
            return EMPTY
        df = df[df["game_name"].isin(game_filter)]

    # Ensure 'played_at' is a datetime column
    if "played_at" in df.columns:
        df["played_at"] = pd.to_datetime(df["played_at"], errors="coerce")

    if date_range and all(date_range):
        start, end = _range_to_df_tz(df["played_at"], date_range[0], date_range[1])
        df = df[df["played_at"].between(start, end)]

    if df.empty:
        return EMPTY

    # Resolve winners per-session
    df = df.sort_values(["played_at", "session_id"]).reset_index(drop=True)
    df["resolved_winner"] = False
    for sid, g in df.groupby("session_id", sort=False):
        mask = resolve_winners(g)
        df.loc[g.index, "resolved_winner"] = mask.values

    # Aggregate per player
    gp = (
        df.groupby("player_id")
        .agg(
            player_name=("player_name", "first"),
            games_played=("session_id", pd.Series.nunique),
            wins=("resolved_winner", "sum"),
            avg_position=("position", "mean"),
            avg_points=("points", "mean"),
            last_played=("played_at", "max"),
        )
        .reset_index()
    )
    gp["win_rate"] = (gp["wins"] / gp["games_played"]).fillna(0.0)

    # Add nickname + display name
    if "player_nick" in df.columns:
        nick_map = df.groupby("player_id")["player_nick"].first()
        gp["player_nick"] = gp["player_id"].map(nick_map).fillna("")
    else:
        gp["player_nick"] = ""
    gp["display_name"] = np.where(
        gp["player_nick"].str.strip() != "",
        gp["player_name"].astype(str) + " (" + gp["player_nick"].astype(str) + ")",
        gp["player_name"].astype(str),
    )
    disp_map = dict(zip(gp["player_id"], gp["display_name"]))
    ratings_timeline: List[Dict[str, Any]] = []

    # ELO calculations
    ratings: Dict[str, float] = {
        pid: float(base_rating) for pid in gp["player_id"].tolist()
    }

    for sid, g in df.groupby("session_id", sort=False):
        teams = g["team"].fillna("").astype(str).str.upper().str.strip()
        distinct_teams = sorted([t for t in teams.unique() if t])
        has_positions = "position" in g.columns and g["position"].notna().any()
        winners_mask = g["resolved_winner"].values

        # Co-op heuristic
        is_coop_like = (
            (teams.nunique() <= 1)
            and (winners_mask.all() or (~g["resolved_winner"]).all())
        ) or (
            (not has_positions)
            and (winners_mask.all() or (~g["resolved_winner"]).all())
        )

        # Solo detection
        is_solo = (len(g) == 1) or (teams.nunique() == 1 and g.shape[0] == 1)

        # Points range for MoV
        pmin = (
            g["points"].min()
            if "points" in g.columns and g["points"].notna().any()
            else np.nan
        )
        pmax = (
            g["points"].max()
            if "points" in g.columns and g["points"].notna().any()
            else np.nan
        )

        if len(distinct_teams) >= 2:
            # TEAM MODE
            team_groups = {
                t: g[teams == t]["player_id"].tolist() for t in distinct_teams
            }
            winners_per_team = {
                t: g[(teams == t) & (g["resolved_winner"])].shape[0] > 0
                for t in distinct_teams
            }
            n_winner_teams = sum(1 for v in winners_per_team.values() if v)

            for i in range(len(distinct_teams)):
                for j in range(i + 1, len(distinct_teams)):
                    ta, tb = distinct_teams[i], distinct_teams[j]
                    mem_a = team_groups[ta]
                    mem_b = team_groups[tb]
                    if not mem_a or not mem_b:
                        continue

                    r_a = float(np.mean([ratings[m] for m in mem_a]))
                    r_b = float(np.mean([ratings[m] for m in mem_b]))
                    e_a = _elo_expected(r_a, r_b)

                    # Determine team outcome
                    if n_winner_teams == 0:
                        s_a, s_b = 0.5, 0.5
                    elif winners_per_team[ta] and winners_per_team[tb]:
                        s_a, s_b = 0.5, 0.5
                    elif winners_per_team[ta]:
                        s_a, s_b = 1.0, 0.0
                    elif winners_per_team[tb]:
                        s_a, s_b = 0.0, 1.0
                    else:
                        if has_positions:
                            avg_a = g[teams == ta]["position"].mean()
                            avg_b = g[teams == tb]["position"].mean()
                            if pd.notna(avg_a) and pd.notna(avg_b):
                                if avg_a < avg_b:
                                    s_a, s_b = 1.0, 0.0
                                elif avg_a > avg_b:
                                    s_a, s_b = 0.0, 1.0
                                else:
                                    s_a, s_b = 0.5, 0.5
                            else:
                                s_a, s_b = 0.5, 0.5
                        else:
                            s_a, s_b = 0.5, 0.5

                    # MoV via points
                    k_mult = 1.0
                    if use_points_weight and "points" in g.columns:
                        pa = g[teams == ta]["points"].mean()
                        pb = g[teams == tb]["points"].mean()
                        mv = abs(_norm_margin(pa, pb, pmin, pmax))
                        k_mult += points_alpha * mv

                    for m in mem_a:
                        ratings[m] += (k_team * k_mult * (s_a - e_a)) / max(
                            1, len(mem_a)
                        )
                    for m in mem_b:
                        ratings[m] += (k_team * k_mult * ((1 - s_a) - (1 - e_a))) / max(
                            1, len(mem_b)
                        )

        elif is_solo:
            # SOLO MODE — drift vs system
            drift = float(solo_drift)
            won = bool(g["resolved_winner"].iloc[0])
            pid = g["player_id"].iloc[0]
            ratings[pid] += drift if won else -drift

        elif is_coop_like:
            # CO-OP MODE — optional drift
            drift = float(coop_drift)
            if abs(drift) > 1e-9:
                all_win = winners_mask.all()
                all_lose = (~g["resolved_winner"]).all()
                if all_win or all_lose:
                    val = drift if all_win else -drift
                    for pid in g["player_id"].tolist():
                        ratings[pid] += val
        else:
            # FFA MODE — pairwise by positions (+ optional points & gap bonuses)
            pids = g["player_id"].tolist()
            n = len(pids)
            if n < 2:
                continue

            Kpair = k_ffa_base / max(1, n - 1)

            g2 = g.copy()
            if g2["position"].isna().all():
                g2["position"] = (~g2["resolved_winner"]).astype(int) + 1

            for i in range(n):
                for j in range(i + 1, n):
                    a = pids[i]
                    b = pids[j]
                    pa = g2.loc[g2["player_id"] == a, "position"].iloc[0]
                    pb = g2.loc[g2["player_id"] == b, "position"].iloc[0]
                    if pd.isna(pa) or pd.isna(pb):
                        continue

                    if pa == pb:
                        s_a, s_b = 0.5, 0.5
                    else:
                        s_a, s_b = (1.0, 0.0) if pa < pb else (0.0, 1.0)

                    # K multiplier
                    k_mult = 1.0
                    if (
                        use_points_weight
                        and "points" in g2.columns
                        and g2["points"].notna().any()
                    ):
                        pts_a = g2.loc[g2["player_id"] == a, "points"].iloc[0]
                        pts_b = g2.loc[g2["player_id"] == b, "points"].iloc[0]
                        mv = abs(_norm_margin(pts_a, pts_b, pmin, pmax))
                        k_mult += points_alpha * mv

                    if gap_bonus and pd.notna(pa) and pd.notna(pb):
                        gap = abs(pb - pa)
                        if gap > 1:
                            k_mult += gap_bonus_coeff * (gap - 1)

                    e_a = _elo_expected(ratings[a], ratings[b])
                    ratings[a] += Kpair * k_mult * (s_a - e_a)
                    ratings[b] += Kpair * k_mult * (s_b - (1 - e_a))

            # Exceptional performance bonus (one-threshold version)
            if (
                exceptional_bonus
                and n >= 4
                and "points" in g.columns
                and g["points"].notna().any()
            ):
                g2 = g2.sort_values("position", ascending=True)
                top = g2.iloc[0]
                others = g2.iloc[1:]
                top_pts = top["points"]
                second_pts = others["points"].max() if not others.empty else np.nan
                mv = abs(_norm_margin(top_pts, second_pts, pmin, pmax))
                if mv >= float(exceptional_threshold) and pd.notna(top["player_id"]):
                    ratings[top["player_id"]] += float(exceptional_points)

        played_at_dt = g["played_at"].iloc[0]
        for pid in g["player_id"].tolist():
            ratings_timeline.append(
                {
                    "played_at": played_at_dt,
                    "player_id": pid,
                    "player_name": disp_map.get(pid, str(pid)),
                    "elo": ratings[pid],
                }
            )

    gp["elo"] = gp["player_id"].map(ratings)

    # Streaks
    def _streaks(pid: str) -> Tuple[int, int]:
        x = df[df["player_id"] == pid].sort_values(["played_at", "session_id"])
        cur = best = 0
        for _, row in x.iterrows():
            if bool(row["resolved_winner"]):
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return cur, best

    streaks = [_streaks(pid) for pid in gp["player_id"].tolist()]
    if streaks:
        gp["current_streak"], gp["best_streak"] = zip(*streaks)
    else:
        gp["current_streak"], gp["best_streak"] = [], []

    # Order leaderboard — rank by highest ELO (ties: more wins, then higher win%)
    lb = gp.sort_values(
        ["elo", "wins", "win_rate"], ascending=[False, False, False]
    ).reset_index(drop=True)

    # Head-to-Head (times row finished ahead of column)
    ids = lb["player_id"].tolist()
    idx = {pid: i for i, pid in enumerate(ids)}
    h2h = np.zeros((len(ids), len(ids)), dtype=int)

    for sid, g in df.groupby("session_id"):
        g2 = g.copy()
        if g2["position"].isna().all():
            g2["position"] = (~g2["resolved_winner"]).astype(int) + 1
        for i in range(len(g2)):
            for j in range(len(g2)):
                if i == j:
                    continue
                a = g2.iloc[i]
                b = g2.iloc[j]
                if a["player_id"] in idx and b["player_id"] in idx:
                    if (
                        pd.notna(a["position"])
                        and pd.notna(b["position"])
                        and a["position"] < b["position"]
                    ):
                        h2h[idx[a["player_id"]], idx[b["player_id"]]] += 1

    h2h_df = pd.DataFrame(
        h2h, index=lb["display_name"], columns=lb["display_name"]
    ).astype(int)

    # Recent sessions (safe column selection)
    recent_ids = (
        df.sort_values(["played_at", "session_id"], ascending=[False, False])[
            "session_id"
        ]
        .drop_duplicates()
        .head(15)
    )
    recent = df[df["session_id"].isin(recent_ids)].copy()
    winners_names = (
        recent[recent["resolved_winner"]]
        .groupby("session_id")["player_name"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .rename("winners")
    )
    base_cols = ["session_id", "played_at", "game_name", "location", "notes"]
    existing_cols = [c for c in base_cols if c in recent.columns]
    recent_slim = (
        recent.drop_duplicates("session_id")[existing_cols]
        .merge(winners_names, left_on="session_id", right_index=True, how="left")
        .sort_values(["played_at"], ascending=False)
    )

    ratings_over_time = pd.DataFrame(ratings_timeline).sort_values(
        ["played_at", "player_id"]
    )

    return lb, h2h_df, recent_slim, ratings_over_time


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────


def render_h2h_heatmap(h2h_df: pd.DataFrame, title: str = "Head-to-Head") -> None:
    """Render a Plotly heatmap with numbers in each cell; diagonal blank."""
    if h2h_df.empty:
        st.info("No head-to-head data for current filters.")
        return

    # Mask diagonal for nicer look
    z = h2h_df.values.astype(float)
    text = h2h_df.astype(str).values
    for i in range(len(h2h_df)):
        z[i, i] = np.nan
        text[i, i] = ""

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=h2h_df.columns.tolist(),
            y=h2h_df.index.tolist(),
            colorscale="Blues",
            reversescale=False,
            colorbar=dict(title="Wins"),
            hovertemplate="<b>%{y}</b> finished ahead of <b>%{x}</b>: %{z}<extra></extra>",
            zmin=0,
        )
    )

    # Add numbers as annotations
    annotations = []
    for i, row in enumerate(text):
        for j, val in enumerate(row):
            if val != "":
                annotations.append(
                    go.layout.Annotation(
                        text=val,
                        x=h2h_df.columns[j],
                        y=h2h_df.index[i],
                        xref="x",
                        yref="y",
                        showarrow=False,
                        font=dict(color="black"),
                    )
                )
    fig.update_layout(
        title=title,
        annotations=annotations,
        xaxis=dict(title="", side="top", tickangle=45),
        yaxis=dict(title="", autorange="reversed"),
        margin=dict(l=0, r=0, t=60, b=0),
        height=400 + 20 * len(h2h_df),
    )
    st.plotly_chart(fig, use_container_width=True)


# TZ helper function
def _range_to_df_tz(series: pd.Series, start_d, end_d):
    s = pd.to_datetime(start_d)
    e = pd.to_datetime(end_d) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    # Old (deprecated):
    # if pd.api.types.is_datetime64tz_dtype(series):

    # New:
    if isinstance(getattr(series, "dtype", None), pd.DatetimeTZDtype):
        tz = series.dt.tz
        if s.tzinfo is None:
            s = s.tz_localize(tz)
        else:
            s = s.tz_convert(tz)
        if e.tzinfo is None:
            e = e.tz_localize(tz)
        else:
            e = e.tz_convert(tz)
    return s, e


# ─────────────────────────────────────────────────────────────────────────────
# Daily mini-game: Odd Emoji Out (OEO)
# ─────────────────────────────────────────────────────────────────────────────

OEO_PAIRS = [
    ("🍎", "🍏"),
    ("😃", "😄"),
    ("😺", "😸"),
    ("🐶", "🐺"),
    ("🐻", "🐻‍❄️"),
    ("⭐", "✨"),
    ("🟦", "🟩"),
    ("🍋", "🍊"),
    ("🌕", "🌖"),
    ("💙", "💜"),
]


def _oeo_new_round_state(state: Dict[str, Any]):
    # escalate difficulty with score
    score = int(state.get("score", 0))
    n = min(OEO_MAX_GRID, OEO_START_GRID + score // 2)
    pair_idx = np.random.randint(0, len(OEO_PAIRS))
    base, odd = OEO_PAIRS[pair_idx]
    target_r = np.random.randint(0, n)
    target_c = np.random.randint(0, n)
    state.update(
        {
            "grid_n": n,
            "pair": (
                (base, odd) if np.random.rand() < 0.5 else (odd, base)
            ),  # flip sometimes
            "target": (int(target_r), int(target_c)),
            "round_id": _uuid(),
        }
    )
    return state


def _oeo_finalize_and_save(state: Dict[str, Any], player_id: str):
    if not supabase:
        return
    try:
        date_key = _today_key_utc()
        started_at = state.get("started_at")
        finished_at = _utc_now()
        score = int(state.get("score", 0))
        rounds = int(state.get("rounds_played", 0))
        supabase.table("minigame_scores").insert(
            {
                "date_key": date_key,
                "player_id": player_id,
                "score": score,
                "rounds_played": rounds,
                "duration_s": OEO_DURATION_S,
                "started_at": started_at.isoformat() if started_at else None,
                "finished_at": finished_at.isoformat(),
                "is_best_for_day": False,
                "meta": {"minigame": MINIGAME_NAME},
            }
        ).execute()
        _set_best_of_day(date_key, player_id)
    except Exception as e:
        st.error(f"Failed to save score: {e}")


def render_daily_game():
    st.header("🎮 Daily Mini-Game — Odd Emoji Out")
    _run_daily_awards_if_needed()  # award yesterday (idempotent)

    players_df = sb_select("players")

    # Login gate
    pid = _login_gate(players_df, key_prefix="oeo")
    if not pid:
        st.info("Login to play.")
        return

    # Show personal best today
    today_key = _today_key_utc()
    best_today = 0
    try:
        if supabase:
            r = (
                supabase.table("minigame_scores")
                .select("score")
                .eq("date_key", today_key)
                .eq("player_id", pid)
                .eq("is_best_for_day", True)
                .order("score", desc=True)
                .limit(1)
                .execute()
            )
            rows = r.data or []
            if rows:
                best_today = int(rows[0]["score"])
    except Exception:
        pass

    c_head = st.container()
    with c_head:
        colA, colB, colC = st.columns([2, 1, 1])

        # ⬇️ use only the styled header (delete the earlier unstyled colA.subheader line)
        eq = _equipped_for_player(pid)
        scope = f"pfont-{pid}"
        _inject_font_css(eq.get("font"), scope)
        colA.subheader(
            f"<span class='{scope}'>Hi, {players_df.set_index('id').loc[pid, 'name']}!</span>",
            unsafe_allow_html=True,
        )

        colB.metric("Best today", best_today)
        colC.metric("Resets (UTC)", f"{DAY_ROLLOVER_HOUR_UTC:02d}:00")

    # Load font cosmetic for header
    eq = _equipped_for_player(pid)
    scope = f"pfont-{pid}"
    _inject_font_css(eq.get("font"), scope)
    colA.subheader(
        f"<span class='{scope}'>Hi, {players_df.set_index('id').loc[pid, 'name']}!</span>",
        unsafe_allow_html=True,
    )
    # Run state
    state = st.session_state.setdefault("oeo_state", {})
    running = bool(state.get("running", False))
    cooldown_until = state.get("cooldown_until")

    # start button (with concurrency lock: 70s after last start)
    now = _utc_now()
    locked = bool(cooldown_until and now < cooldown_until)
    if not running:
        st.markdown("Find the **one** odd emoji in the grid. You have **60 seconds**.")
        st.caption("Tip: difficulty ramps up as you score more.")
        if locked:
            remain = int((cooldown_until - now).total_seconds())
            st.warning(f"Please wait {remain}s before starting another run.")
        if st.button("▶️ Start 60-second run", disabled=locked):
            state.clear()
            state.update(
                {
                    "running": True,
                    "score": 0,
                    "rounds_played": 0,
                    "started_at": _utc_now(),
                    "cooldown_until": _utc_now() + timedelta(seconds=70),
                }
            )
            _oeo_new_round_state(state)
            st.rerun()
        st.divider()
    else:
        # Running UI
        started_at = state.get("started_at", _utc_now())
        time_left = OEO_DURATION_S - int((_utc_now() - started_at).total_seconds())
        if time_left <= 0:
            # time up
            state["running"] = False
            _oeo_finalize_and_save(state, pid)
            st.success(f"Time! Final score: {state.get('score', 0)}")
            st.balloons()
        else:
            # Header bar with countdown + score
            b1, b2 = st.columns([1, 1])
            b1.subheader(f"⏳ {time_left}s left")
            b2.subheader(f"🏅 Score: {state.get('score',0)}")

            n = state.get("grid_n", OEO_START_GRID)
            base, odd = state.get("pair", ("🍎", "🍏"))
            tr, tc = state.get("target", (0, 0))
            rid = state.get("round_id")

            # draw grid of buttons
            clicked = None
            for r in range(n):
                cols = st.columns(n)
                for c in range(n):
                    is_target = r == tr and c == tc
                    em = odd if is_target else base
                    if cols[c].button(em, key=f"oeo_btn_{rid}_{r}_{c}"):
                        clicked = (r, c)

            # handle click
            if clicked is not None:
                state["rounds_played"] = int(state.get("rounds_played", 0)) + 1
                if clicked == (tr, tc):
                    state["score"] = int(state.get("score", 0)) + 1
                _oeo_new_round_state(state)
                st.rerun()

    # Daily leaderboard (best of day)
    st.subheader("🏆 Today’s Leaderboard")
    try:
        if supabase:
            best_rows = (
                supabase.table("minigame_scores")
                .select("player_id, score")
                .eq("date_key", today_key)
                .eq("is_best_for_day", True)
                .order("score", desc=True)
                .limit(100)
                .execute()
                .data
                or []
            )
            if best_rows:
                df = pd.DataFrame(best_rows)
                names = sb_select("players")[["id", "name"]]
                df = df.merge(
                    names.rename(columns={"id": "player_id"}),
                    on="player_id",
                    how="left",
                )
                df = df.rename(columns={"name": "Player", "score": "Score"})
                df.insert(0, "#", np.arange(1, len(df) + 1))
                st.dataframe(
                    df[["#", "Player", "Score"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No scores yet today.")
    except Exception as e:
        st.error(f"Leaderboard failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Locker tab UI
# ─────────────────────────────────────────────────────────────────────────────
def render_locker():
    st.title("🎒 Locker — Open, Craft, Equip")

    players_df = sb_select("players")
    pid = _login_gate(players_df, key_prefix="locker")
    if not pid:
        st.info("Login to manage your locker.")
        return

    # Load bag & cosmetics
    bag = _get_player_bag(pid)
    inv = bag.get("inventory", {})
    cos = bag.get("cosmetics", {})
    owned = cos.get("owned", [])
    eq = cos.get("equipped", {})

    # Font preview
    _inject_font_css(eq.get("font"))

    # Nameplate
    pname = players_df.set_index("id").loc[pid, "name"]
    scope = f"pfont-{pid}"
    _inject_font_css(eq.get("font"), scope)
    st.markdown(
        f"""
        <div class="player-nameplate {scope}" style="
            padding:12px 16px;border-radius:10px;background:rgba(240,240,240,0.6);
            border:2px solid #ddd;display:inline-block;">
            <div style="font-size:20px;font-weight:700;">{pname}</div>
            <div style="font-size:13px;opacity:0.8;">
                {ITEM_DB.get(eq.get('badge',''),{}).get('icon','')} {ITEM_DB.get(eq.get('badge',''),{}).get('label','No badge')}
                • {ITEM_DB.get(eq.get('title',''),{}).get('label','No title')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    tabs = st.tabs(["Inventory", "Open Boxes", "Crafting", "Equip"])

    # ============ Inventory ============
    with tabs[0]:
        if inv:
            df = pd.DataFrame(
                [
                    {"Item": _item_label(k), "Code": k, "Qty": v}
                    for k, v in sorted(inv.items())
                ]
            )
            st.dataframe(
                df[["Item", "Qty", "Code"]], use_container_width=True, hide_index=True
            )
        else:
            st.caption("Your inventory is empty.")
        st.caption(
            "Voucher items (e.g., 🎟️) appear here until an admin honors them in person."
        )

    # ============ Open Boxes ============
    with tabs[1]:
        st.subheader("Open Boxes")
        counts = {k: inv.get(k, 0) for k in ["BOX_BRONZE", "BOX_SILVER", "BOX_GOLD"]}
        c1, c2, c3 = st.columns(3)
        c1.metric(_item_label("BOX_BRONZE"), counts["BOX_BRONZE"])
        c2.metric(_item_label("BOX_SILVER"), counts["BOX_SILVER"])
        c3.metric(_item_label("BOX_GOLD"), counts["BOX_GOLD"])

        box_choice = st.radio(
            "Choose box",
            ["BOX_BRONZE", "BOX_SILVER", "BOX_GOLD"],
            format_func=_item_label,
            horizontal=True,
        )
        max_open = int(min(25, counts.get(box_choice, 0)))
        qty = st.number_input(
            "How many to open",
            min_value=1,
            max_value=max_open if max_open > 0 else 1,
            value=min(5, max_open if max_open > 0 else 1),
            step=1,
        )
        if max_open <= 0:
            st.warning("You have none of this box.")
        else:
            if st.button("Open now ✨"):
                results = _open_boxes(pid, box_choice, int(qty))
                # Aggregate results
                agg: Dict[str, Dict[str, Any]] = {}
                cos_drops = 0
                for code, q, is_cos in results:
                    d = agg.setdefault(code, {"qty": 0, "cos": is_cos})
                    d["qty"] += q
                    if is_cos:
                        cos_drops += 1
                df = pd.DataFrame(
                    [
                        {
                            "Drop": _item_label(k),
                            "Qty": v["qty"],
                            "Type": "Cosmetic" if v["cos"] else "Item",
                        }
                        for k, v in agg.items()
                    ]
                ).sort_values(["Type", "Drop"], ascending=[False, True])
                st.success("Opened! Here’s what you got:")
                st.dataframe(df, use_container_width=True, hide_index=True)
                if cos_drops:
                    st.balloons()

    # ============ Crafting ============
    with tabs[2]:
        st.subheader("Craft cosmetics from items")
        if not CRAFT_RECIPES:
            st.caption("No recipes yet.")
        else:
            for cos_code, req in CRAFT_RECIPES.items():
                have = " / ".join(
                    [
                        f"{_item_label(k)} {min(_bag_count(pid,k),v)}/{v}"
                        for k, v in req.items()
                    ]
                )
                cA, cB = st.columns([3, 1])
                with cA:
                    st.markdown(f"**{_item_label(cos_code)}**  \nRequires: {have}")
                can = _can_craft(pid, cos_code)
                disabled = not can
                if cB.button("Craft", key=f"craft_{cos_code}", disabled=disabled):
                    if _craft(pid, cos_code):
                        st.success(f"Crafted {_item_label(cos_code)}!")
                        st.balloons()
                    else:
                        st.error("Craft failed. Check materials.")

    # ============ Equip ============
    with tabs[3]:
        st.subheader("Equip your look")
        # Owned buckets
        owned_fonts = [c for c in owned if c.startswith("font_")]
        owned_badges = [c for c in owned if c.startswith("badge_")]
        owned_titles = [c for c in owned if c.startswith("title_")]

        # Fonts
        st.markdown("**Font**")
        if owned_fonts:
            sel_f = st.selectbox(
                "Choose font",
                options=owned_fonts,
                format_func=_item_label,
                index=(
                    owned_fonts.index(eq.get("font"))
                    if eq.get("font") in owned_fonts
                    else 0
                ),
                key="equip_font_sel",
            )
            e1, e2 = st.columns([1, 1])
            if e1.button("Equip font", key="equip_font_btn"):
                if _equip_cosmetic(pid, "font", sel_f):
                    st.success("Font equipped.")
                    _inject_font_css(sel_f)
            if e2.button("Unequip font", key="unequip_font_btn"):
                if _equip_cosmetic(pid, "font", None):
                    st.success("Font removed.")
        else:
            st.caption("You don’t own any font cosmetics yet.")

        st.markdown("---")

        # Badge
        st.markdown("**Badge**")
        if owned_badges:
            sel_b = st.selectbox(
                "Choose badge",
                options=owned_badges,
                format_func=_item_label,
                index=(
                    owned_badges.index(eq.get("badge"))
                    if eq.get("badge") in owned_badges
                    else 0
                ),
                key="equip_badge_sel",
            )
            b1, b2 = st.columns([1, 1])
            if b1.button("Equip badge", key="equip_badge_btn"):
                if _equip_cosmetic(pid, "badge", sel_b):
                    st.success("Badge equipped.")
            if b2.button("Unequip badge", key="unequip_badge_btn"):
                if _equip_cosmetic(pid, "badge", None):
                    st.success("Badge removed.")
        else:
            st.caption("No badges yet. Craft or find them in boxes.")

        st.markdown("---")

        # Title
        st.markdown("**Title**")
        if owned_titles:
            sel_t = st.selectbox(
                "Choose title",
                options=owned_titles,
                format_func=_item_label,
                index=(
                    owned_titles.index(eq.get("title"))
                    if eq.get("title") in owned_titles
                    else 0
                ),
                key="equip_title_sel",
            )
            t1, t2 = st.columns([1, 1])
            if t1.button("Equip title", key="equip_title_btn"):
                if _equip_cosmetic(pid, "title", sel_t):
                    st.success("Title equipped.")
            if t2.button("Unequip title", key="unequip_title_btn"):
                if _equip_cosmetic(pid, "title", None):
                    st.success("Title removed.")
        else:
            st.caption("No titles yet. Craft or find them in boxes.")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["General", "Moderator", "Admin"], horizontal=True)

    if mode == "Moderator":
        mpw = st.text_input("Moderator password", type="password")
        moderator_name = st.text_input("Your name (shows in approvals)", "")
        if mpw != MODERATOR_PASSWORD:
            st.warning(
                "Enter the correct moderator password to access Moderator tools."
            )
            st.stop()
    if mode == "Admin":
        pw = st.text_input("Admin password", type="password")
        if pw != ADMIN_PASSWORD:
            st.warning("Enter the correct admin password to access Admin tools.")
            st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# General (View-only)
# ─────────────────────────────────────────────────────────────────────────────

if mode == "General":
    st.title("🏆 Game Night Leaderboard")
    tabs_general = st.tabs(["Leaderboard", "Daily Game", "Profiles", "Locker"])
    with tabs_general[0]:
        joined = sb_select_sessions_joined()
        games_df = sb_select("games")

        if joined.empty:
            st.info("No session data yet. Add sessions from the Admin tab.")
        else:
            cfg = load_config()
            # (Everything from you

            # ---- Filters (General) ----
            c1, c2 = st.columns([2, 1])
            game_names = sorted(joined["game_name"].dropna().unique().tolist())

            # Use a stable key + initialize session state BEFORE creating the widget
            ms_key = "game_filter"
            if ms_key not in st.session_state:
                st.session_state[ms_key] = game_names[
                    :
                ]  # start with all games selected

            # Multiselect bound to session_state; no `default` needed when `key` is used
            selected_games = c1.multiselect(
                "Filter by game",
                options=game_names,
                key=ms_key,
            )

            # Quick actions — update state via callbacks
            bcol1, bcol2 = c1.columns(2)
            bcol1.button(
                "All games",
                key="btn_all_games",
                use_container_width=True,
                on_click=lambda: st.session_state.update({ms_key: game_names[:]}),
            )
            bcol2.button(
                "Clear",
                key="btn_clear_games",
                use_container_width=True,
                on_click=lambda: st.session_state.update({ms_key: []}),
            )

            # Date range fallback: use today if no valid dates
            min_date = joined["played_at"].dt.date.min()
            max_date = joined["played_at"].dt.date.max()
            if not min_date or not max_date or pd.isna(min_date) or pd.isna(max_date):
                min_date = date.today()
                max_date = date.today()
            start_date, end_date = c2.date_input(
                "Date range", value=(min_date, max_date)
            )

            # If exactly one game is selected, show a polished info card (+ BGG link if provided)
            if len(selected_games) == 1:
                gname = selected_games[0]
                grow = (
                    games_df.loc[games_df["name"] == gname].to_dict("records")[0]
                    if not games_df.empty and (games_df["name"] == gname).any()
                    else {}
                )
                bgg_slug = (grow.get("bgg_slug") or "").strip()
                minp = grow.get("min_players", None)
                maxp = grow.get("max_players", None)
                notes = grow.get("notes", "")

                # Stats for current filter range
                start_dt, end_dt = _range_to_df_tz(
                    joined["played_at"], start_date, end_date
                )
                filt = joined[
                    (joined["game_name"] == gname)
                    & (joined["played_at"].between(start_dt, end_dt))
                ]
                sessions_count = filt["session_id"].nunique()
                unique_players = filt["player_id"].nunique()
                avg_table = (
                    (filt.groupby("session_id")["player_id"].nunique().mean())
                    if not filt.empty
                    else np.nan
                )

                colA, colB = st.columns([3, 2])
                with colA:
                    st.markdown(f"### 🎲 {gname}")
                    if minp or maxp:
                        st.caption(
                            f"Players: {int(minp) if pd.notna(minp) else '?'}–{int(maxp) if pd.notna(maxp) else '?'}"
                        )
                    if notes:
                        st.write(notes)
                    if bgg_slug:
                        st.markdown(
                            f"[View on BoardGameGeek](https://boardgamegeek.com/boardgame/{bgg_slug})"
                        )
                with colB:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Sessions", sessions_count)
                    m2.metric("Unique players", unique_players)
                    m3.metric(
                        "Avg table size",
                        f"{avg_table:.1f}" if pd.notna(avg_table) else "—",
                    )
                st.divider()

            lb, h2h, recent, ratings_over_time = compute_leaderboard(
                joined,
                selected_games,
                (start_date, end_date),
                base_rating=float(cfg["starting_elo"]),
                k_ffa_base=float(cfg["k_ffa_base"]),
                k_team=float(cfg["k_team"]),
                coop_drift=float(cfg["coop_drift"]),
                solo_drift=float(cfg["solo_drift"]),
                use_points_weight=bool(cfg["use_points_weight"]),
                points_alpha=float(cfg["points_alpha"]),
                gap_bonus=bool(cfg["gap_bonus"]),
                gap_bonus_coeff=float(cfg["gap_bonus_coeff"]),
                exceptional_bonus=bool(cfg["exceptional_bonus"]),
                exceptional_threshold=float(cfg["exceptional_threshold"]),
                exceptional_points=float(cfg["exceptional_points"]),
            )

            # Cosmetics map for players
            players_df_all = sb_select("players")
            eq_map: Dict[str, Dict[str, Optional[str]]] = {}
            if not players_df_all.empty and "bag" in players_df_all.columns:
                for _, r in players_df_all.iterrows():
                    pid = r["id"]
                    bag = _as_dict(r.get("bag"))
                    eq_map[pid] = (
                        bag.get("cosmetics", {}).get("equipped", {})
                        if isinstance(bag, dict)
                        else {}
                    )

            def _styled_name(pid: str, base_name: str) -> str:
                eq = eq_map.get(pid, {})
                badge_icon = ITEM_DB.get(eq.get("badge", ""), {}).get("icon", "")
                title_txt = _title_text(eq.get("title"))
                left = (badge_icon + " ") if badge_icon else ""
                right = f" ({title_txt})" if title_txt else ""
                return f"{left}{base_name}{right}"

            # Styled names for current leaderboard players
            styled_by_pid = {
                row["player_id"]: _styled_name(row["player_id"], row["display_name"])
                for _, row in lb.iterrows()
            }

            # 1) Leaderboard table column: use styled text
            lb_disp = lb.copy()
            lb_disp.insert(0, "#", np.arange(1, len(lb_disp) + 1))
            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
            lb_disp.insert(1, "🏅", lb_disp["#"].map(medals).fillna(""))
            lb_disp["🔥"] = np.where(lb_disp["current_streak"] >= 3, "🔥", "")
            lb_disp["Win%"] = lb_disp["win_rate"] * 100.0
            lb_disp["Player"] = (
                lb_disp["player_id"].map(styled_by_pid).fillna(lb_disp["display_name"])
            )

            # 1a) (Optional) Compact roster preview with real fonts (top 12)
            st.caption("Styled roster (shows fonts, badges, titles):")
            roster_cols = st.columns(3)
            for i, (pid, disp) in enumerate(styled_by_pid.items()):
                scope = f"pfont-{pid}"
                _inject_font_css(eq_map.get(pid, {}).get("font"), scope)
                roster_cols[i % 3].markdown(
                    f"<div class='{scope}'>{disp}</div>", unsafe_allow_html=True
                )

            # 2) H2H labels: replace index/columns with styled names
            labels = [
                styled_by_pid.get(pid, nm)
                for pid, nm in zip(lb["player_id"], lb["display_name"])
            ]
            if not h2h.empty:
                h2h.index = labels
                h2h.columns = labels

            # 3) Recent winners: use styled names
            if not recent.empty:
                # recent already has resolved_winner earlier; rebuild winners with styled names
                rec = joined.copy()
                rec = rec[rec["session_id"].isin(recent["session_id"].unique())]
                rec["resolved_winner"] = False
                for sid, g_ in rec.groupby("session_id", sort=False):
                    mask = resolve_winners(g_)
                    rec.loc[g_.index, "resolved_winner"] = mask.values
                winners_styled = (
                    rec[rec["resolved_winner"]]
                    .groupby("session_id")[["player_id", "player_name"]]
                    .apply(
                        lambda s: ", ".join(
                            sorted(
                                set(
                                    _styled_name(pid, nm)
                                    for pid, nm in zip(s["player_id"], s["player_name"])
                                )
                            )
                        )
                    )
                    .rename("winners")
                )
                recent = recent.drop(columns=["winners"], errors="ignore").merge(
                    winners_styled, left_on="session_id", right_index=True, how="left"
                )

            # 4) Ratings-over-time: show styled names in legend and selector
            if not ratings_over_time.empty:
                ratings_over_time["player_name"] = (
                    ratings_over_time["player_id"]
                    .map(styled_by_pid)
                    .fillna(ratings_over_time["player_name"])
                )

            if lb.empty:
                st.info("No data for current filters.")
            else:
                # KPI
                k1, k2, k3, k4 = st.columns(4)
                start_dt, end_dt = _range_to_df_tz(
                    joined["played_at"], start_date, end_date
                )
                total_sessions = joined[
                    (joined["game_name"].isin(selected_games))
                    & (joined["played_at"].between(start_dt, end_dt))
                ]["session_id"].nunique()
                total_players = lb.shape[0]
                total_games = len(selected_games)
                last_play = joined["played_at"].max()
                k1.metric("Sessions", total_sessions)
                k2.metric("Players", total_players)
                k3.metric("Games", total_games)
                k4.metric(
                    "Last played",
                    last_play.strftime("%Y-%m-%d") if pd.notna(last_play) else "—",
                )

                # ── Highlights strip ─────────────────────────────────────────────
                champ = lb.loc[lb["elo"].idxmax()] if not lb.empty else None
                longest = lb.loc[lb["best_streak"].idxmax()] if not lb.empty else None
                eligible = lb[lb["games_played"] >= 3]
                best_wr = (
                    eligible.loc[eligible["win_rate"].idxmax()]
                    if not eligible.empty
                    else None
                )
                most_active = (
                    lb.loc[lb["games_played"].idxmax()] if not lb.empty else None
                )

                cA, cB, cC, cD = st.columns(4)

                def metric_card(title, value, delta, color="green"):
                    display_value = value if value else "—"
                    display_delta = delta if delta else ""
                    st.markdown(
                        f"""
                        <div style="text-align:center; padding:0.5em; border-radius:8px; background-color:rgba(240,240,240,0.5);">
                            <div style="font-size:0.9em; font-weight:bold;">{title}</div>
                            <div style="font-size:1.1em; white-space:normal; word-break:break-word;">{display_value}</div>
                            <div style="color:{color}; font-size:0.85em;">{display_delta}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                def _pretty(row):
                    if row is None:
                        return None
                    return styled_by_pid.get(row["player_id"], row["display_name"])

                with cA:
                    metric_card(
                        "🏆 Champion",
                        _pretty(champ),
                        f"ELO {champ['elo']:.0f}" if champ is not None else None,
                    )
                with cB:
                    metric_card(
                        "🔥 Longest Streak",
                        _pretty(longest),
                        (
                            f"{int(longest['best_streak'])} wins"
                            if longest is not None
                            else None
                        ),
                    )
                with cC:
                    metric_card(
                        "🎯 Best Win% (≥3 GP)",
                        _pretty(best_wr),
                        (
                            f"{best_wr['win_rate']*100:.0f}%"
                            if best_wr is not None
                            else None
                        ),
                    )
                with cD:
                    metric_card(
                        "👥 Most Active",
                        _pretty(most_active),
                        (
                            f"{int(most_active['games_played'])} GP"
                            if most_active is not None
                            else None
                        ),
                    )

                # ── Sexy Leaderboard: rank, medals, win% progress, streak flair ──
                lb_disp = lb.copy()  # lb is already ELO-sorted from compute_leaderboard
                lb_disp.insert(0, "#", np.arange(1, len(lb_disp) + 1))
                medals = {1: "🥇", 2: "🥈", 3: "🥉"}
                lb_disp.insert(1, "🏅", lb_disp["#"].map(medals).fillna(""))
                lb_disp["🔥"] = np.where(lb_disp["current_streak"] >= 3, "🔥", "")
                lb_disp["Win%"] = lb_disp["win_rate"] * 100.0

                lb_disp["Player"] = (
                    lb_disp["player_id"]
                    .map(styled_by_pid)
                    .fillna(lb_disp["display_name"])
                )

                show_cols = [
                    "#",
                    "🏅",
                    "display_name",
                    "elo",
                    "wins",
                    "games_played",
                    "Win%",
                    "avg_position",
                    "avg_points",
                    "current_streak",
                    "best_streak",
                    "last_played",
                    "🔥",
                ]

                st.subheader("Leaderboard")

                show_cols = [
                    "#",
                    "🏅",
                    "Player",
                    "elo",
                    "wins",
                    "games_played",
                    "Win%",
                    "avg_position",
                    "avg_points",
                    "current_streak",
                    "best_streak",
                    "last_played",
                    "🔥",
                ]

                st.dataframe(
                    lb_disp[show_cols].rename(
                        columns={
                            "elo": "ELO",
                            "wins": "W",
                            "games_played": "GP",
                            "avg_position": "Avg Pos",
                            "avg_points": "Avg Pts",
                            "current_streak": "Streak",
                            "best_streak": "Best Streak",
                            "last_played": "Last",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "#": st.column_config.NumberColumn(width="small"),
                        "🏅": st.column_config.TextColumn(width="small", help="Top 3"),
                        "Player": st.column_config.TextColumn(width="large"),
                        "ELO": st.column_config.NumberColumn(
                            format="%.0f", help="Higher is better"
                        ),
                        "W": st.column_config.NumberColumn(width="small"),
                        "GP": st.column_config.NumberColumn(width="small"),
                        "Win%": st.column_config.ProgressColumn(
                            "Win%",
                            help="Wins / Games Played",
                            format="%.0f%%",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "Avg Pos": st.column_config.NumberColumn(format="%.2f"),
                        "Avg Pts": st.column_config.NumberColumn(format="%.2f"),
                        "Streak": st.column_config.NumberColumn(
                            width="small", help="Current winning streak"
                        ),
                        "Best Streak": st.column_config.NumberColumn(width="small"),
                        "Last": st.column_config.DatetimeColumn(
                            format="YYYY-MM-DD HH:mm"
                        ),
                        "🔥": st.column_config.TextColumn(
                            width="small", help="Hot streak (3+ wins)"
                        ),
                    },
                )

                # ── Rating history chart ────────────────────────────────────────
                st.subheader("📈 Player Rating Over Time")

                # Choose players (defaults to top 3 by current ELO)
                player_opts = lb_disp["Player"].tolist()
                default_sel = player_opts[:3]
                selected_players_rt = st.multiselect(
                    "Select players",
                    options=player_opts,
                    default=default_sel,
                )

                if selected_players_rt and not ratings_over_time.empty:
                    plot_df = ratings_over_time[
                        ratings_over_time["player_name"].isin(selected_players_rt)
                    ]
                    if not plot_df.empty:
                        # make sure it's sorted for nice lines
                        plot_df = plot_df.sort_values(["player_name", "played_at"])

                        fig = go.Figure()
                        for pname, grp in plot_df.groupby("player_name", sort=False):
                            fig.add_trace(
                                go.Scatter(
                                    x=grp["played_at"],
                                    y=grp["elo"],
                                    mode="lines+markers",
                                    name=pname,
                                    hovertemplate="<b>%{text}</b><br>%{x|%Y-%m-%d %H:%M}<br>ELO: %{y:.0f}<extra></extra>",
                                    text=[pname] * len(grp),
                                )
                            )
                        fig.update_layout(
                            title="ELO progression (respects current filters)",
                            xaxis_title="Date",
                            yaxis_title="ELO",
                            hovermode="x unified",
                            margin=dict(l=0, r=0, t=60, b=0),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No data for selected players in current filters.")
                else:
                    st.caption("Pick one or more players to see their rating history.")

                st.subheader("Head-to-Head (times row finished ahead of column)")
                render_h2h_heatmap(
                    h2h, title="Head-to-Head (row finished ahead of column)"
                )

                st.subheader("Recent Sessions")
                rename_map = {
                    "played_at": "Date",
                    "game_name": "Game",
                    "location": "Location",
                    "notes": "Notes",
                    "winners": "Winners",
                }
                st.dataframe(
                    recent.rename(columns=rename_map),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Date": st.column_config.DatetimeColumn(
                            format="YYYY-MM-DD HH:mm"
                        )
                    },
                )

                st.download_button(
                    "Download leaderboard (CSV)",
                    data=lb.to_csv(index=False).encode("utf-8"),
                    file_name="leaderboard.csv",
                )
        # ===== TAB 2: Daily Game =====
    with tabs_general[1]:
        render_daily_game()

        # ===== TAB 3: Profiles (read-only inspection) =====
    with tabs_general[2]:
        st.title("👤 Player Profiles")
        players_df = sb_select("players")
        if players_df.empty:
            st.info("No players yet.")
        else:
            pmap = dict(zip(players_df["name"], players_df["id"]))
            selected_name = st.selectbox("Select a player", options=sorted(pmap.keys()))
            pid = pmap[selected_name]
            eq = _equipped_for_player(pid)
            scope = f"pfont-{pid}"
            _inject_font_css(eq.get("font"), scope)

            # Stats from the same ELO engine used in General
            joined_all = sb_select_sessions_joined()
            cfg = load_config()
            lb_all, _, _, _ = compute_leaderboard(
                joined_all,
                None,
                None,
                base_rating=float(cfg["starting_elo"]),
                k_ffa_base=float(cfg["k_ffa_base"]),
                k_team=float(cfg["k_team"]),
                coop_drift=float(cfg["coop_drift"]),
                solo_drift=float(cfg["solo_drift"]),
                use_points_weight=bool(cfg["use_points_weight"]),
                points_alpha=float(cfg["points_alpha"]),
                gap_bonus=bool(cfg["gap_bonus"]),
                gap_bonus_coeff=float(cfg["gap_bonus_coeff"]),
                exceptional_bonus=bool(cfg["exceptional_bonus"]),
                exceptional_threshold=float(cfg["exceptional_threshold"]),
                exceptional_points=float(cfg["exceptional_points"]),
            )
            row = (
                lb_all[lb_all["player_id"] == pid].iloc[0]
                if not lb_all.empty and (lb_all["player_id"] == pid).any()
                else None
            )

            # Best game (by Win% with ≥3 games; fallback to most wins)
            best_game_txt = "—"
            if not joined_all.empty:
                g = joined_all.copy()
                g["resolved_winner"] = False
                for sid, gg in g.groupby("session_id", sort=False):
                    mask = resolve_winners(gg)
                    g.loc[gg.index, "resolved_winner"] = mask.values
                me = g[g["player_id"] == pid]
                if not me.empty:
                    per_game = me.groupby("game_name").agg(
                        GP=("session_id", pd.Series.nunique),
                        W=("resolved_winner", "sum"),
                    )
                    if not per_game.empty:
                        per_game["WR"] = (per_game["W"] / per_game["GP"]).fillna(0.0)
                        candidates = per_game[per_game["GP"] >= 3]
                        pick = candidates.sort_values(
                            ["WR", "GP", "W"], ascending=[False, False, False]
                        ).head(1)
                        if pick.empty:
                            pick = per_game.sort_values(
                                ["W", "GP", "WR"], ascending=[False, False, False]
                            ).head(1)
                        if not pick.empty:
                            gname = pick.index[0]
                            WR = pick["WR"].iloc[0] * 100
                            GP = int(pick["GP"].iloc[0])
                            best_game_txt = f"{gname} ({WR:.0f}% WR, {GP} GP)"

            # Nameplate
            title_txt = _title_text(eq.get("title"))
            badge_icon = ITEM_DB.get(eq.get("badge", ""), {}).get("icon", "")
            st.markdown(
                f"""
                <div class="player-nameplate {scope}" style="
                    padding:12px 16px;border-radius:10px;background:rgba(240,240,240,0.6);
                    border:2px solid #ddd;display:inline-block;">
                    <div style="font-size:20px;font-weight:700;">{selected_name}</div>
                    <div style="font-size:13px;opacity:0.8;">{badge_icon} {title_txt or 'No title'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("ELO", f"{row['elo']:.0f}" if row is not None else "—")
            k2.metric("Win%", f"{row['win_rate']*100:.0f}%" if row is not None else "—")
            k3.metric(
                "Games Played", int(row["games_played"]) if row is not None else 0
            )
            k4.metric("Best Streak", int(row["best_streak"]) if row is not None else 0)

            st.caption(f"⭐ Best game: {best_game_txt}")

            # Cosmetics & inventory (compact)
            bag = _get_player_bag(pid)
            owned = bag.get("cosmetics", {}).get("owned", [])
            inv = bag.get("inventory", {})
            st.markdown("**Equipped**")
            st.write({k: v for k, v in _equipped_for_player(pid).items() if v} or "—")
            st.markdown("**Owned cosmetics**")
            st.write(owned or "—")
            st.markdown("**Inventory** (boxes/items)")
            if inv:
                inv_df = pd.DataFrame([{"Item": k, "Qty": v} for k, v in inv.items()])
                st.dataframe(inv_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Empty")
            # ===== TAB 4: Locker (open, craft, equip) =====
        with tabs_general[3]:
            render_locker()

# ─────────────────────────────────────────────────────────────────────────────
# Moderator (submit session requests only)
# ─────────────────────────────────────────────────────────────────────────────
if mode == "Moderator":
    st.title("🧑‍⚖️ Moderator — Submit Session for Approval")

    players_df = sb_select("players")
    games_df = sb_select("games")

    if games_df.empty or players_df.empty:
        st.info("Admin must add at least one game and one player first.")
        st.stop()

    # ===== This is the same form as Admin → Add Session, but saves to session_requests
    game_records = games_df.to_dict("records")
    game_row = st.selectbox(
        "Game", game_records, format_func=lambda r: r.get("name", "—")
    )
    game_min = int((game_row.get("min_players") or 1))
    game_max = int((game_row.get("max_players") or max(2, game_min)))

    c0, c1 = st.columns([1, 2])
    played_at = c0.date_input("Date", value=date.today())
    played_time = c0.time_input(
        "Time", value=datetime.now().replace(second=0, microsecond=0).time()
    )
    played_at_dt = datetime.combine(played_at, played_time)

    location = c1.text_input("Location", value="")
    notes = st.text_area("Notes (optional)", value="")

    session_mode = st.radio(
        "Session type", ["Free-for-all (FFA)", "Team", "Co-op", "Solo"], horizontal=True
    )

    if session_mode == "Solo":
        n_min, n_max, default_n = 1, 1, 1
    else:
        n_min, n_max, default_n = (
            max(2, game_min),
            max(game_min, game_max),
            max(game_min, min(4, game_max)),
        )

    n_players = st.number_input(
        "Number of players", min_value=n_min, max_value=n_max, value=default_n, step=1
    )

    entries: List[Dict[str, Any]] = []
    player_options = players_df["name"].tolist()
    name_to_id = dict(zip(players_df["name"], players_df["id"]))

    if session_mode == "Team":
        st.markdown(
            "**Participants (assign to team labels A/B/C/… and pick the winning team(s))**"
        )
        team_labels = ["A", "B", "C", "D", "E"]
        for i in range(int(n_players)):
            c1_, c2_, c3_ = st.columns([3, 1, 1])
            pname = c1_.selectbox(
                f"Player {i+1}", options=player_options, key=f"mod_team_p_{i}"
            )
            team = c2_.selectbox(
                "Team", options=team_labels, index=0, key=f"mod_team_label_{i}"
            )
            pts = c3_.number_input(
                "Pts",
                min_value=0.0,
                max_value=1e9,
                value=0.0,
                step=1.0,
                key=f"mod_team_pts_{i}",
            )
            entries.append(
                {
                    "pname": pname,
                    "player_id": name_to_id.get(pname),
                    "team": team,
                    "position": None,
                    "points": float(pts),
                    "is_winner": False,
                }
            )
        winning_teams = st.multiselect(
            "Winning team(s)", options=team_labels, default=["A"], help="Supports ties"
        )

    elif session_mode == "Co-op":
        st.markdown("**Participants (co-op — outcome applies to all)**")
        for i in range(int(n_players)):
            c1_, c2_ = st.columns([3, 1])
            pname = c1_.selectbox(
                f"Player {i+1}", options=player_options, key=f"mod_coop_p_{i}"
            )
            pts = c2_.number_input(
                "Pts",
                min_value=0.0,
                max_value=1e9,
                value=0.0,
                step=1.0,
                key=f"mod_coop_pts_{i}",
            )
            entries.append(
                {
                    "pname": pname,
                    "player_id": name_to_id.get(pname),
                    "team": "COOP",
                    "position": None,
                    "points": float(pts),
                    "is_winner": False,
                }
            )
        coop_outcome = st.radio(
            "Outcome", ["Win", "Loss"], horizontal=True, key="mod_coop_outcome"
        )

    elif session_mode == "Solo":
        st.markdown("**Solo session (single player vs system)**")
        c1_, c2_, c3_ = st.columns([3, 1, 1])
        pname = c1_.selectbox("Player", options=player_options, key="mod_solo_p")
        outcome = c2_.radio(
            "Result", ["Win", "Loss"], horizontal=True, key="mod_solo_outcome"
        )
        pts = c3_.number_input(
            "Pts (optional)",
            min_value=0.0,
            max_value=1e9,
            value=0.0,
            step=1.0,
            key="mod_solo_pts",
        )
        entries.append(
            {
                "pname": pname,
                "player_id": name_to_id.get(pname),
                "team": "SOLO",
                "position": None,
                "points": float(pts),
                "is_winner": (outcome == "Win"),
            }
        )

    else:  # FFA
        st.markdown("**Participants (FFA)**")
        single_winner_ui = st.checkbox(
            "Enforce single winner (radio)",
            value=False,
            help="If on, pick exactly one winner via radio.",
        )
        for i in range(int(n_players)):
            c1_, c2_, c3_, c4_ = st.columns([3, 1, 1, 1])
            pname = c1_.selectbox(
                f"Player {i+1}", options=player_options, key=f"mod_ffa_p_{i}"
            )
            pos = c2_.number_input(
                "Pos",
                min_value=1,
                max_value=99,
                value=i + 1,
                step=1,
                key=f"mod_ffa_pos_{i}",
            )
            pts = c3_.number_input(
                "Pts",
                min_value=0.0,
                max_value=1e9,
                value=0.0,
                step=1.0,
                key=f"mod_ffa_pts_{i}",
            )
            if single_winner_ui:
                win = False
            else:
                win = c4_.checkbox("Winner", value=False, key=f"mod_ffa_win_{i}")
            entries.append(
                {
                    "pname": pname,
                    "player_id": name_to_id.get(pname),
                    "team": "",
                    "position": int(pos),
                    "points": float(pts),
                    "is_winner": bool(win),
                }
            )

        if single_winner_ui and entries:
            winner_names = [e["pname"] for e in entries]
            chosen = st.radio(
                "Winner",
                options=winner_names,
                horizontal=False,
                key="mod_ffa_single_winner",
            )
            for e in entries:
                e["is_winner"] = e["pname"] == chosen

    # Normalize winners for Team/Co-op into entries
    if session_mode == "Team":
        for e in entries:
            e["is_winner"] = e["team"] in winning_teams
    elif session_mode == "Co-op":
        win_all = coop_outcome == "Win"
        for e in entries:
            e["is_winner"] = win_all

    submitted = st.button("Submit request for admin approval", type="primary")

    if submitted:
        if any(e.get("player_id") is None for e in entries):
            st.error("One or more players couldn't be resolved to an ID.")
            st.stop()

        payload = {
            "game_id": game_row["id"],
            "game_name": game_row.get("name"),
            "played_at": played_at_dt.isoformat(),
            "location": (location or "").strip() or None,
            "notes": (notes or "").strip() or None,
            "mode": session_mode,
            "entries": entries,  # each has player_id + display fields
        }
        ok = sb_insert(
            "session_requests",
            {
                "id": _uuid(),
                "payload": payload,
                "status": "pending",
                "created_by": (moderator_name or None),
            },
        )
        if ok:
            st.success("Request submitted. Admin will review it in Approvals.")
            st.toast("Sent to Approvals ✅")
            st.cache_data.clear()

# ─────────────────────────────────────────────────────────────────────────────
# Admin
# ─────────────────────────────────────────────────────────────────────────────

if mode == "Admin":
    st.title("🛠️ Admin")

    tabs = st.tabs(
        [
            "Add Session",
            "Players",
            "Games",
            "Manage Sessions",
            "ELO Settings",
            "Approvals",
            "Economy",
        ]
    )
    players_df = sb_select("players")
    games_df = sb_select("games")

    # ------------------------- Add Session -------------------------
    with tabs[0]:
        st.subheader("Add a Session")

        if games_df.empty or players_df.empty:
            st.info("Add at least one game and one player first.")
        else:
            # Choose game and clamp lobby size to game's min/max
            game_records = games_df.to_dict("records")
            game_row = st.selectbox(
                "Game", game_records, format_func=lambda r: r.get("name", "—")
            )
            game_min = int((game_row.get("min_players") or 1))
            game_max = int((game_row.get("max_players") or max(2, game_min)))

            c0, c1 = st.columns([1, 2])
            played_at = c0.date_input("Date", value=date.today())
            played_time = c0.time_input(
                "Time", value=datetime.now().replace(second=0, microsecond=0).time()
            )
            # Combine to a proper datetime (naive local time is fine here)
            played_at_dt = datetime.combine(played_at, played_time)

            location = c1.text_input("Location", value="")
            notes = st.text_area("Notes (optional)", value="")

            session_mode = st.radio(
                "Session type",
                ["Free-for-all (FFA)", "Team", "Co-op", "Solo"],
                horizontal=True,
            )

            # Number of players UI varies by mode (and constrained by game min/max)
            if session_mode == "Solo":
                n_min, n_max, default_n = 1, 1, 1
            else:
                n_min, n_max, default_n = (
                    max(2, game_min),
                    max(game_min, game_max),
                    max(game_min, min(4, game_max)),
                )

            n_players = st.number_input(
                "Number of players",
                min_value=n_min,
                max_value=n_max,
                value=default_n,
                step=1,
            )

            entries: List[Dict[str, Any]] = []
            player_options = players_df["name"].tolist()
            name_to_id = dict(zip(players_df["name"], players_df["id"]))

            if session_mode == "Team":
                st.markdown(
                    "**Participants (assign to team labels A/B/C/… and pick the winning team(s))**"
                )
                team_labels = ["A", "B", "C", "D", "E"]
                for i in range(int(n_players)):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    pname = c1.selectbox(
                        f"Player {i+1}", options=player_options, key=f"team_p_{i}"
                    )
                    team = c2.selectbox(
                        "Team", options=team_labels, index=0, key=f"team_label_{i}"
                    )
                    pts = c3.number_input(
                        "Pts",
                        min_value=0.0,
                        max_value=1e9,
                        value=0.0,
                        step=1.0,
                        key=f"team_pts_{i}",
                    )
                    entries.append(
                        {
                            "pname": pname,
                            "team": team,
                            "position": None,
                            "points": float(pts),
                            "is_winner": False,
                        }
                    )
                winning_teams = st.multiselect(
                    "Winning team(s)",
                    options=team_labels,
                    default=["A"],
                    help="Supports ties",
                )

            elif session_mode == "Co-op":
                st.markdown("**Participants (co-op — outcome applies to all)**")
                for i in range(int(n_players)):
                    c1, c2 = st.columns([3, 1])
                    pname = c1.selectbox(
                        f"Player {i+1}", options=player_options, key=f"coop_p_{i}"
                    )
                    pts = c2.number_input(
                        "Pts",
                        min_value=0.0,
                        max_value=1e9,
                        value=0.0,
                        step=1.0,
                        key=f"coop_pts_{i}",
                    )
                    entries.append(
                        {
                            "pname": pname,
                            "team": "COOP",
                            "position": None,
                            "points": float(pts),
                            "is_winner": False,
                        }
                    )
                coop_outcome = st.radio(
                    "Outcome", ["Win", "Loss"], horizontal=True, key="coop_outcome"
                )

            elif session_mode == "Solo":
                st.markdown("**Solo session (single player vs system)**")
                c1, c2, c3 = st.columns([3, 1, 1])
                pname = c1.selectbox("Player", options=player_options, key="solo_p")
                outcome = c2.radio(
                    "Result", ["Win", "Loss"], horizontal=True, key="solo_outcome"
                )
                pts = c3.number_input(
                    "Pts (optional)",
                    min_value=0.0,
                    max_value=1e9,
                    value=0.0,
                    step=1.0,
                    key="solo_pts",
                )
                entries.append(
                    {
                        "pname": pname,
                        "team": "SOLO",
                        "position": None,
                        "points": float(pts),
                        "is_winner": (outcome == "Win"),
                    }
                )

            else:  # FFA
                st.markdown("**Participants (FFA)**")
                single_winner_ui = st.checkbox(
                    "Enforce single winner (radio)",
                    value=False,
                    help="If on, pick exactly one winner via radio.",
                )
                for i in range(int(n_players)):
                    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                    pname = c1.selectbox(
                        f"Player {i+1}", options=player_options, key=f"ffa_p_{i}"
                    )
                    pos = c2.number_input(
                        "Pos",
                        min_value=1,
                        max_value=99,
                        value=i + 1,
                        step=1,
                        key=f"ffa_pos_{i}",
                    )
                    pts = c3.number_input(
                        "Pts",
                        min_value=0.0,
                        max_value=1e9,
                        value=0.0,
                        step=1.0,
                        key=f"ffa_pts_{i}",
                    )
                    if single_winner_ui:
                        win = False
                        entries.append(
                            {
                                "pname": pname,
                                "team": "",
                                "position": int(pos),
                                "points": float(pts),
                                "is_winner": win,
                            }
                        )
                    else:
                        win = c4.checkbox("Winner", value=False, key=f"ffa_win_{i}")
                        entries.append(
                            {
                                "pname": pname,
                                "team": "",
                                "position": int(pos),
                                "points": float(pts),
                                "is_winner": bool(win),
                            }
                        )

                if single_winner_ui and entries:
                    winner_names = [e["pname"] for e in entries]
                    chosen = st.radio(
                        "Winner",
                        options=winner_names,
                        horizontal=False,
                        key="ffa_single_winner",
                    )
                    for e in entries:
                        e["is_winner"] = e["pname"] == chosen

            submitted = st.button("Save session", type="primary")

            if submitted:
                # Compose and insert session
                sid = _uuid()
                ok = sb_insert(
                    "sessions",
                    {
                        "id": sid,
                        "game_id": game_row["id"],
                        "played_at": played_at_dt.isoformat(),
                        "location": location.strip() or None,
                        "notes": notes.strip() or None,
                    },
                )
                if not ok:
                    st.stop()

                # Mode-specific post-processing
                if session_mode == "Team":
                    for e in entries:
                        e["is_winner"] = e["team"] in winning_teams
                elif session_mode == "Co-op":
                    win_all = coop_outcome == "Win"
                    for e in entries:
                        e["is_winner"] = win_all
                elif session_mode == "Solo":
                    pass
                else:
                    # FFA sanity fallback
                    if not any(e.get("is_winner") for e in entries):
                        best = None
                        best_key = None
                        for e in entries:
                            key = (
                                e["position"] if e["position"] is not None else 1e9,
                                -(e["points"] if e["points"] is not None else -1e9),
                            )
                            if best_key is None or key < best_key:
                                best, best_key = e, key
                        if best:
                            best["is_winner"] = True

                # Insert participants
                all_ok = True
                for e in entries:
                    pid = name_to_id.get(e["pname"])
                    row = {
                        "id": _uuid(),
                        "session_id": sid,
                        "player_id": pid,
                        "team": (e.get("team") or None),
                        "position": e.get("position"),
                        "points": e.get("points"),
                        "is_winner": bool(e.get("is_winner", False)),
                    }
                    if not sb_insert("session_players", row):
                        all_ok = False

                if all_ok:
                    st.success("Session saved.")
                    st.cache_data.clear()
                else:
                    st.error("Some participants failed to save. Check errors above.")

    # ------------------------- Players -------------------------
    with tabs[1]:
        st.subheader("Players")
        if not players_df.empty:
            cols = [
                c for c in ["name", "nickname", "joined_on"] if c in players_df.columns
            ]
            view = players_df[cols].rename(
                columns={"name": "Player", "nickname": "Nick"}
            )
            st.dataframe(view, use_container_width=True, hide_index=True)
            # Light controls for PIN/emoji + cosmetics
            if not players_df.empty:
                st.markdown("### Manage players")
                for _, prow in players_df.iterrows():
                    with st.expander(f"⚙️ {prow.get('name','(unknown)')}"):
                        pid = prow["id"]
                        c1, c2, c3 = st.columns([1, 1, 2])
                        # PIN / Emoji
                        new_pin = c1.text_input(
                            "PIN (4 digits)",
                            value=str(prow.get("pin_code") or ""),
                            max_chars=4,
                            key=f"adm_pin_{pid}",
                        )
                        new_emo = c2.text_input(
                            "Emoji lock",
                            value=str(prow.get("emoji_lock") or ""),
                            max_chars=2,
                            key=f"adm_emo_{pid}",
                        )
                        if c3.button("Save login", key=f"adm_save_login_{pid}"):
                            if sb_update_by_id(
                                "players",
                                pid,
                                {"pin_code": new_pin, "emoji_lock": new_emo},
                            ):
                                st.success("Saved")
                                st.cache_data.clear()

                        # Quick grant (boxes / voucher)
                        g1, g2, g3 = st.columns(3)
                        grant_code = g1.selectbox(
                            "Grant item",
                            [
                                "BOX_BRONZE",
                                "BOX_SILVER",
                                "BOX_GOLD",
                                "V_PICK_NEXT_GAME",
                            ],
                            key=f"adm_grant_code_{pid}",
                        )
                        grant_qty = g2.number_input(
                            "Qty",
                            min_value=1,
                            max_value=10,
                            value=1,
                            step=1,
                            key=f"adm_grant_qty_{pid}",
                        )
                        if g3.button("Grant", key=f"adm_grant_btn_{pid}"):
                            if _bag_add(pid, grant_code, int(grant_qty)):
                                st.success("Granted")
                            else:
                                st.error("Grant failed")

                        # Equip font cosmetic (if owned) or force-equip (admin)
                        bag = _get_player_bag(pid)
                        owned = bag.get("cosmetics", {}).get("owned", [])
                        eq = bag.get("cosmetics", {}).get("equipped", {})
                        font_choices = list(FONT_CATALOG.keys())
                        sel_font = st.selectbox(
                            "Equip font",
                            font_choices,
                            index=(
                                font_choices.index(eq.get("font"))
                                if eq.get("font") in font_choices
                                else 0
                            ),
                            key=f"adm_font_sel_{pid}",
                        )
                        cols_fx = st.columns([1, 1])
                        if cols_fx[0].button(
                            "Equip (require owned)", key=f"adm_font_equip_{pid}"
                        ):
                            if sel_font in owned:
                                _equip_cosmetic(pid, "font", sel_font)
                                st.success("Equipped")
                            else:
                                st.warning("Player does not own this font cosmetic.")
                        if cols_fx[1].button(
                            "Force equip (admin)", key=f"adm_font_force_{pid}"
                        ):
                            # ensure it is listed as owned then equip
                            bag = _get_player_bag(pid)
                            bag["cosmetics"].setdefault("owned", [])
                            if sel_font not in bag["cosmetics"]["owned"]:
                                bag["cosmetics"]["owned"].append(sel_font)
                                _save_player_bag(pid, bag)
                            _equip_cosmetic(pid, "font", sel_font)
                            st.success("Force equipped")

                        st.caption(
                            "Tip: fonts are also part of box rewards/crafting; here you can override as admin."
                        )
        with st.form("add_player_form", clear_on_submit=True):
            st.markdown("**Add Player**")
            name = st.text_input("Name", "")
            nickname = st.text_input("Nickname (opt)", "")
            add_p = st.form_submit_button("Add player", type="primary")

        if add_p and name.strip():
            if sb_insert(
                "players",
                {
                    "id": _uuid(),
                    "name": name.strip(),
                    "nickname": nickname.strip() or None,
                },
            ):
                st.success(f"Added player {name}")
                st.cache_data.clear()

    # ------------------------- Games -------------------------
    with tabs[2]:
        st.subheader("Games (any type)")
        if not games_df.empty:
            st.dataframe(
                games_df[
                    ["name", "min_players", "max_players", "bgg_slug", "notes"]
                ].rename(
                    columns={
                        "name": "Game",
                        "min_players": "Min",
                        "max_players": "Max",
                        "bgg_slug": "BGG",
                        "notes": "Notes",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        with st.form("add_game_form", clear_on_submit=True):
            st.markdown("**Add Game**")
            gname = st.text_input("Name", "")
            minp_col, maxp_col = st.columns(2)
            min_players = minp_col.number_input(
                "Min players", min_value=1, max_value=50, value=2, step=1
            )
            max_players = maxp_col.number_input(
                "Max players", min_value=1, max_value=50, value=4, step=1
            )
            bgg = st.text_input("BGG slug (opt)", "")
            gnotes = st.text_area("Notes (opt)", "")
            add_g = st.form_submit_button("Add game", type="primary")
        if add_g and gname.strip():
            if sb_insert(
                "games",
                {
                    "id": _uuid(),
                    "name": gname.strip(),
                    "min_players": int(min_players),
                    "max_players": int(max_players),
                    "bgg_slug": bgg.strip() or None,
                    "notes": gnotes.strip() or None,
                },
            ):
                st.success(f"Added game {gname}")
                st.cache_data.clear()

    # ------------------------- Manage Sessions -------------------------
    with tabs[3]:
        st.subheader("Manage Sessions (delete)")
        joined_admin = sb_select_sessions_joined()
        if joined_admin.empty:
            st.info("No sessions yet.")
        else:
            winners = joined_admin.copy()
            winners["resolved_winner"] = False
            for sid, g in winners.groupby("session_id"):
                winners_mask = resolve_winners(g)
                winners.loc[g.index, "resolved_winner"] = winners_mask.values

            winners_names = (
                winners[winners["resolved_winner"]]
                .groupby("session_id")["player_name"]
                .apply(lambda s: ", ".join(sorted(set(s))))
                .rename("Winners")
            )
            view_cols = ["session_id", "played_at", "game_name", "location"]
            view_cols = [c for c in view_cols if c in winners.columns]
            view = (
                winners.drop_duplicates("session_id")[view_cols]
                .merge(
                    winners_names, left_on="session_id", right_index=True, how="left"
                )
                .sort_values("played_at", ascending=False)
            )

            st.dataframe(
                view.rename(columns={"played_at": "Date", "game_name": "Game"}),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**Delete a session**")
            sid_options = view["session_id"].tolist()
            to_delete = st.selectbox("Select session id", sid_options)
            if st.button(
                "Delete session", type="secondary", disabled=not bool(sid_options)
            ):
                if sb_delete_session(to_delete):
                    st.success("Session and its participants deleted.")
                    st.cache_data.clear()

    # ------------------------- ELO Settings -------------------------
    with tabs[4]:
        st.subheader("ELO Settings (affects General tab)")

        cfg = load_config()
        c1, c2 = st.columns(2)

        with c1:
            starting_elo = st.number_input(
                "Starting ELO",
                min_value=400.0,
                max_value=2000.0,
                value=float(cfg["starting_elo"]),
                step=25.0,
            )
            k_ffa_base = st.number_input(
                "K (FFA base)",
                min_value=4.0,
                max_value=64.0,
                value=float(cfg["k_ffa_base"]),
                step=1.0,
            )
            k_team = st.number_input(
                "K (Team)",
                min_value=4.0,
                max_value=64.0,
                value=float(cfg["k_team"]),
                step=1.0,
            )
            coop_drift = st.number_input(
                "Co-op drift (± per team win/loss)",
                min_value=0.0,
                max_value=20.0,
                value=float(cfg["coop_drift"]),
                step=0.5,
            )
            solo_drift = st.number_input(
                "Solo drift (± per win/loss)",
                min_value=0.0,
                max_value=20.0,
                value=float(cfg["solo_drift"]),
                step=0.5,
            )

        with c2:
            use_points_weight = st.checkbox(
                "Weight ELO by points (MoV)", value=bool(cfg["use_points_weight"])
            )
            points_alpha = st.slider(
                "Points influence on K", 0.0, 1.0, float(cfg["points_alpha"]), 0.05
            )
            gap_bonus = st.checkbox(
                "Bonus for larger position gaps", value=bool(cfg["gap_bonus"])
            )
            gap_bonus_coeff = st.slider(
                "Gap bonus coefficient", 0.0, 0.5, float(cfg["gap_bonus_coeff"]), 0.01
            )
            exceptional_bonus = st.checkbox(
                "Exceptional performance bonus", value=bool(cfg["exceptional_bonus"])
            )
            exceptional_threshold = st.slider(
                "Exceptional margin threshold (normalized)",
                0.0,
                1.0,
                float(cfg["exceptional_threshold"]),
                0.01,
            )
            exceptional_points = st.number_input(
                "Exceptional bonus points",
                min_value=0.0,
                max_value=50.0,
                value=float(cfg["exceptional_points"]),
                step=1.0,
            )

        if st.button("Save settings", type="primary"):
            new_cfg = {
                "starting_elo": starting_elo,
                "k_ffa_base": k_ffa_base,
                "k_team": k_team,
                "coop_drift": coop_drift,
                "solo_drift": solo_drift,
                "use_points_weight": use_points_weight,
                "points_alpha": points_alpha,
                "gap_bonus": gap_bonus,
                "gap_bonus_coeff": gap_bonus_coeff,
                "exceptional_bonus": exceptional_bonus,
                "exceptional_threshold": exceptional_threshold,
                "exceptional_points": exceptional_points,
            }
            if save_config(new_cfg):
                st.success("Settings saved. Refresh the General tab to apply.")
            else:
                st.error("Failed to save settings.")

        if st.button("Reset to defaults", type="secondary"):
            if save_config(DEFAULT_CONFIG):
                st.success(
                    "Settings reset to defaults. Refresh the General tab to apply."
                )
            else:
                st.error("Failed to reset settings.")

    # ------------------------- Approvals -------------------------
    with tabs[5]:
        st.subheader("Pending Session Requests")

        req_df = sb_select("session_requests")
        if req_df.empty or "payload" not in req_df.columns:
            st.info("No pending requests.")
        else:
            # Most recent first
            req_df = req_df.sort_values("created_at", ascending=False)

            players_df_ap = sb_select("players")
            games_df_ap = sb_select("games")

            id_to_player = dict(zip(players_df_ap["id"], players_df_ap["name"]))
            name_to_id = dict(zip(players_df_ap["name"], players_df_ap["id"]))
            id_to_game = dict(zip(games_df_ap["id"], games_df_ap["name"]))
            game_name_to_id = {v: k for k, v in id_to_game.items()}
            all_player_names = sorted(players_df_ap["name"].tolist())

            for _, row in req_df.iterrows():
                rid = row["id"]
                payload = _as_dict(row.get("payload"))
                status = (row.get("status") or "pending").lower()
                if status != "pending":
                    continue

                # Header line for the expander
                game_nm = payload.get("game_name") or id_to_game.get(
                    payload.get("game_id"), "Unknown game"
                )
                submitted_by = row.get("created_by") or "Unknown"
                when_iso = payload.get("played_at") or "—"
                header = f"📝 {game_nm} • {when_iso} • by {submitted_by}"

                with st.expander(header, expanded=False):
                    # --- Editable session header ---
                    lcol, mcol, rcol = st.columns([2, 1, 1])

                    game_names_list = list(id_to_game.values()) or ["(no games)"]
                    try:
                        initial_game_idx = game_names_list.index(game_nm)
                    except ValueError:
                        initial_game_idx = 0

                    sel_game_name = lcol.selectbox(
                        "Game",
                        options=game_names_list,
                        index=initial_game_idx,
                        key=f"ap_game_{rid}",
                    )
                    sel_game_id = game_name_to_id.get(sel_game_name)

                    # datetime
                    try:
                        dt_default = pd.to_datetime(when_iso)
                        if pd.isna(dt_default):
                            raise ValueError
                    except Exception:
                        dt_default = pd.Timestamp.now()

                    d_val = mcol.date_input(
                        "Date", value=dt_default.date(), key=f"ap_date_{rid}"
                    )
                    t_val = rcol.time_input(
                        "Time",
                        value=dt_default.time().replace(microsecond=0),
                        key=f"ap_time_{rid}",
                    )

                    location_val = st.text_input(
                        "Location",
                        value=(payload.get("location") or ""),
                        key=f"ap_loc_{rid}",
                    )
                    notes_val = st.text_area(
                        "Notes",
                        value=(payload.get("notes") or ""),
                        key=f"ap_notes_{rid}",
                    )

                    # --- Editable participant entries ---
                    entries = payload.get("entries", []) or []
                    if not entries:
                        st.warning("No participant entries in this request.")
                        edited = pd.DataFrame(
                            columns=["pname", "team", "position", "points", "is_winner"]
                        )
                    else:
                        edf = pd.DataFrame(entries)
                        # normalize columns
                        for col in ["pname", "team", "position", "points", "is_winner"]:
                            if col not in edf.columns:
                                edf[col] = None
                        edf["pname"] = edf["pname"].astype(str)
                        edf["team"] = edf["team"].fillna("").astype(str)
                        edf["position"] = pd.to_numeric(
                            edf["position"], errors="coerce"
                        ).astype("Int64")
                        edf["points"] = pd.to_numeric(
                            edf["points"], errors="coerce"
                        ).fillna(0.0)
                        edf["is_winner"] = edf["is_winner"].fillna(False).astype(bool)

                        edited = st.data_editor(
                            edf[["pname", "team", "position", "points", "is_winner"]],
                            key=f"ap_editor_{rid}",
                            use_container_width=True,
                            num_rows="fixed",
                            column_config={
                                "pname": st.column_config.SelectboxColumn(
                                    "Player",
                                    options=all_player_names,
                                    required=True,
                                ),
                                "team": st.column_config.TextColumn(
                                    "Team",
                                    help="Blank for FFA; label (A/B/… or COOP/SOLO) for team modes",
                                ),
                                "position": st.column_config.NumberColumn(
                                    "Pos", min_value=1, max_value=99
                                ),
                                "points": st.column_config.NumberColumn(
                                    "Pts", step=1.0
                                ),
                                "is_winner": st.column_config.CheckboxColumn("Winner"),
                            },
                        )

                    # --- Actions ---
                    ac1, ac2, ac3 = st.columns(3)

                    if ac1.button("✅ Approve & Save", key=f"ap_btn_approve_{rid}"):
                        # Compose session
                        try:
                            played_at_dt = datetime.combine(d_val, t_val)
                        except Exception:
                            st.error("Invalid date/time.")
                            st.stop()

                        sid = _uuid()
                        ok_session = sb_insert(
                            "sessions",
                            {
                                "id": sid,
                                "game_id": sel_game_id,
                                "played_at": played_at_dt.isoformat(),
                                "location": (location_val or "").strip() or None,
                                "notes": (notes_val or "").strip() or None,
                            },
                        )
                        if not ok_session:
                            st.error("Failed to create session.")
                            st.stop()

                        all_ok = True
                        for e in edited.to_dict("records"):
                            pid = name_to_id.get(e["pname"])
                            if not pid:
                                st.error(f"Unknown player: {e['pname']}")
                                all_ok = False
                                break
                            row_sp = {
                                "id": _uuid(),
                                "session_id": sid,
                                "player_id": pid,
                                "team": (e.get("team") or None),
                                "position": (
                                    int(e["position"])
                                    if pd.notna(e["position"])
                                    else None
                                ),
                                "points": float(e.get("points") or 0.0),
                                "is_winner": bool(e.get("is_winner", False)),
                            }
                            if not sb_insert("session_players", row_sp):
                                all_ok = False
                                break

                        if all_ok:
                            sb_update_by_id(
                                "session_requests",
                                rid,
                                {
                                    "status": "approved",
                                    "processed_at": datetime.utcnow().isoformat(),
                                },
                            )
                            st.success(
                                "Approved. Session created and request marked as approved."
                            )
                            st.cache_data.clear()
                        else:
                            st.error(
                                "Some participant rows failed to save. The session may be partial. Review errors above."
                            )

                    if ac2.button("❌ Reject", key=f"ap_btn_reject_{rid}"):
                        if sb_update_by_id(
                            "session_requests",
                            rid,
                            {
                                "status": "rejected",
                                "processed_at": datetime.utcnow().isoformat(),
                            },
                        ):
                            st.info("Request rejected.")
                            st.cache_data.clear()

                    if ac3.button("🗑️ Delete", key=f"ap_btn_delete_{rid}"):
                        if sb_delete_by_id("session_requests", rid):
                            st.warning("Request deleted.")
                            st.cache_data.clear()

    # ------------------------- Economy / Bag Management -------------------------

    with tabs[6]:
        st.subheader("Economy (drops, crafting, fonts)")
        st.caption("Edit JSON and save. Safe defaults are used if JSON is invalid.")

        import json

        def _json_area(label, obj, key):
            raw = st.text_area(
                label, value=json.dumps(obj, indent=2), height=260, key=key
            )
            try:
                parsed = json.loads(raw)
                ok = True
            except Exception as e:
                st.error(f"{label}: invalid JSON → {e}")
                parsed, ok = obj, False
            return parsed, ok

        cur_item_db, ok1 = _json_area("ITEM_DB", ITEM_DB, "econ_items")
        cur_box_tables, ok2 = _json_area("BOX_TABLES", BOX_TABLES, "econ_boxes")
        cur_craft, ok3 = _json_area("CRAFT_RECIPES", CRAFT_RECIPES, "econ_craft")
        cur_fonts, ok4 = _json_area("FONT_CATALOG", FONT_CATALOG, "econ_fonts")

        colA, colB = st.columns([1, 1])
        if colA.button(
            "Save all (to config)",
            type="primary",
            disabled=not (ok1 and ok2 and ok3 and ok4),
        ):
            all_ok = True
            all_ok &= _save_json_config("economy_item_db", cur_item_db)
            all_ok &= _save_json_config("economy_box_tables", cur_box_tables)
            all_ok &= _save_json_config("economy_craft_recipes", cur_craft)
            all_ok &= _save_json_config("economy_font_catalog", cur_fonts)
            if all_ok:
                st.success("Saved. Reload app to take effect.")
        if colB.button("Revert to built-in defaults", type="secondary"):
            _save_json_config("economy_item_db", ITEM_DB)
            _save_json_config("economy_box_tables", BOX_TABLES)
            _save_json_config("economy_craft_recipes", CRAFT_RECIPES)
            _save_json_config("economy_font_catalog", FONT_CATALOG)
            st.warning("Defaults written to config  . Reload app to apply.")
