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
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import plotly.graph_objects as go  # for H2H heatmap

# ─────────────────────────────────────────────────────────────────────────────
# Environment & Supabase
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()


def _sec(key: str, default: str = "") -> str:
    """Prefer st.secrets over env if present."""
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


SUPABASE_URL = _sec("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = _sec("SUPABASE_ANON_KEY", "").strip()
ADMIN_PASSWORD = _sec("ADMIN_PASSWORD", "change-me")

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

st.set_page_config(page_title="🎯 Game Sessions Tracker", page_icon="🏆", layout="wide")

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
        sessions["played_at"] = pd.to_datetime(
            sessions["played_at"], errors="coerce"
        ).dt.date

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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (leaderboard_df, h2h_df, recent_df) using the provided ELO configuration.
    """
    if joined.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = joined.copy()

    if game_filter:
        df = df[df["game_name"].isin(game_filter)]

    if date_range and all(date_range):
        start, end = date_range
        df = df[(df["played_at"] >= start) & (df["played_at"] <= end)]

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Resolve winners per-session
    df = df.sort_values(["played_at", "session_id"]).reset_index(drop=True)
    df["resolved_winner"] = False
    for sid, g in df.groupby("session_id"):
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

    # Order leaderboard
    lb = gp.sort_values(
        ["wins", "win_rate", "elo"], ascending=[False, False, False]
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

    return lb, h2h_df, recent_slim


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


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["General", "Admin"], horizontal=True)
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

    joined = sb_select_sessions_joined()
    games_df = sb_select("games")  # for single-game info/BGG card

    if joined.empty:
        st.info("No session data yet. Add sessions from the Admin tab.")
    else:
        cfg = load_config()  # use admin settings

        # ---- Filters (General) ----
        c1, c2 = st.columns([2, 1])
        game_names = sorted(joined["game_name"].dropna().unique().tolist())

        # Use a stable key + initialize session state BEFORE creating the widget
        ms_key = "game_filter"
        if ms_key not in st.session_state:
            st.session_state[ms_key] = game_names[:]  # start with all games selected

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

        min_date = joined["played_at"].min()
        max_date = joined["played_at"].max()
        start_date, end_date = c2.date_input("Date range", value=(min_date, max_date))

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
            filt = joined[
                (joined["game_name"] == gname)
                & (joined["played_at"].between(start_date, end_date))
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

        lb, h2h, recent = compute_leaderboard(
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

        if lb.empty:
            st.info("No data for current filters.")
        else:
            # KPI
            k1, k2, k3, k4 = st.columns(4)
            total_sessions = joined[
                (joined["game_name"].isin(selected_games))
                & (joined["played_at"].between(start_date, end_date))
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
            champ = lb.iloc[0] if not lb.empty else None
            longest = lb.loc[lb["best_streak"].idxmax()] if not lb.empty else None
            eligible = lb[lb["games_played"] >= 3]
            best_wr = (
                eligible.loc[eligible["win_rate"].idxmax()]
                if not eligible.empty
                else None
            )
            most_active = lb.loc[lb["games_played"].idxmax()] if not lb.empty else None

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

            with cA:
                metric_card(
                    "🏆 Champion",
                    champ["display_name"] if champ is not None else None,
                    f"ELO {int(champ['elo'])}" if champ is not None else None,
                )

            with cB:
                metric_card(
                    "🔥 Longest Streak",
                    longest["display_name"] if longest is not None else None,
                    (
                        f"{int(longest['best_streak'])} wins"
                        if longest is not None
                        else None
                    ),
                )

            with cC:
                metric_card(
                    "🎯 Best Win% (≥3 GP)",
                    best_wr["display_name"] if best_wr is not None else None,
                    f"{best_wr['win_rate']*100:.0f}%" if best_wr is not None else None,
                )

            with cD:
                metric_card(
                    "👥 Most Active",
                    most_active["display_name"] if most_active is not None else None,
                    (
                        f"{int(most_active['games_played'])} GP"
                        if most_active is not None
                        else None
                    ),
                )

            # ── Sexy Leaderboard: rank, medals, win% progress, streak flair ──
            lb_disp = lb.copy()
            lb_disp.insert(0, "#", np.arange(1, len(lb_disp) + 1))
            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
            lb_disp.insert(1, "🏅", lb_disp["#"].map(medals).fillna(""))
            lb_disp["🔥"] = np.where(lb_disp["current_streak"] >= 3, "🔥", "")
            lb_disp["Win%"] = lb_disp["win_rate"] * 100.0  # 0..100 for ProgressColumn

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
            st.dataframe(
                lb_disp[show_cols].rename(
                    columns={
                        "display_name": "Player",
                        "games_played": "GP",
                        "wins": "W",
                        "avg_position": "Avg Pos",
                        "avg_points": "Avg Pts",
                        "current_streak": "Streak",
                        "best_streak": "Best Streak",
                        "elo": "ELO",
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
                    "Last": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "🔥": st.column_config.TextColumn(
                        width="small", help="Hot streak (3+ wins)"
                    ),
                },
            )

            st.subheader("Head-to-Head (times row finished ahead of column)")
            render_h2h_heatmap(h2h, title="Head-to-Head (row finished ahead of column)")

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
            )

            st.download_button(
                "Download leaderboard (CSV)",
                data=lb.to_csv(index=False).encode("utf-8"),
                file_name="leaderboard.csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# Admin
# ─────────────────────────────────────────────────────────────────────────────

if mode == "Admin":
    st.title("🛠️ Admin")

    tabs = st.tabs(
        ["Add Session", "Players", "Games", "Manage Sessions", "ELO Settings"]
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
                        "played_at": str(played_at),
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
            st.dataframe(
                players_df[["name", "nickname", "joined_on"]].rename(
                    columns={"name": "Player", "nickname": "Nick"}
                ),
                use_container_width=True,
                hide_index=True,
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
                if sb_delete_by_id("sessions", to_delete):
                    st.success("Session deleted (participants removed via cascade).")
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
