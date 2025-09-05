# galios_den_game.py
# -----------------------------------------------------------------------------
# Galio's Den - Daily Mini-Game for Streamlit (instant transitions)
# - Unlimited time per run
# - No save during combat; save/exit only on victory (death auto-saves 0)
# - Mercy auto-saves ceil(score/2) and ends run
# - Uncapped HP
# - Immediate reruns on ALL state-changing actions (no double-clicks)
# - Unique button keys (no DuplicateElementId)
# - UTC-aware timestamps
# - Leaderboard: highest score, then shortest time
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, date, UTC
from typing import Any, Dict

import pandas as pd
import streamlit as st

# --------------------------- Game Constants ----------------------------------

PLAYER_MAX_HP = 100
BASE_ATTACK_MAX = 40  # player attack is 1..(BASE_ATTACK_MAX + score//3)
RESET_UTC_LABEL = "03:00"

# --- Weekly window (fixed 7-day buckets) -------------------------------------

WEEK_ANCHOR_DATE_UTC = date(2024, 1, 1)  # change if you want a different fixed anchor


def _current_week_bounds_keys_utc(rollover_hour: int = 3):
    """
    Returns (week_start_key, week_end_key, week_start_date, week_end_date) for a fixed
    7-day bucket aligned to WEEK_ANCHOR_DATE_UTC. Day boundary honors the same 03:00 UTC rollover.
    """
    now = datetime.now(UTC)
    # respect daily rollover hour for which "day" we're in
    current_day = (
        (now.date() - timedelta(days=1)) if now.hour < rollover_hour else now.date()
    )
    # which 7-day bucket since anchor?
    delta_days = (current_day - WEEK_ANCHOR_DATE_UTC).days
    bucket_idx = max(0, delta_days // 7)
    week_start = WEEK_ANCHOR_DATE_UTC + timedelta(days=bucket_idx * 7)
    week_end = week_start + timedelta(days=6)
    return (
        week_start.strftime("%Y-%m-%d"),
        week_end.strftime("%Y-%m-%d"),
        week_start,
        week_end,
    )


# Cleaned list
GALIOS_DEN_ENEMIES = [
    "Gatekeeper Galio",
    "One Punch Man",
    "Skeletons",
    "Baby Zombie",
    "Raging Warrior",
    "Faceless Assassin",
    "Ghoul",
    "Wisp",
    "Slasher",
    "Warden",
    "Phantom",
    "Meep",
    "Dragon",
    "Baby Dragon",
    "Oversized Spider",
    "Reaper",
    "Grim Reaper",
    "Sleepy Murderer",
    "Shock Wizard",
    "Earth Wizard",
    "Fire Wizard",
    "Windcaster Wizard",
    "Craving Giant Rat",
    "Weeping Angel",
    "Ninja Flamingo",
    "Yourself but better",
    "Wolves",
    "Grump",
    "Velociraptors",
    "Jon Snow",
    "Harambe",
    "Caesar",
    "Wanderer",
    "Giant Penguin",
    "Chucky",
    "Silver Rat",
    "Gold Rat",
    "Mutant",
    "Fish-man",
    "Con-man",
    "Ludicurious Clown",
    "Man-eater Plant",
    "Giant Man-eater Plant",
    "Giant Worm",
    "Warper",
    "Sweapster",
    "Evil Speedster",
    "Sharkface",
    "Freddy",
    "Evil Poro",
    "Dancing Demons",
    "Servant of Galio",
    "Descendant of Galio",
    "Karate Master Slender",
    "Armed Jellyfish",
    "Armed Squid",
    "Hungry Cyclops",
    "Clumsy Minotaur",
    "Zombie",
    "Vampire",
]

# --------------------------- Helpers -----------------------------------------


def _galios_den_is_servant(enemy: str, galio_health: int) -> int:
    """If a Servant appears, Galio gets tougher later on."""
    if enemy == "Servant of Galio":
        return galio_health + 70
    return galio_health


def _percent(cur: int, maxv: int) -> int:
    """Convert to 0-100 for st.progress; guards div/0."""
    if maxv <= 0:
        return 0
    v = int(round(100 * max(0.0, min(1.0, cur / maxv))))
    return v


def _btn_key(state: Dict[str, Any], name: str) -> str:
    """Stable unique keys to avoid StreamlitDuplicateElementId."""
    started = state.get("started_at") or "nostart"
    fight_id = state.get("fight_id", 0)
    score = state.get("score", 0)
    return f"gd_{name}_{started}_{fight_id}_{score}"


def _galios_den_finalize_and_save(
    state: Dict[str, Any], player_id: str, supabase, _today_key_utc, _set_best_of_day
) -> None:
    """Save the final game score to the database (duration captured; optional override)."""
    if not supabase:
        return
    try:
        date_key = _today_key_utc()

        started_at = state.get("started_at")
        finished_at = datetime.now(UTC)

        # Score override (e.g., death=0, mercy=ceil(score/2))
        score = int(state.get("score", 0))
        if state.get("force_score_override") is not None:
            score = int(state["force_score_override"])

        # Duration in seconds
        duration_s = None
        if started_at and isinstance(started_at, datetime):
            try:
                duration_s = int((finished_at - started_at).total_seconds())
            except Exception:
                duration_s = None

        payload = {
            "date_key": date_key,
            "player_id": player_id,
            "score": score,
            "rounds_played": score,  # enemies defeated
            "duration_s": duration_s,  # time-to-completion
            "started_at": started_at.isoformat() if started_at else None,
            "finished_at": finished_at.isoformat(),
            "is_best_for_day": False,
            "meta": {"minigame": "Galio's Den"},
        }

        supabase.table("minigame_scores").insert(payload).execute()
        _set_best_of_day(date_key, player_id)
    except Exception as e:
        st.error(f"Failed to save score: {e}")


# --------------------------- UI Entrypoint -----------------------------------


def render_galios_den_game(
    players_df,
    _login_gate,
    supabase,
    _today_key_utc,
    _set_best_of_day,
    _run_daily_awards_if_needed,
    _equipped_for_player,
    _inject_font_css,
) -> None:
    st.header("🏰 Daily Mini-Game — Galio's Den")
    _run_daily_awards_if_needed()

    pid = _login_gate(players_df, key_prefix="galios_den")
    if not pid:
        st.info("Login to enter Galio's Den.")
        return

    # Best-of-day (today) for player metric
    today_key = _today_key_utc()
    best_today = 0
    try:
        if supabase:
            r = (
                supabase.table("minigame_scores")
                .select("score, duration_s")
                .eq("date_key", today_key)
                .eq("player_id", pid)
                .eq("is_best_for_day", True)
                .order("score", desc=True)
                .order("duration_s", desc=False)
                .limit(1)
                .execute()
            )
            rows = r.data or []
            if rows:
                best_today = int(rows[0]["score"])
    except Exception:
        pass

    # Weekly window (for leaderboard reset every 7 days)
    week_start_key, week_end_key, _, _ = _current_week_bounds_keys_utc(3)

    # Styled header
    with st.container():
        colA, colB, colC = st.columns([2, 1, 1])

        eq = _equipped_for_player(pid)
        scope = f"pfont-{pid}"
        _inject_font_css(eq.get("font"), scope)

        pname = "Player"
        try:
            if not players_df.empty and {"id", "name"} <= set(players_df.columns):
                pname = str(players_df.set_index("id")["name"].get(pid) or "Player")
        except Exception:
            pass

        colA.markdown(
            f"<h3 class='{scope}' style='margin:0;'>Welcome, {pname}!</h3>",
            unsafe_allow_html=True,
        )
        colB.metric("Best today", best_today)
        colC.metric("Resets (UTC)", RESET_UTC_LABEL)

    # Game state
    state = st.session_state.setdefault("galios_den_state", {})
    running = bool(state.get("running", False))

    if not running:
        st.markdown("**Welcome to Galio's Den!**")
        st.markdown(
            "- Start with 100 HP and 3 health potions  \n"
            "- Attack damage scales as you progress  \n"
            "- Special bosses: **Gatekeeper Galio** and **One Punch Man**  \n"
            "- Health potions can drop after victories  \n"
            "- **No timer — take as long as you like**"
        )

        if st.button(
            "⚔️ Enter Galio's Den", type="primary", width="stretch", key="gd_enter"
        ):
            state.clear()
            state.update(
                {
                    "running": True,
                    "started_at": datetime.now(UTC),
                    "score": 0,
                    "health": PLAYER_MAX_HP,
                    "num_health_potions": 3,
                    "galio_health": 175,  # initial Galio HP baseline
                    "combat_state": None,  # "fighting" | "victory" | None
                    "current_enemy": None,
                    "enemy_health": 0,
                    "enemy_max_health": 0,
                    "message": "You enter the dark dungeon...",
                    "fight_id": 0,  # increments each new enemy
                    "_saved_once": False,
                    "force_score_override": None,
                }
            )
            # Instant transition to first fight
            st.rerun()

        st.divider()
    else:
        render_game_interface(state, pid, supabase, _today_key_utc, _set_best_of_day)

    # Weekly leaderboard (resets every 7 days); daily rewards still handled separately
    render_weekly_leaderboard(week_start_key, week_end_key, supabase, players_df)


# --------------------------- Gameplay Screens --------------------------------


def render_game_interface(
    state: Dict[str, Any], pid: str, supabase, _today_key_utc, _set_best_of_day
) -> None:
    """Render the main game interface during gameplay."""
    # If we need a new enemy, spawn it and rerun immediately to render fresh UI.
    if (
        state.get("combat_state") not in ("fighting", "victory")
        and int(state.get("health", 0)) > 0
    ):
        spawn_new_enemy(state)
        st.rerun()
        return

    # Stats header (uncapped HP)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("❤️ Health", f"{int(state.get('health', 0))} HP")
    col2.metric("🧪 Potions", int(state.get("num_health_potions", 0)))
    col3.metric("🏆 Score", int(state.get("score", 0)))
    col4.metric(
        "⚔️ Attack Power", f"1-{BASE_ATTACK_MAX + (int(state.get('score', 0)) // 3)}"
    )

    # Any message from last action
    msg = state.get("message", "")
    if msg:
        st.info(msg)

    # Ended (health <= 0) → auto-save (0 unless a mercy override was set)
    if int(state.get("health", 0)) <= 0:
        if not state.get("_saved_once", False):
            if state.get("force_score_override") is None:
                state["force_score_override"] = 0  # death = zero score
            _galios_den_finalize_and_save(
                state, pid, supabase, _today_key_utc, _set_best_of_day
            )
            state["_saved_once"] = True
        # Reset to landing view and show it immediately
        st.success("💾 Score saved. Thanks for playing!")
        st.balloons()
        state.clear()
        st.rerun()
        return

    # Screen routing
    if state.get("combat_state") == "victory":
        render_victory_screen(state, pid, supabase, _today_key_utc, _set_best_of_day)
    else:
        render_combat_screen(state, pid, supabase, _today_key_utc, _set_best_of_day)


def spawn_new_enemy(state: Dict[str, Any]) -> None:
    """Spawn a new enemy and set up combat."""
    score = int(state.get("score", 0))
    enemy = random.choice(GALIOS_DEN_ENEMIES)

    # Servant boosts future Galio
    state["galio_health"] = _galios_den_is_servant(
        enemy, int(state.get("galio_health", 185))
    )

    # Enemy stats scale with progress
    if enemy == "Gatekeeper Galio" and int(state.get("galio_health", 0)) > 0:
        enemy_health = int(state.get("galio_health", 185)) + (5 * score)
        state["message"] = (
            f"🛡️ **{enemy}** appears! 'You cannot run away from me, I am going to smash you!!!'"
        )
    elif enemy == "One Punch Man":
        enemy_health = 1  # 1 HP but terrifying counter
        state["message"] = (
            "👊 **One Punch Man** appears! 'I will become stronger. Have you seen Genos? Btw.. today is discount day.'"
        )
    else:
        max_enemy_health = 55 + (3 * score)
        min_enemy_health = 1 + (3 * score)
        enemy_health = random.randint(min_enemy_health, max(1, max_enemy_health))
        state["message"] = f"👹 **{enemy}** has appeared!"

    state.update(
        {
            "combat_state": "fighting",
            "current_enemy": enemy,
            "enemy_health": int(enemy_health),
            "enemy_max_health": int(enemy_health),
            "fight_id": int(state.get("fight_id", 0)) + 1,
        }
    )


def render_combat_screen(
    state: Dict[str, Any], pid: str, supabase, _today_key_utc, _set_best_of_day
) -> None:
    """Render the combat interface (no save options here)."""
    enemy = state.get("current_enemy", "Unknown Enemy")
    enemy_health = int(state.get("enemy_health", 0))
    enemy_max_health = int(state.get("enemy_max_health", 1))

    # Enemy health bar
    st.progress(
        _percent(enemy_health, enemy_max_health),
        text=f"🐉 {enemy}: {enemy_health}/{enemy_max_health} HP",
    )

    st.markdown("### What will you do?")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "⚔️ Attack", type="primary", width="stretch", key=_btn_key(state, "attack")
        ):
            handle_attack(state)
            # Always rerun so damage, HP bars, and victory/death screen show instantly
            st.rerun()

    with col2:
        potions = int(state.get("num_health_potions", 0))
        btn_label = f"🧪 Drink Potion ({potions})" if potions > 0 else "🧪 No Potions"
        if st.button(
            btn_label,
            disabled=(potions <= 0),
            width="stretch",
            key=_btn_key(state, "heal"),
        ):
            handle_heal(state)
            # Rerun to reflect new HP immediately
            st.rerun()

    with col3:
        if enemy == "Gatekeeper Galio":
            if st.button(
                "🙏 Beg Forgiveness", width="stretch", key=_btn_key(state, "beg")
            ):
                handle_beg_forgiveness(
                    state, pid, supabase, _today_key_utc, _set_best_of_day
                )
                st.rerun()
        else:
            if st.button("🏃 Run Away", width="stretch", key=_btn_key(state, "run")):
                handle_run_away(state)
                st.rerun()


def handle_attack(state: Dict[str, Any]) -> None:
    """Handle player attack + enemy retaliation."""
    score = int(state.get("score", 0))
    enemy = state.get("current_enemy", "")

    # Player damage
    max_attack_damage = BASE_ATTACK_MAX + (score // 3)
    damage_dealt = random.randint(1, max(1, max_attack_damage))

    # Enemy damage
    if enemy == "Gatekeeper Galio":
        max_galio_attack = 23 + (2 * score)
        damage_taken = random.randint(1, max(1, max_galio_attack))
    elif enemy == "One Punch Man":
        damage_taken = max(1, 5 * (score + 1) * (score + 1))
    else:
        max_enemy_attack = 26 + score
        damage_taken = random.randint(1, max(1, max_enemy_attack))

    # Apply results
    state["enemy_health"] = max(0, int(state.get("enemy_health", 0)) - damage_dealt)
    state["health"] = max(0, int(state.get("health", 0)) - damage_taken)

    # Message
    state["message"] = (
        f"⚔️ You strike {enemy} for {damage_dealt} damage. You receive {damage_taken} damage in retaliation."
    )

    # Outcome checks
    if int(state.get("health", 0)) <= 0:
        state["combat_state"] = None  # game end path (auto-save in interface)
        if enemy == "Gatekeeper Galio":
            state[
                "message"
            ] += "\n💀 He broke you too, like the other knights who tried to overcome his glory."
        elif enemy == "One Punch Man":
            state[
                "message"
            ] += "\n💀 Got one-punched! A weak body, perhaps, but a brave heart."
        else:
            state["message"] += "\n💀 You have been destroyed — your journey ends here."
    elif int(state.get("enemy_health", 0)) <= 0:
        state["combat_state"] = "victory"
        state["score"] = score + 1
        state["message"] = (
            f"🎉 {enemy} was defeated! You have {int(state.get('health', 0))} HP left."
        )


def handle_heal(state: Dict[str, Any]) -> None:
    """Drink a potion to heal (no cap)."""
    potions = int(state.get("num_health_potions", 0))
    if potions <= 0:
        state["message"] = "❌ You are out of health potions!"
        return

    score = int(state.get("score", 0))
    heal_amount = 30 + (8 * (score // 2))
    state["health"] = int(state.get("health", 0)) + heal_amount  # no cap
    state["num_health_potions"] = potions - 1
    state["message"] = (
        f"🧪 You drink a health potion, healing {heal_amount}. "
        f"HP: {state.get('health', 0)} • Potions left: {state.get('num_health_potions', 0)}."
    )


def handle_beg_forgiveness(
    state: Dict[str, Any], pid: str, supabase, _today_key_utc, _set_best_of_day
) -> None:
    """Begging Gatekeeper Galio for mercy: half score (ceil) & auto-save, or instant death."""
    mercy = random.choice([True, False])
    cur = int(state.get("score", 0))

    if mercy:
        awarded = int(math.ceil(cur / 2.0))
        state["force_score_override"] = awarded
        state["message"] = f"🙏 Galio shows mercy! Final score: {awarded} (auto-saved)."
        if not state.get("_saved_once", False):
            _galios_den_finalize_and_save(
                state, pid, supabase, _today_key_utc, _set_best_of_day
            )
            state["_saved_once"] = True
        state.clear()  # end run now
    else:
        state["message"] = "💀 He smashed you in pieces. Galio has no mercy!"
        state["combat_state"] = None
        state["health"] = 0  # triggers death auto-save in interface


def handle_run_away(state: Dict[str, Any]) -> None:
    """Run away from the current enemy (no score gain)."""
    enemy = state.get("current_enemy", "")
    if enemy == "One Punch Man":
        state["message"] = "🏃 You ran away from One Punch Man! Wise choice."
    else:
        state["message"] = f"🏃 You ran away from {enemy}!"
    state["combat_state"] = None  # next render spawns new enemy


def render_victory_screen(
    state: Dict[str, Any], pid: str, supabase, _today_key_utc, _set_best_of_day
) -> None:
    """Post-victory loot + choices (the only place you can save/exit by choice)."""
    enemy = state.get("current_enemy", "")
    score = int(state.get("score", 0))

    # Potion drop chance decreases as you get stronger
    drop_chance = max(5, 45 - (3 * (score // 2)))
    if random.randint(1, 100) <= drop_chance:
        state["num_health_potions"] = int(state.get("num_health_potions", 0)) + 1
        state[
            "message"
        ] += f"\n🧪 {enemy} dropped a health potion! Potions: {state.get('num_health_potions', 0)}."

    st.markdown("### What would you like to do now?")
    c1, c2, c3 = st.columns(3)

    if c1.button(
        "⚔️ Continue Fighting",
        type="primary",
        width="stretch",
        key=_btn_key(state, "vict_continue"),
    ):
        if enemy == "Gatekeeper Galio":
            state["message"] = (
                "🏆 You press on after felling Gatekeeper Galio — impressive!"
            )
        elif enemy == "One Punch Man":
            state["message"] = "💪 You survived and still continue — nerve of steel!"
        else:
            state["message"] = "⚔️ You continue your adventure!"
        state["combat_state"] = None
        st.rerun()

    if c2.button(
        "🚪 Exit Dungeon (no save)",
        width="stretch",
        key=_btn_key(state, "vict_exit_nosave"),
    ):
        st.warning("Run discarded.")
        state.clear()
        st.rerun()

    if c3.button(
        "💾 Save Score & Exit", width="stretch", key=_btn_key(state, "vict_save_exit")
    ):
        _galios_den_finalize_and_save(
            state, pid, supabase, _today_key_utc, _set_best_of_day
        )
        st.success("Saved! See you next time.")
        st.balloons()
        state.clear()
        st.rerun()


# --------------------------- Leaderboard -------------------------------------


def render_weekly_leaderboard(
    week_start_key: str, week_end_key: str, supabase, players_df
) -> None:
    """Render the weekly leaderboard (best weekly score per player, ties by shortest time)."""
    st.subheader("🏆 This Week's Leaderboard")
    st.caption(f"Week window: {week_start_key} → {week_end_key} (resets every 7 days)")
    try:
        if supabase:
            rows = (
                supabase.table("minigame_scores")
                .select("player_id, score, duration_s, finished_at, date_key")
                .gte("date_key", week_start_key)
                .lte("date_key", week_end_key)
                .eq("is_best_for_day", True)  # consider each day's personal bests
                .execute()
                .data
                or []
            )
            if rows:
                df = pd.DataFrame(rows)

                # one row per player: best weekly score, tie-break shortest time, then earliest finish
                df["_sort_time"] = pd.to_numeric(df["duration_s"]).fillna(10**12)
                df["_sort_finish"] = pd.to_datetime(
                    df.get("finished_at", None), errors="coerce"
                )
                df = df.sort_values(
                    ["score", "_sort_time", "_sort_finish"],
                    ascending=[False, True, True],
                )
                df = df.drop_duplicates(subset=["player_id"], keep="first")

                # names
                names = (
                    players_df[["id", "name"]]
                    if (isinstance(players_df, pd.DataFrame) and not players_df.empty)
                    else pd.DataFrame()
                )
                if not names.empty:
                    df = df.merge(
                        names.rename(columns={"id": "player_id"}),
                        on="player_id",
                        how="left",
                    )

                # final display sort
                df = df.sort_values(["score", "_sort_time"], ascending=[False, True])

                df = df.rename(
                    columns={
                        "name": "Player",
                        "score": "Best Score",
                        "duration_s": "Best Time (s)",
                    }
                )
                df.insert(0, "#", range(1, len(df) + 1))
                st.dataframe(
                    df[["#", "Player", "Best Score", "Best Time (s)"]],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info(
                    "No scores recorded this week yet. Be the first to enter Galio's Den!"
                )
    except Exception as e:
        st.error(f"Leaderboard failed: {e}")


# (Kept for reference; no longer used)
def render_daily_leaderboard(today_key: str, supabase, players_df) -> None:
    """Render the daily leaderboard for Galio's Den."""
    st.subheader("🏆 Today's Leaderboard")
    try:
        if supabase:
            best_rows = (
                supabase.table("minigame_scores")
                .select("player_id, score, duration_s")
                .eq("date_key", today_key)
                .eq("is_best_for_day", True)
                .execute()
                .data
                or []
            )
            if best_rows:
                df = pd.DataFrame(best_rows)

                names = (
                    players_df[["id", "name"]]
                    if (isinstance(players_df, pd.DataFrame) and not players_df.empty)
                    else pd.DataFrame()
                )
                if not names.empty:
                    df = df.merge(
                        names.rename(columns={"id": "player_id"}),
                        on="player_id",
                        how="left",
                    )

                # Sort: highest score, then shortest time (None at bottom)
                sort_time = df["duration_s"].fillna(10**12)
                df = df.assign(_sort_time=sort_time).sort_values(
                    ["score", "_sort_time"], ascending=[False, True]
                )

                df = df.rename(
                    columns={
                        "name": "Player",
                        "score": "Enemies Defeated",
                        "duration_s": "Time (s)",
                    }
                )
                df.insert(0, "#", range(1, len(df) + 1))
                st.dataframe(
                    df[["#", "Player", "Enemies Defeated", "Time (s)"]],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No scores yet today. Be the first to enter Galio's Den!")
    except Exception as e:
        st.error(f"Leaderboard failed: {e}")
