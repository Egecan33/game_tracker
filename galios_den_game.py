# galios_den_game.py
# -----------------------------------------------------------------------------
# Galio's Den - Daily Mini-Game for Streamlit
# - Unlimited time per run (no countdowns)
# - Uses the same Supabase "minigame_scores" table and best-of-day logic
# - Preserves your existing function signatures to avoid churn
# -----------------------------------------------------------------------------

from __future__ import annotations

import random
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import streamlit as st

# --------------------------- Game Constants ----------------------------------

PLAYER_MAX_HP = 100
BASE_ATTACK_MAX = 43  # player attack is 1..(BASE_ATTACK_MAX + score//3)

# Cleaned list (typo fixes + removed offensive wording, duplicates trimmed)
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
        return galio_health + 75
    return galio_health


def _percent(cur: int, maxv: int) -> int:
    """Convert to 0-100 for st.progress; guards div/0."""
    if maxv <= 0:
        return 0
    v = int(round(100 * max(0.0, min(1.0, cur / maxv))))
    return v


def _galios_den_finalize_and_save(
    state: Dict[str, Any], player_id: str, supabase, _today_key_utc, _set_best_of_day
) -> None:
    """Save the final game score to the database."""
    if not supabase:
        return
    try:
        date_key = _today_key_utc()
        started_at = state.get("started_at")
        finished_at = datetime.utcnow()
        score = int(state.get("score", 0))

        payload = {
            "date_key": date_key,
            "player_id": player_id,
            "score": score,
            "rounds_played": score,  # score = enemies defeated
            # Unlimited-time game: allow NULL duration (omit if your column is NOT NULL)
            "duration_s": None,
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
    """Render the complete Galio's Den game interface."""
    st.header("🏰 Daily Mini-Game — Galio's Den")
    _run_daily_awards_if_needed()

    # Login
    pid = _login_gate(players_df, key_prefix="galios_den")
    if not pid:
        st.info("Login to enter Galio's Den.")
        return

    # Best-of-day (today)
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
        colC.metric("Resets (UTC)", "03:00")

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

        if st.button("⚔️ Enter Galio's Den", type="primary"):
            state.clear()
            state.update(
                {
                    "running": True,
                    "started_at": datetime.utcnow(),
                    "score": 0,
                    "health": PLAYER_MAX_HP,
                    "num_health_potions": 3,
                    "galio_health": 185,  # initial Galio HP baseline
                    "combat_state": None,  # "fighting" | "victory" | None
                    "current_enemy": None,
                    "enemy_health": 0,
                    "enemy_max_health": 0,
                    "message": "You enter the dark dungeon...",
                }
            )
            st.rerun()

        st.divider()
    else:
        # Running game loop
        render_game_interface(state, pid, supabase, _today_key_utc, _set_best_of_day)

    # Daily leaderboard
    render_daily_leaderboard(today_key, supabase, players_df)


# --------------------------- Gameplay Screens --------------------------------


def render_game_interface(
    state: Dict[str, Any], pid: str, supabase, _today_key_utc, _set_best_of_day
) -> None:
    """Render the main game interface during gameplay."""
    # Stats header
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("❤️ Health", f"{state.get('health', 0)}/{PLAYER_MAX_HP}")
    col2.metric("🧪 Potions", state.get("num_health_potions", 0))
    col3.metric("🏆 Score", state.get("score", 0))
    col4.metric("⚔️ Attack Power", f"1-{BASE_ATTACK_MAX + (state.get('score', 0) // 3)}")

    # Any message from last action
    msg = state.get("message", "")
    if msg:
        st.info(msg)

    # Early exit/save option while alive
    with st.expander("Save & exit options", expanded=False):
        c1, c2 = st.columns(2)
        if c1.button("💾 Save Score & Exit Now", use_container_width=True):
            _galios_den_finalize_and_save(
                state, pid, supabase, _today_key_utc, _set_best_of_day
            )
            final_score = int(state.get("score", 0))
            state.clear()
            st.success(f"Saved! Final score: {final_score}")
            st.balloons()
            st.rerun()
        if c2.button("🗙 Quit without saving", use_container_width=True):
            state.clear()
            st.warning("Run discarded.")
            st.rerun()

    # Dead → finalization prompt
    if state.get("health", 0) <= 0:
        st.error("💀 You have been defeated!")
        st.markdown(f"**Final Score: {state.get('score', 0)} enemies defeated**")
        if st.button("💾 Save Score & Exit", type="primary"):
            _galios_den_finalize_and_save(
                state, pid, supabase, _today_key_utc, _set_best_of_day
            )
            state.clear()
            st.success("Score saved! Thanks for playing!")
            st.balloons()
            st.rerun()
        return

    # Combat state
    combat_state = state.get("combat_state")
    if combat_state == "victory":
        render_victory_screen(state)
    elif combat_state == "fighting":
        render_combat_screen(state)
    else:
        spawn_new_enemy(state)
        st.rerun()


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
        max_enemy_health = 82 + (2 * score)
        enemy_health = random.randint(1, max(1, max_enemy_health))
        state["message"] = f"👹 **{enemy}** has appeared!"

    state.update(
        {
            "combat_state": "fighting",
            "current_enemy": enemy,
            "enemy_health": int(enemy_health),
            "enemy_max_health": int(enemy_health),
        }
    )


def render_combat_screen(state: Dict[str, Any]) -> None:
    """Render the combat interface."""
    enemy = state.get("current_enemy", "Unknown Enemy")
    enemy_health = int(state.get("enemy_health", 0))
    enemy_max_health = int(state.get("enemy_max_health", 1))

    # Enemy health bar (Streamlit wants 0–100)
    st.progress(
        _percent(enemy_health, enemy_max_health),
        text=f"🐉 {enemy}: {enemy_health}/{enemy_max_health} HP",
    )

    st.markdown("### What will you do?")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⚔️ Attack", type="primary", use_container_width=True):
            handle_attack(state)
            st.rerun()

    with col2:
        potions = int(state.get("num_health_potions", 0))
        btn_label = f"🧪 Drink Potion ({potions})" if potions > 0 else "🧪 No Potions"
        if st.button(btn_label, disabled=(potions <= 0), use_container_width=True):
            handle_heal(state)
            st.rerun()

    with col3:
        if enemy == "Gatekeeper Galio":
            if st.button("🙏 Beg Forgiveness", use_container_width=True):
                handle_beg_forgiveness(state)
                st.rerun()
        else:
            if st.button("🏃 Run Away", use_container_width=True):
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
        state["combat_state"] = None  # game end path
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
        state["score"] = int(state.get("score", 0)) + 1
        state["message"] = (
            f"🎉 {enemy} was defeated! You have {int(state.get('health', 0))} HP left."
        )


def handle_heal(state: Dict[str, Any]) -> None:
    """Drink a potion to heal."""
    potions = int(state.get("num_health_potions", 0))
    if potions <= 0:
        state["message"] = "❌ You are out of health potions!"
        return

    score = int(state.get("score", 0))
    heal_amount = 40 + (19 * (score // 3))
    state["health"] = min(PLAYER_MAX_HP, int(state.get("health", 0)) + heal_amount)
    state["num_health_potions"] = potions - 1
    state["message"] = (
        f"🧪 You drink a health potion, healing {heal_amount}. "
        f"HP: {state.get('health', 0)}/{PLAYER_MAX_HP} • Potions left: {state.get('num_health_potions', 0)}."
    )


def handle_beg_forgiveness(state: Dict[str, Any]) -> None:
    """Begging Gatekeeper Galio for mercy."""
    mercy = random.choice([True, False])
    score = int(state.get("score", 0))

    if mercy:
        state["message"] = (
            f"🙏 Galio shows mercy! You exit the dungeon alive. Final score: {score}"
        )
        state["combat_state"] = None
        state["health"] = 0  # triggers end flow
    else:
        state["message"] = (
            "💀 He smashed you in pieces. Galio has no mercy for interlopers!"
        )
        state["combat_state"] = None
        state["health"] = 0


def handle_run_away(state: Dict[str, Any]) -> None:
    """Run away from the current enemy (no score gain)."""
    enemy = state.get("current_enemy", "")
    if enemy == "One Punch Man":
        state["message"] = "🏃 You ran away from One Punch Man! Wise choice."
    else:
        state["message"] = f"🏃 You ran away from {enemy}!"
    state["combat_state"] = None  # next render spawns new enemy


def render_victory_screen(state: Dict[str, Any]) -> None:
    """Post-victory loot + choices."""
    enemy = state.get("current_enemy", "")
    score = int(state.get("score", 0))

    # Potion drop chance decreases as you get stronger
    drop_chance = max(5, 47 - (2 * (score // 3)))
    if random.randint(1, 100) <= drop_chance:
        state["num_health_potions"] = int(state.get("num_health_potions", 0)) + 1
        state[
            "message"
        ] += f"\n🧪 {enemy} dropped a health potion! Potions: {state.get('num_health_potions', 0)}."

    st.markdown("### What would you like to do now?")
    c1, c2, c3 = st.columns(3)

    if c1.button("⚔️ Continue Fighting", type="primary", use_container_width=True):
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

    if c2.button("🚪 Exit Dungeon (no save)", use_container_width=True):
        final_score = int(state.get("score", 0))
        state["message"] = f"🏁 You leave with {final_score} victory/victories."
        state["health"] = 0  # end without saving
        st.rerun()

    if c3.button("💾 Save Score & Exit", use_container_width=True):
        # This button simply signals the outer loop to present save UI;
        # we emulate defeat flow to reuse the save path.
        state["message"] = f"💾 Final score: {int(state.get('score', 0))}."
        state["health"] = 0
        st.rerun()


# --------------------------- Leaderboard -------------------------------------


def render_daily_leaderboard(today_key: str, supabase, players_df) -> None:
    """Render the daily leaderboard for Galio's Den."""
    st.subheader("🏆 Today's Leaderboard")
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
                df = df.rename(columns={"name": "Player", "score": "Enemies Defeated"})
                df.insert(0, "#", range(1, len(df) + 1))
                st.dataframe(
                    df[["#", "Player", "Enemies Defeated"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No scores yet today. Be the first to enter Galio's Den!")
    except Exception as e:
        st.error(f"Leaderboard failed: {e}")
