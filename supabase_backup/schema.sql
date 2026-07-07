-- =============================================================================
-- Schema backup of Supabase project "board-game responses"
-- Project ref: wsbwrytlkonmnzksqrqb (https://wsbwrytlkonmnzksqrqb.supabase.co)
-- Postgres 17.4 | Snapshot taken: 2026-07-07
--
-- Captured from the live database catalogs (pg_indexes, pg_policies,
-- information_schema). The public schema had NO custom functions, views,
-- triggers, or storage buckets at snapshot time.
--
-- Restore into any Postgres: psql -f schema.sql, then -f data.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

CREATE TABLE public.players (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text NOT NULL,
    nickname    text,
    joined_on   timestamptz DEFAULT now(),
    pin_code    text,
    emoji_lock  text,
    bag         jsonb DEFAULT '{}'::jsonb,
    CONSTRAINT players_name_key UNIQUE (name)
);

CREATE TABLE public.games (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name           text NOT NULL,
    bgg_slug       text,
    min_players    integer,
    max_players    integer,
    notes          text,
    mechanics      text[],
    release_year   integer,
    bgg_weight     numeric,
    game_type      text CHECK (game_type = ANY (ARRAY['board'::text, 'digital'::text, 'sport'::text, 'other'::text])),
    supports_ffa   boolean,
    supports_team  boolean,
    supports_coop  boolean,
    supports_solo  boolean,
    CONSTRAINT games_name_key UNIQUE (name)
);

CREATE TABLE public.sessions (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    played_at  timestamptz NOT NULL,
    game_id    uuid REFERENCES public.games (id),
    location   text,
    notes      text
);

CREATE TABLE public.session_players (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  uuid REFERENCES public.sessions (id),
    player_id   uuid REFERENCES public.players (id),
    team        text,
    position    integer,
    points      numeric,
    is_winner   boolean,
    CONSTRAINT session_players_session_id_player_id_key UNIQUE (session_id, player_id)
);

CREATE TABLE public.config (
    key    text PRIMARY KEY,
    value  text
);

CREATE TABLE public.session_requests (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    payload     jsonb NOT NULL,
    created_by  text,
    status      text NOT NULL DEFAULT 'pending'::text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE public.minigame_scores (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    date_key         text NOT NULL,
    player_id        uuid NOT NULL REFERENCES public.players (id),
    score            integer NOT NULL DEFAULT 0,
    rounds_played    integer NOT NULL DEFAULT 0,
    duration_s       integer NOT NULL DEFAULT 60,
    started_at       timestamptz NOT NULL DEFAULT now(),
    finished_at      timestamptz,
    is_best_for_day  boolean NOT NULL DEFAULT false,
    meta             jsonb DEFAULT '{}'::jsonb
);

CREATE TABLE public.minigame_daily_awards (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    date_key     text NOT NULL,
    player_id    uuid NOT NULL REFERENCES public.players (id),
    placement    integer,
    reward_code  text NOT NULL,
    awarded_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE public.survey_responses (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    ts            timestamptz NOT NULL DEFAULT now(),
    "user"        text,
    answers_json  jsonb,
    ranking_json  jsonb
);

COMMENT ON TABLE public.survey_responses IS 'adds sure responses of people';

-- ---------------------------------------------------------------------------
-- Indexes (beyond primary keys / unique constraints)
-- ---------------------------------------------------------------------------

CREATE INDEX idx_sessions_game            ON public.sessions (game_id);
CREATE INDEX idx_session_players_session  ON public.session_players (session_id);
CREATE INDEX idx_session_players_player   ON public.session_players (player_id);
CREATE INDEX idx_minigame_scores_day_best   ON public.minigame_scores (date_key, is_best_for_day);
CREATE INDEX idx_minigame_scores_day_player ON public.minigame_scores (date_key, player_id);
CREATE INDEX idx_minigame_awards_day        ON public.minigame_daily_awards (date_key);

-- ---------------------------------------------------------------------------
-- Row Level Security (state as found in the live project)
-- RLS was ENABLED only on: players, sessions, session_players.
-- All policies were fully permissive (qual/with_check = true) for role public,
-- i.e. they provided no real restriction. games had policies defined but RLS
-- itself was DISABLED. The remaining tables (config, session_requests,
-- minigame_scores, minigame_daily_awards, survey_responses) had RLS disabled
-- and no policies.
-- ---------------------------------------------------------------------------

ALTER TABLE public.players         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.session_players ENABLE ROW LEVEL SECURITY;

CREATE POLICY players_read  ON public.players  FOR SELECT USING (true);
CREATE POLICY players_write ON public.players  FOR ALL    USING (true) WITH CHECK (true);
CREATE POLICY games_read    ON public.games    FOR SELECT USING (true);
CREATE POLICY games_write   ON public.games    FOR ALL    USING (true) WITH CHECK (true);
CREATE POLICY sessions_read  ON public.sessions FOR SELECT USING (true);
CREATE POLICY sessions_write ON public.sessions FOR ALL    USING (true) WITH CHECK (true);
CREATE POLICY session_players_read  ON public.session_players FOR SELECT USING (true);
CREATE POLICY session_players_write ON public.session_players FOR ALL    USING (true) WITH CHECK (true);
