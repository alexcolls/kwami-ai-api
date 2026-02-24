-- =============================================================================
-- User Kwamis (workspaces) - One row per kwami; config holds avatar/voice/scene/theme.
-- Run after 001_credits_system.sql.
-- =============================================================================

CREATE TABLE user_kwamis (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name       text NOT NULL DEFAULT 'Kwami',
    emoji      text NOT NULL DEFAULT '🌸',
    colors     jsonb NOT NULL DEFAULT '{"x":"#00d9ff","y":"#a855f7","z":"#22c55e"}'::jsonb,
    config     jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_user_kwamis_user ON user_kwamis(user_id);
CREATE INDEX idx_user_kwamis_updated ON user_kwamis(user_id, updated_at DESC);

COMMENT ON TABLE user_kwamis IS 'Per-user kwami workspaces; config stores avatar, voice, scene, theme snapshots.';

-- RLS: users can only access their own kwamis
ALTER TABLE user_kwamis ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own kwamis"
    ON user_kwamis FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own kwamis"
    ON user_kwamis FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own kwamis"
    ON user_kwamis FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own kwamis"
    ON user_kwamis FOR DELETE
    USING (auth.uid() = user_id);

-- Keep updated_at in sync
CREATE OR REPLACE FUNCTION set_user_kwamis_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TRIGGER user_kwamis_updated_at
    BEFORE UPDATE ON user_kwamis
    FOR EACH ROW
    EXECUTE PROCEDURE set_user_kwamis_updated_at();
