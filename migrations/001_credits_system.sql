-- =============================================================================
-- Credits System Migration
-- Creates tables for user credits, transactions, and usage logging.
-- Run this in Supabase SQL Editor or via migration tool.
-- =============================================================================

-- 1. Enum types
-- -----------------------------------------------------------------------------

CREATE TYPE credit_transaction_type AS ENUM ('purchase', 'usage', 'bonus', 'refund');
CREATE TYPE credit_model_type AS ENUM ('stt', 'llm', 'tts', 'realtime');


-- 2. user_credits - Current credit balance per user
-- -----------------------------------------------------------------------------

CREATE TABLE user_credits (
    user_id    uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    balance    bigint NOT NULL DEFAULT 0,
    lifetime_purchased bigint NOT NULL DEFAULT 0,
    lifetime_used      bigint NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE  user_credits IS 'Stores the current credit balance for each user. Balance is in micro-credits (1 credit = 1000 micro-credits).';
COMMENT ON COLUMN user_credits.balance IS 'Current balance in micro-credits. 1 displayed credit = 1000 micro-credits.';


-- 3. credit_transactions - Immutable ledger of all credit changes
-- -----------------------------------------------------------------------------

CREATE TABLE credit_transactions (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    type          credit_transaction_type NOT NULL,
    amount        bigint NOT NULL,          -- positive = credit, negative = debit
    balance_after bigint NOT NULL,
    description   text,
    metadata      jsonb DEFAULT '{}',       -- stripe session id, model info, etc.
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_credit_transactions_user ON credit_transactions(user_id, created_at DESC);

COMMENT ON TABLE credit_transactions IS 'Immutable ledger of every credit balance change.';


-- 4. credit_usage_logs - Granular per-session AI usage
-- -----------------------------------------------------------------------------

CREATE TABLE credit_usage_logs (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id      text NOT NULL,                -- LiveKit room name
    model_type      credit_model_type NOT NULL,
    model_id        text NOT NULL,                -- e.g. 'openai/gpt-4o-mini'
    units_used      double precision NOT NULL,    -- tokens (LLM), minutes (STT), characters (TTS)
    cost_usd        double precision NOT NULL,    -- raw provider cost
    credits_charged bigint NOT NULL,              -- micro-credits deducted
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_credit_usage_logs_user    ON credit_usage_logs(user_id, created_at DESC);
CREATE INDEX idx_credit_usage_logs_session ON credit_usage_logs(session_id);

COMMENT ON TABLE credit_usage_logs IS 'Detailed per-model usage records for each AI session.';


-- 5. Row-Level Security (RLS)
-- -----------------------------------------------------------------------------

ALTER TABLE user_credits ENABLE ROW LEVEL SECURITY;
ALTER TABLE credit_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE credit_usage_logs ENABLE ROW LEVEL SECURITY;

-- Users can read their own credit balance
CREATE POLICY "Users can view own credits"
    ON user_credits FOR SELECT
    USING (auth.uid() = user_id);

-- Users can read their own transactions
CREATE POLICY "Users can view own transactions"
    ON credit_transactions FOR SELECT
    USING (auth.uid() = user_id);

-- Users can read their own usage logs
CREATE POLICY "Users can view own usage"
    ON credit_usage_logs FOR SELECT
    USING (auth.uid() = user_id);

-- Service role (backend) can do everything (implicit via supabase service key)
-- No explicit INSERT/UPDATE/DELETE policies for users -- all writes go through the API.


-- 6. Welcome bonus trigger
-- When a new row is inserted into auth.users, auto-create a credit row with
-- 500 credits (= 500,000 micro-credits) and a bonus transaction.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION handle_new_user_credits()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    welcome_bonus bigint := 500000;  -- 500 credits in micro-credits
BEGIN
    INSERT INTO user_credits (user_id, balance, lifetime_purchased, lifetime_used)
    VALUES (NEW.id, welcome_bonus, 0, 0);

    INSERT INTO credit_transactions (user_id, type, amount, balance_after, description, metadata)
    VALUES (
        NEW.id,
        'bonus',
        welcome_bonus,
        welcome_bonus,
        'Welcome bonus - 500 free energy',
        '{"reason": "welcome_bonus"}'::jsonb
    );

    RETURN NEW;
END;
$$;

CREATE TRIGGER on_auth_user_created_credits
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION handle_new_user_credits();


-- 7. Helper function: atomic credit deduction (used by API)
-- Returns the new balance, or raises an exception if insufficient funds.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION deduct_credits(
    p_user_id uuid,
    p_amount  bigint,
    p_description text DEFAULT NULL,
    p_metadata jsonb DEFAULT '{}'
)
RETURNS bigint
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    new_balance bigint;
BEGIN
    UPDATE user_credits
    SET balance     = balance - p_amount,
        lifetime_used = lifetime_used + p_amount,
        updated_at  = now()
    WHERE user_id   = p_user_id
      AND balance  >= p_amount
    RETURNING balance INTO new_balance;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Insufficient credits';
    END IF;

    INSERT INTO credit_transactions (user_id, type, amount, balance_after, description, metadata)
    VALUES (p_user_id, 'usage', -p_amount, new_balance, p_description, p_metadata);

    RETURN new_balance;
END;
$$;


-- 8. Helper function: atomic credit addition (used by webhook)
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION add_credits(
    p_user_id     uuid,
    p_amount      bigint,
    p_type        credit_transaction_type DEFAULT 'purchase',
    p_description text DEFAULT NULL,
    p_metadata    jsonb DEFAULT '{}'
)
RETURNS bigint
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    new_balance bigint;
BEGIN
    -- Upsert: create row if user doesn't have one yet (edge case)
    INSERT INTO user_credits (user_id, balance, lifetime_purchased, lifetime_used)
    VALUES (p_user_id, p_amount, p_amount, 0)
    ON CONFLICT (user_id) DO UPDATE
    SET balance            = user_credits.balance + p_amount,
        lifetime_purchased = user_credits.lifetime_purchased + p_amount,
        updated_at         = now()
    RETURNING balance INTO new_balance;

    INSERT INTO credit_transactions (user_id, type, amount, balance_after, description, metadata)
    VALUES (p_user_id, p_type, p_amount, new_balance, p_description, p_metadata);

    RETURN new_balance;
END;
$$;
