-- =============================================================
-- CropGuard AI — Supabase Schema
-- =============================================================
-- Run this in the Supabase SQL Editor to set up your database.
-- Dashboard → SQL Editor → New Query → Paste & Run

-- Enable Row Level Security on all tables
-- This ensures users can only access their own data.

-- ─── Scans Table ────────────────────────────────────────────
-- Stores disease scan results from the camera
CREATE TABLE IF NOT EXISTS scans (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    detections JSONB DEFAULT '[]'::jsonb,          -- Array of detection objects
    spray_decision TEXT DEFAULT 'NO_SPRAY',         -- SPRAY, NO_SPRAY, WAIT_FOR_BEES, etc.
    severity FLOAT DEFAULT 0.0,                     -- Overall severity score (0-1)
    image_url TEXT,                                  -- Optional: stored image URL
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

-- Index for fast user lookups
CREATE INDEX IF NOT EXISTS idx_scans_user_id ON scans(user_id);
CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at DESC);

-- RLS: Users can only see and insert their own scans
ALTER TABLE scans ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own scans"
    ON scans FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own scans"
    ON scans FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own scans"
    ON scans FOR DELETE
    USING (auth.uid() = user_id);

-- ─── Subscriptions Table (for future monetization) ──────────
CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
    plan TEXT DEFAULT 'free',                        -- free, pro
    status TEXT DEFAULT 'active',                    -- active, cancelled, expired
    stripe_customer_id TEXT,                         -- For Stripe integration later
    stripe_subscription_id TEXT,
    current_period_start TIMESTAMPTZ,
    current_period_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions(user_id);

ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own subscription"
    ON subscriptions FOR SELECT
    USING (auth.uid() = user_id);
