create table if not exists public.user_medicine_context (
    line_uid text primary key references public.user_profiles(line_uid) on delete cascade,
    primary_drug_id text,
    trade_name text,
    generic_name text,
    indication text,
    dosage text,
    instruction text,
    warning text,
    raw_context_json jsonb default '{}'::jsonb,
    updated_at timestamptz default now(),
    expires_at timestamptz not null
);

create index if not exists idx_user_medicine_context_expires_at
    on public.user_medicine_context (expires_at);
