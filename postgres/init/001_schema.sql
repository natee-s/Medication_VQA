create extension if not exists vector;
create extension if not exists pgcrypto;
create extension if not exists pg_trgm;

create table if not exists public."Medication_VQA" (
    source_row_number integer primary key,
    source_item_id text,
    label_name text,
    initial text,
    trade_name text,
    generic_name text,
    indication text,
    dosage_frequency text,
    instruction_time text,
    precaution text,
    rag_text text,
    embedding vector(768)
);

create table if not exists public.user_profiles (
    line_uid text primary key,
    language varchar default 'th',
    default_morning time default '08:30:00',
    default_noon time default '12:30:00',
    default_evening time default '19:30:00',
    default_bedtime time default '21:00:00',
    created_at timestamptz default now()
);

create table if not exists public.reminder_schedules (
    id uuid primary key default gen_random_uuid(),
    line_uid text not null,
    drug_name text not null,
    morning boolean default false,
    noon boolean default false,
    evening boolean default false,
    bedtime boolean default false,
    is_active boolean default true,
    created_at timestamptz default now(),
    trade_name text,
    meal_timing text
);

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

create index if not exists idx_medication_vqa_trade_name
    on public."Medication_VQA" using gin (trade_name gin_trgm_ops);

create index if not exists idx_medication_vqa_generic_name
    on public."Medication_VQA" using gin (generic_name gin_trgm_ops);

create index if not exists idx_reminder_schedules_line_uid_active
    on public.reminder_schedules (line_uid, is_active);

create index if not exists idx_user_medicine_context_expires_at
    on public.user_medicine_context (expires_at);

create or replace function public.match_symptoms(
    query_embedding vector(768),
    match_threshold double precision default 0.4,
    match_count integer default 3
)
returns table (
    source_row_number integer,
    source_item_id text,
    label_name text,
    initial text,
    trade_name text,
    generic_name text,
    indication text,
    dosage_frequency text,
    instruction_time text,
    precaution text,
    rag_text text,
    similarity double precision
)
language sql
stable
as $$
    select
        m.source_row_number,
        m.source_item_id,
        m.label_name,
        m.initial,
        m.trade_name,
        m.generic_name,
        m.indication,
        m.dosage_frequency,
        m.instruction_time,
        m.precaution,
        m.rag_text,
        1 - (m.embedding <=> query_embedding) as similarity
    from public."Medication_VQA" m
    where m.embedding is not null
      and 1 - (m.embedding <=> query_embedding) >= match_threshold
    order by m.embedding <=> query_embedding
    limit match_count;
$$;
