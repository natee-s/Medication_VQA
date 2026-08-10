create extension if not exists pg_trgm;
create extension if not exists pgcrypto;

create or replace function public.normalize_drug_identity_text(input text)
returns text
language sql
immutable
as $$
    select upper(regexp_replace(coalesce(input, ''), '[^[:alnum:]ก-๙]+', '', 'g'));
$$;

create table if not exists public.drug_identity (
    id uuid primary key default gen_random_uuid(),
    source_row_number integer unique,
    canonical_name text not null,
    trade_name text,
    generic_name text,
    source_name text not null default 'Medication_VQA',
    source_priority integer not null default 100,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.drug_aliases (
    id uuid primary key default gen_random_uuid(),
    drug_identity_id uuid not null references public.drug_identity(id) on delete cascade,
    alias_name text not null,
    alias_type text not null default 'alias',
    normalized_alias text not null,
    source_name text not null default 'Medication_VQA',
    created_at timestamptz not null default now(),
    unique (drug_identity_id, normalized_alias, alias_type, source_name)
);

create table if not exists public.drug_identity_match_logs (
    id uuid primary key default gen_random_uuid(),
    line_uid text,
    raw_query text,
    normalized_query text,
    matched_alias text,
    matched_drug_identity_id uuid references public.drug_identity(id) on delete set null,
    match_score double precision,
    created_at timestamptz not null default now()
);

create index if not exists idx_drug_identity_source_row_number
    on public.drug_identity (source_row_number);

create index if not exists idx_drug_aliases_normalized_alias_trgm
    on public.drug_aliases using gin (normalized_alias gin_trgm_ops);

create index if not exists idx_drug_aliases_drug_identity_id
    on public.drug_aliases (drug_identity_id);

insert into public.drug_identity (
    source_row_number,
    canonical_name,
    trade_name,
    generic_name,
    source_name,
    source_priority
)
select
    m.source_row_number,
    coalesce(nullif(trim(m.generic_name), ''), nullif(trim(m.trade_name), ''), nullif(trim(m.label_name), ''), 'UNKNOWN'),
    nullif(trim(m.trade_name), ''),
    nullif(trim(m.generic_name), ''),
    'Medication_VQA',
    10
from public."Medication_VQA" m
on conflict (source_row_number) do update set
    canonical_name = excluded.canonical_name,
    trade_name = excluded.trade_name,
    generic_name = excluded.generic_name,
    source_priority = excluded.source_priority,
    updated_at = now();

with alias_values as (
    select
        d.id as drug_identity_id,
        v.alias_name,
        v.alias_type
    from public.drug_identity d
    cross join lateral (
        values
            (d.trade_name, 'trade_name'),
            (d.generic_name, 'generic_name'),
            (d.canonical_name, 'canonical_name')
    ) as v(alias_name, alias_type)
    where d.source_name = 'Medication_VQA'

    union all

    select
        d.id as drug_identity_id,
        trim(component) as alias_name,
        'generic_component' as alias_type
    from public.drug_identity d
    cross join lateral regexp_split_to_table(coalesce(d.generic_name, ''), '\s*(,|\+|/)\s*') as component
    where d.source_name = 'Medication_VQA'
)
insert into public.drug_aliases (
    drug_identity_id,
    alias_name,
    alias_type,
    normalized_alias,
    source_name
)
select
    drug_identity_id,
    trim(alias_name),
    alias_type,
    public.normalize_drug_identity_text(alias_name),
    'Medication_VQA'
from alias_values
where alias_name is not null
  and trim(alias_name) <> ''
  and lower(trim(alias_name)) not in ('ไม่ระบุ', 'unknown', 'none', 'null')
  and length(public.normalize_drug_identity_text(alias_name)) >= 3
on conflict (drug_identity_id, normalized_alias, alias_type, source_name) do nothing;
