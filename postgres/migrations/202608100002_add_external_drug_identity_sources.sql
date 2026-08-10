alter table public.drug_identity
    add column if not exists source_key text,
    add column if not exists tmt_code text,
    add column if not exists fda_registration_no text,
    add column if not exists active_ingredient text,
    add column if not exists strength text,
    add column if not exists dosage_form text,
    add column if not exists external_metadata jsonb not null default '{}'::jsonb;

do $$
begin
    alter table public.drug_identity
        add constraint drug_identity_source_name_key_unique unique (source_name, source_key);
exception
    when duplicate_object then null;
end $$;

create table if not exists public.drug_identity_sources (
    id uuid primary key default gen_random_uuid(),
    drug_identity_id uuid not null references public.drug_identity(id) on delete cascade,
    source_name text not null,
    source_key text not null,
    source_version text,
    source_url text,
    raw_payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (source_name, source_key)
);

create table if not exists public.staging_drug_identity_imports (
    id uuid primary key default gen_random_uuid(),
    import_batch_id text not null,
    source_name text not null,
    source_version text,
    source_url text,
    source_key text not null,
    tmt_code text,
    fda_registration_no text,
    trade_name text,
    generic_name text,
    active_ingredient text,
    strength text,
    dosage_form text,
    manufacturer text,
    registration_status text,
    raw_payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_drug_identity_tmt_code
    on public.drug_identity (tmt_code);

create index if not exists idx_drug_identity_fda_registration_no
    on public.drug_identity (fda_registration_no);

create index if not exists idx_drug_identity_source_name_key
    on public.drug_identity (source_name, source_key);

create index if not exists idx_drug_identity_sources_drug_identity_id
    on public.drug_identity_sources (drug_identity_id);

create index if not exists idx_staging_drug_identity_imports_batch
    on public.staging_drug_identity_imports (import_batch_id);

create or replace function public.sync_staging_drug_identity_imports(batch_id text)
returns table (
    staged_rows integer,
    identity_rows integer,
    alias_rows integer
)
language plpgsql
as $$
begin
    insert into public.drug_identity (
        source_key,
        canonical_name,
        trade_name,
        generic_name,
        active_ingredient,
        strength,
        dosage_form,
        tmt_code,
        fda_registration_no,
        source_name,
        source_priority,
        external_metadata
    )
    select
        s.source_key,
        coalesce(
            nullif(trim(s.generic_name), ''),
            nullif(trim(s.trade_name), ''),
            nullif(trim(s.active_ingredient), ''),
            nullif(trim(s.tmt_code), ''),
            nullif(trim(s.fda_registration_no), ''),
            s.source_key
        ) as canonical_name,
        nullif(trim(s.trade_name), ''),
        nullif(trim(s.generic_name), ''),
        nullif(trim(s.active_ingredient), ''),
        nullif(trim(s.strength), ''),
        nullif(trim(s.dosage_form), ''),
        nullif(trim(s.tmt_code), ''),
        nullif(trim(s.fda_registration_no), ''),
        s.source_name,
        case
            when upper(s.source_name) = 'TMT' then 20
            when upper(s.source_name) in ('THAIFDA', 'THAI_FDA', 'FDA') then 30
            else 80
        end,
        jsonb_strip_nulls(jsonb_build_object(
            'manufacturer', nullif(trim(s.manufacturer), ''),
            'registration_status', nullif(trim(s.registration_status), ''),
            'source_version', nullif(trim(s.source_version), ''),
            'source_url', nullif(trim(s.source_url), '')
        ))
    from public.staging_drug_identity_imports s
    where s.import_batch_id = batch_id
    on conflict (source_name, source_key) do update set
        canonical_name = excluded.canonical_name,
        trade_name = excluded.trade_name,
        generic_name = excluded.generic_name,
        active_ingredient = excluded.active_ingredient,
        strength = excluded.strength,
        dosage_form = excluded.dosage_form,
        tmt_code = excluded.tmt_code,
        fda_registration_no = excluded.fda_registration_no,
        source_priority = excluded.source_priority,
        external_metadata = excluded.external_metadata,
        updated_at = now();

    insert into public.drug_identity_sources (
        drug_identity_id,
        source_name,
        source_key,
        source_version,
        source_url,
        raw_payload
    )
    select
        d.id,
        s.source_name,
        s.source_key,
        nullif(trim(s.source_version), ''),
        nullif(trim(s.source_url), ''),
        s.raw_payload
    from public.staging_drug_identity_imports s
    join public.drug_identity d
      on d.source_name = s.source_name
     and d.source_key = s.source_key
    where s.import_batch_id = batch_id
    on conflict (source_name, source_key) do update set
        drug_identity_id = excluded.drug_identity_id,
        source_version = excluded.source_version,
        source_url = excluded.source_url,
        raw_payload = excluded.raw_payload,
        updated_at = now();

    with alias_values as (
        select
            d.id as drug_identity_id,
            s.source_name,
            v.alias_name,
            v.alias_type
        from public.staging_drug_identity_imports s
        join public.drug_identity d
          on d.source_name = s.source_name
         and d.source_key = s.source_key
        cross join lateral (
            values
                (s.trade_name, 'trade_name'),
                (s.generic_name, 'generic_name'),
                (s.active_ingredient, 'active_ingredient'),
                (s.tmt_code, 'tmt_code'),
                (s.fda_registration_no, 'fda_registration_no')
        ) as v(alias_name, alias_type)
        where s.import_batch_id = batch_id

        union all

        select
            d.id as drug_identity_id,
            s.source_name,
            trim(component) as alias_name,
            'active_component' as alias_type
        from public.staging_drug_identity_imports s
        join public.drug_identity d
          on d.source_name = s.source_name
         and d.source_key = s.source_key
        cross join lateral regexp_split_to_table(
            coalesce(s.generic_name, s.active_ingredient, ''),
            '\s*(,|\+|/)\s*'
        ) as component
        where s.import_batch_id = batch_id
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
        source_name
    from alias_values
    where alias_name is not null
      and trim(alias_name) <> ''
      and lower(trim(alias_name)) not in ('ไม่ระบุ', 'unknown', 'none', 'null', '-')
      and length(public.normalize_drug_identity_text(alias_name)) >= 3
    on conflict (drug_identity_id, normalized_alias, alias_type, source_name) do nothing;

    return query
    select
        count(distinct s.id)::integer as staged_rows,
        count(distinct d.id)::integer as identity_rows,
        count(distinct a.id)::integer as alias_rows
    from public.staging_drug_identity_imports s
    left join public.drug_identity d
      on d.source_name = s.source_name
     and d.source_key = s.source_key
    left join public.drug_aliases a
      on a.drug_identity_id = d.id
     and a.source_name = s.source_name
    where s.import_batch_id = batch_id;
end;
$$;
