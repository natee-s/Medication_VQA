alter table if exists public.reminder_schedules
    add column if not exists trade_name text;
