update public.user_profiles
set language = lower(language)
where language is not null;

update public.user_profiles
set language = 'th'
where language is null;

alter table public.user_profiles
alter column language set default 'th';

alter table public.user_profiles
alter column language set not null;

alter table public.user_profiles
drop constraint if exists user_profiles_language_check;

alter table public.user_profiles
add constraint user_profiles_language_check
check (language in ('th', 'en', 'my', 'lo', 'zh'));
