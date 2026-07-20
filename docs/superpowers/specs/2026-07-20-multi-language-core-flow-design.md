# Multi-Language Core Flow Design

Date: 2026-07-20
Project: Medication_VQA

## Goal

Add a core multi-language flow for the LINE medication assistant so users can choose Thai, English, Burmese, Lao, or Simplified Chinese once, store that preference in Supabase, change it later from the Rich Menu, and receive core system messages plus AI medication responses in the selected language.

## Confirmed Decisions

- Add a Supabase migration file for `preferred_language`.
- Default language for existing and new users is Thai: `th`.
- First implementation scope is core flow only, not every static string in `main.py`.
- Rich Menu language change action sends text to the webhook.
- Chinese language support uses Simplified Chinese with language code `zh`.
- The current backend stack is Python/FastAPI, so the implementation will use Python helpers rather than Node.js.

## Supported Languages

| Code | Display Name | AI Prompt Name |
| --- | --- | --- |
| `th` | ไทย | Thai |
| `en` | English | English |
| `my` | မြန်မာ | Burmese |
| `lo` | ລາວ | Lao |
| `zh` | 中文 | Simplified Chinese |

## Data Layer

Create a SQL migration under `supabase/migrations/` that adds `preferred_language` to `user_profiles`.

The column should:

- be `text`
- be `not null`
- default to `'th'`
- allow only `th`, `en`, `my`, `lo`, and `zh`

The migration should be safe for existing users by backfilling null values to `th`.

Expected shape:

```sql
alter table public.user_profiles
add column if not exists preferred_language text not null default 'th';

update public.user_profiles
set preferred_language = 'th'
where preferred_language is null;

alter table public.user_profiles
drop constraint if exists user_profiles_preferred_language_check;

alter table public.user_profiles
add constraint user_profiles_preferred_language_check
check (preferred_language in ('th', 'en', 'my', 'lo', 'zh'));
```

## Service Layer

Add small helpers in `services/supabase_service.py` so `main.py` does not repeat profile and language queries.

Initial helper responsibilities:

- `ensure_user_profile(line_uid: str) -> dict | None`
  - Find a user profile by `line_uid`.
  - Create one with `preferred_language = "th"` if missing.
  - Return the profile data when possible.
- `get_user_language(line_uid: str) -> str`
  - Return the user profile language.
  - Fall back to `th` if Supabase is unavailable, the profile is missing, or the value is unsupported.
- `set_user_language(line_uid: str, language: str) -> bool`
  - Validate the language code.
  - Ensure the profile exists.
  - Update `preferred_language`.
  - Return whether the update succeeded.

These helpers should preserve the existing behavior where missing Supabase credentials warn and leave `supabase = None`.

## Locale Layer

Expand the current locale data into a nested dictionary by language. Keep the first pass focused on core messages only.

Core keys:

- `language_picker_alt`
- `language_picker_title`
- `language_picker_subtitle`
- `language_saved`
- `change_language_command`
- `unsupported_message_type`
- `generic_processing_error`
- `supabase_not_connected`
- `ocr_rotated_image`
- `ocr_unclear_drug_name`
- `ocr_no_database_match`
- `ai_format_error`

The app should use a helper such as:

```python
def t(lang: str, key: str, **kwargs) -> str:
    ...
```

The helper should fall back to Thai when a language or key is missing.

## Presentation Layer

### Onboarding

Add LINE `FollowEvent` handling. When a user adds the LINE OA:

1. Ensure a profile exists with default `preferred_language = "th"`.
2. Reply with a Flex Message language picker.

The picker should include five buttons:

- 🇹🇭 ไทย
- 🇬🇧 English
- 🇲🇲 မြန်မာ
- 🇱🇦 ລາວ
- 🇨🇳 中文

Each button sends postback data:

```text
action=set_language&lang=th
action=set_language&lang=en
action=set_language&lang=my
action=set_language&lang=lo
action=set_language&lang=zh
```

### Settings / Rich Menu

The Rich Menu will send text. `handle_text_message` should intercept these text commands before AI routing:

- `เปลี่ยนภาษา`
- `Change Language`
- `เปลี่ยนภาษา / Change Language`

When matched, it should reply with the same language picker. The picker labels can remain multilingual so users can understand them even before switching language.

### Language Selection

`handle_postback` should support `action=set_language`.

When a valid language is selected:

1. Save `preferred_language` to Supabase.
2. Reply with a localized confirmation message in the selected language.

When an invalid language is sent:

1. Do not update the database.
2. Reply with a generic Thai fallback error.

## Application And AI Layer

Before handling user-visible responses, retrieve the user language:

```python
user_language = get_user_language(user_id)
```

For dynamic AI content, include the selected language in the prompt. The prompt should use the AI prompt name, not just the language code.

Example:

```python
language_name = get_ai_language_name(user_language)

final_prompt = f"""
...
You must answer the user only in: {language_name}.
...
"""
```

Apply language control to:

- OCR/image medication extraction response prompt where user-visible text is generated.
- RAG medication advice prompt in text queries.

Do not change database search behavior, table names, RPC names, embedding model, or prompt output JSON structure unless required to preserve JSON parsing.

## Core Static Text Scope

This first pass localizes only core control flow messages and language selection UX.

In scope:

- language picker
- language saved confirmation
- change-language command interception
- unsupported message type
- generic processing errors used by core paths
- key OCR errors
- AI prompt language instruction

Out of scope for this pass:

- Full translation of all reminder Flex cards.
- Full translation of all medication detail Flex labels.
- Full translation of every debug log.
- LINE Rich Menu creation through LINE API.
- Supabase dashboard changes beyond the SQL migration file.

## Data Flow

```text
LINE FollowEvent
  -> ensure_user_profile(default th)
  -> reply language picker Flex

Rich Menu text command
  -> handle_text_message intercepts command
  -> reply language picker Flex

Language picker postback
  -> handle_postback action=set_language
  -> set_user_language
  -> localized confirmation

Text/Image medication request
  -> get_user_language
  -> localized static core messages
  -> AI prompt includes selected language name
  -> AI response rendered in selected language
```

## Error Handling

- If Supabase is unavailable, default language is `th`.
- If the profile is missing during a normal message, create it with `th` where safe.
- If an unsupported language code appears in the database, treat it as `th`.
- If locale JSON is missing a key, fall back to Thai for that key.
- If AI returns malformed JSON, use the localized `ai_format_error` where the current code path supports it.

## Testing And Verification

Implementation should verify:

- SQL migration exists and contains `preferred_language`.
- `services/supabase_service.py`, `main.py`, and locale JSON compile/validate.
- Locale helper returns Thai fallback for unknown language and missing key.
- Language picker Flex contains five postback buttons with expected language codes.
- Rich Menu text commands return the language picker.
- `action=set_language&lang=en` calls the Supabase update helper and returns English confirmation.
- AI prompts include the selected language name.

## Rollout Notes

Before deploying code that reads or writes `preferred_language`, run the SQL migration in Supabase.

Because the repository currently ignores new files with `*` in `.gitignore`, new migration, locale, and service files must be force-added if committing with git.
