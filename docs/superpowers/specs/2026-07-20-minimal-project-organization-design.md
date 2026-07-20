# Minimal Project Organization Design

Date: 2026-07-20
Project: Medication_VQA

## Goal

Reorganize the project with minimal behavior change so the LINE webhook can keep running while database code, translation text, and test assets are separated into clearer folders.

## Current State

- `main.py` should be the LINE webhook entry point, but the working tree currently contains PDPA/image preprocessing test code in `main.py`.
- Git still has the previous webhook implementation in `main.py`, so the webhook can be restored from the repository version.
- `embed_data.py` currently owns its own Supabase and Gemini setup for embedding generation.
- `services/supabase_service.py`, `locales/i18n.json`, and `tests/` already exist, but the service and locale files are empty.
- `tests/` already contains `test_pdpa.py`, `test_label.jpg`, and `debug_thresh.jpg`.

## Recommended Approach

Use a minimal refactor. Restore the original webhook behavior first, then extract only the Supabase connection and repeated database access helpers into `services/supabase_service.py`.

Avoid moving every large block of LINE Flex message logic or Gemini prompt text during this pass. Those can be refactored later after the application is stable again.

## Target Structure

```text
MEDICATION_VQA/
  main.py
  embed_data.py
  services/
    __init__.py
    supabase_service.py
  locales/
    i18n.json
  tests/
    test_pdpa.py
    test_label.jpg
    debug_thresh.jpg
```

## Component Design

### `main.py`

`main.py` remains the FastAPI and LINE webhook entry point. It should continue to contain route registration, LINE event handlers, and high-level message flow.

The file should no longer directly create the Supabase client. Instead, it imports the shared client or helper functions from `services.supabase_service`.

### `services/supabase_service.py`

This file owns Supabase setup and database operations used by the webhook and scripts.

Initial responsibilities:

- Read `SUPABASE_URL` and `SUPABASE_KEY` from environment variables.
- Create a Supabase client when credentials are present.
- Provide a shared `supabase` object or `get_supabase_client()` helper.
- Provide small helpers only where they reduce duplication and do not change behavior.

The first pass should keep SQL/table names and query conditions equivalent to the original `main.py`.

### `embed_data.py`

`embed_data.py` remains the embedding-generation script. It should reuse the Supabase setup from `services.supabase_service` while keeping Gemini embedding logic local to the script.

### `locales/i18n.json`

This file starts as a small text dictionary for stable, reusable user-facing messages.

Initial candidates:

- unsupported message type
- generic system error
- missing database configuration warning
- image quality warnings if they are simple and reused

Dynamic Flex message bodies and long AI prompts stay in Python for now to reduce risk.

### `tests/`

Test and debug assets stay under `tests/`. Test scripts should use paths relative to their own folder so they can be run from the project root without missing images.

## Data Flow

```text
LINE -> FastAPI /webhook -> main.py handler
                       -> services.supabase_service for database reads/writes
                       -> Gemini client in main.py or embed_data.py
                       -> LINE reply/push message
```

## Error Handling

- Missing Gemini API key should continue to fail clearly, as in the original webhook.
- Missing Supabase credentials should not crash the entire app if the previous behavior only warned. The refactor should preserve the original behavior.
- Database query errors should still be caught at the same call sites in `main.py`.

## Testing And Verification

After implementation:

- Run a Python syntax check for `main.py`, `embed_data.py`, `services/supabase_service.py`, and test scripts.
- Import `main.py` only if required environment variables are available or if import-time behavior supports missing secrets safely.
- Run the PDPA test script from the project root and confirm it reads assets from `tests/`.
- Optionally start the FastAPI app locally if credentials are configured.

## Out Of Scope

- Full service-layer rewrite for LINE, Gemini, reminders, image processing, or Flex messages.
- Changing database schema, table names, RPC names, or query behavior.
- Rewriting prompts or chatbot behavior.
- Adding full pytest coverage.
