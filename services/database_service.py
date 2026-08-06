import os
from typing import Any

from services.supabase_service import SUPABASE_URL, supabase


SUPPORTED_LANGUAGES = ("th", "en", "my", "lo", "zh")
DEFAULT_LANGUAGE = "th"
DB_BACKEND = os.environ.get("DB_BACKEND", "supabase").strip().lower()
DATABASE_URL = os.environ.get("DATABASE_URL")

_postgres_conn = None


def normalize_language(language: str | None) -> str:
    if language in SUPPORTED_LANGUAGES:
        return language
    return DEFAULT_LANGUAGE


def _use_postgres() -> bool:
    return DB_BACKEND == "postgres"


def _get_postgres_conn():
    global _postgres_conn
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required when DB_BACKEND=postgres")

    if _postgres_conn is None or _postgres_conn.closed:
        import psycopg
        from psycopg.rows import dict_row

        _postgres_conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        _postgres_conn.autocommit = True
    return _postgres_conn


def is_database_available() -> bool:
    if _use_postgres():
        return bool(DATABASE_URL)
    return bool(supabase)


def database_status() -> dict[str, Any]:
    return {
        "backend": DB_BACKEND,
        "supabase_url": SUPABASE_URL,
        "postgres_configured": bool(DATABASE_URL),
    }


def _fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict]:
    conn = _get_postgres_conn()
    with conn.cursor() as cur:
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def _fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict | None:
    conn = _get_postgres_conn()
    with conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None


def _execute(query: str, params: tuple[Any, ...] = ()) -> None:
    conn = _get_postgres_conn()
    with conn.cursor() as cur:
        cur.execute(query, params)


def _medication_search_like_query() -> str:
    return """
        select *
        from public."Medication_VQA"
        where generic_name ilike %(term)s
           or trade_name ilike %(term)s
    """


def search_medication_rows(drug_name: str) -> list[dict]:
    if not drug_name or not is_database_available():
        return []

    if not _use_postgres():
        search_query = f"generic_name.ilike.%{drug_name}%,trade_name.ilike.%{drug_name}%"
        response = supabase.table("Medication_VQA").select("*").or_(search_query).execute()
        return response.data or []

    return _fetch_all(_medication_search_like_query(), {"term": f"%{drug_name}%"})


def fetch_all_medication_rows() -> list[dict]:
    if not is_database_available():
        return []

    if not _use_postgres():
        response = supabase.table("Medication_VQA").select("*").execute()
        return response.data or []

    return _fetch_all('select * from public."Medication_VQA"')


def search_medication_by_generic_name(drug_name: str) -> list[dict]:
    if not drug_name or not is_database_available():
        return []

    if not _use_postgres():
        response = (
            supabase.table("Medication_VQA")
            .select("*")
            .ilike("generic_name", f"%{drug_name}%")
            .execute()
        )
        return response.data or []

    return _fetch_all(
        'select * from public."Medication_VQA" where generic_name ilike %s',
        (f"%{drug_name}%",),
    )


def match_symptoms(query_embedding: list[float], match_threshold: float = 0.4, match_count: int = 3) -> list[dict]:
    if not is_database_available():
        return []

    if not _use_postgres():
        db_res = supabase.rpc(
            "match_symptoms",
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
            },
        ).execute()
        return db_res.data or []

    vector_literal = "[" + ",".join(str(float(value)) for value in query_embedding) + "]"
    return _fetch_all(
        "select * from public.match_symptoms(%s::vector, %s, %s)",
        (vector_literal, match_threshold, match_count),
    )


def ensure_user_profile(line_uid: str) -> dict | None:
    if not line_uid or not is_database_available():
        return None

    if not _use_postgres():
        existing = supabase.table("user_profiles").select("*").eq("line_uid", line_uid).execute()
        if existing.data:
            return existing.data[0]

        payload = {"line_uid": line_uid, "language": DEFAULT_LANGUAGE}
        created = supabase.table("user_profiles").insert(payload).execute()
        if created.data:
            return created.data[0]
        return payload

    _execute(
        """
        insert into public.user_profiles (line_uid, language)
        values (%s, %s)
        on conflict (line_uid) do nothing
        """,
        (line_uid, DEFAULT_LANGUAGE),
    )
    return _fetch_one("select * from public.user_profiles where line_uid = %s", (line_uid,))


def get_user_language(line_uid: str) -> str:
    profile = ensure_user_profile(line_uid)
    if not profile:
        return DEFAULT_LANGUAGE
    return normalize_language(profile.get("language"))


def set_user_language(line_uid: str, language: str) -> bool:
    normalized = normalize_language(language)
    if normalized != language or not is_database_available():
        return False

    ensure_user_profile(line_uid)
    if not _use_postgres():
        supabase.table("user_profiles").update({"language": normalized}).eq("line_uid", line_uid).execute()
    else:
        _execute(
            "update public.user_profiles set language = %s where line_uid = %s",
            (normalized, line_uid),
        )
    return True


def get_profiles_for_reminder_check(current_time_db: str, future_30_db: str, current_time_str: str) -> list[dict]:
    if not is_database_available():
        return []

    or_conditions = [
        f"default_morning.eq.{current_time_db}", f"default_morning.eq.{future_30_db}",
        f"default_noon.eq.{current_time_db}", f"default_noon.eq.{future_30_db}",
        f"default_evening.eq.{current_time_db}", f"default_evening.eq.{future_30_db}",
        f"default_bedtime.eq.{current_time_db}", f"default_bedtime.eq.{future_30_db}",
    ]

    if current_time_str in ("08:00", "07:30"):
        or_conditions.append("default_morning.is.null")
    if current_time_str in ("12:00", "11:30"):
        or_conditions.append("default_noon.is.null")
    if current_time_str in ("18:00", "17:30"):
        or_conditions.append("default_evening.is.null")
    if current_time_str in ("21:00", "20:30"):
        or_conditions.append("default_bedtime.is.null")

    if not _use_postgres():
        users_res = supabase.table("user_profiles").select("*").or_(",".join(or_conditions)).execute()
        return users_res.data or []

    null_checks = []
    if current_time_str in ("08:00", "07:30"):
        null_checks.append("default_morning is null")
    if current_time_str in ("12:00", "11:30"):
        null_checks.append("default_noon is null")
    if current_time_str in ("18:00", "17:30"):
        null_checks.append("default_evening is null")
    if current_time_str in ("21:00", "20:30"):
        null_checks.append("default_bedtime is null")

    where_parts = [
        "default_morning in (%s::time, %s::time)",
        "default_noon in (%s::time, %s::time)",
        "default_evening in (%s::time, %s::time)",
        "default_bedtime in (%s::time, %s::time)",
        *null_checks,
    ]
    params = (
        current_time_db, future_30_db,
        current_time_db, future_30_db,
        current_time_db, future_30_db,
        current_time_db, future_30_db,
    )
    return _fetch_all(
        "select * from public.user_profiles where " + " or ".join(where_parts),
        params,
    )


def get_active_reminder_drugs(line_uid: str, meal_col: str, meal_timing: str) -> list[dict]:
    if meal_col not in {"morning", "noon", "evening", "bedtime"} or not is_database_available():
        return []

    if not _use_postgres():
        res = (
            supabase.table("reminder_schedules")
            .select("drug_name, trade_name")
            .eq("line_uid", line_uid)
            .eq("is_active", True)
            .eq(meal_col, True)
            .eq("meal_timing", meal_timing)
            .execute()
        )
        return res.data or []

    return _fetch_all(
        f"""
        select drug_name, trade_name
        from public.reminder_schedules
        where line_uid = %s
          and is_active is true
          and {meal_col} is true
          and meal_timing = %s
        """,
        (line_uid, meal_timing),
    )


def get_active_reminder_schedules(line_uid: str) -> list[dict]:
    if not is_database_available():
        return []

    columns = "drug_name, trade_name, morning, noon, evening, bedtime, meal_timing"
    if not _use_postgres():
        res = (
            supabase.table("reminder_schedules")
            .select(columns)
            .eq("line_uid", line_uid)
            .eq("is_active", True)
            .execute()
        )
        return res.data or []

    return _fetch_all(
        """
        select drug_name, trade_name, morning, noon, evening, bedtime, meal_timing
        from public.reminder_schedules
        where line_uid = %s and is_active is true
        """,
        (line_uid,),
    )


def create_reminder_schedule(payload: dict) -> None:
    if not is_database_available():
        raise RuntimeError("Database is not connected")

    payload = {**payload, "trade_name": payload.get("trade_name")}
    ensure_user_profile(payload.get("line_uid", ""))
    if not _use_postgres():
        supabase.table("reminder_schedules").insert(payload).execute()
        return

    _execute(
        """
        insert into public.reminder_schedules
            (line_uid, drug_name, trade_name, is_active, morning, noon, evening, bedtime, meal_timing)
        values
            (%(line_uid)s, %(drug_name)s, %(trade_name)s, %(is_active)s, %(morning)s, %(noon)s, %(evening)s, %(bedtime)s, %(meal_timing)s)
        """,
        payload,
    )


def deactivate_reminder(line_uid: str, drug_name: str) -> None:
    if not is_database_available():
        raise RuntimeError("Database is not connected")

    if not _use_postgres():
        (
            supabase.table("reminder_schedules")
            .update({"is_active": False})
            .eq("line_uid", line_uid)
            .eq("drug_name", drug_name)
            .execute()
        )
        return

    _execute(
        """
        update public.reminder_schedules
        set is_active = false
        where line_uid = %s and drug_name = %s
        """,
        (line_uid, drug_name),
    )


def update_user_default_time(line_uid: str, meal_col: str, db_time: str) -> None:
    if meal_col not in {"default_morning", "default_noon", "default_evening", "default_bedtime"}:
        raise ValueError(f"Unsupported meal column: {meal_col}")
    if not is_database_available():
        raise RuntimeError("Database is not connected")

    ensure_user_profile(line_uid)
    if not _use_postgres():
        supabase.table("user_profiles").update({meal_col: db_time}).eq("line_uid", line_uid).execute()
        return

    _execute(
        f"update public.user_profiles set {meal_col} = %s::time where line_uid = %s",
        (db_time, line_uid),
    )
