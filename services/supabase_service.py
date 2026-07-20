import os

from supabase import Client, create_client


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPPORTED_LANGUAGES = ("th", "en", "my", "lo", "zh")
DEFAULT_LANGUAGE = "th"

if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ Warning: SUPABASE_URL หรือ SUPABASE_KEY ยังไม่ได้ตั้งค่าใน Environment Variables")

try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ เชื่อมต่อ Supabase สำเร็จ!")
    else:
        supabase = None
except Exception as e:
    print(f"❌ เชื่อมต่อ Supabase ไม่สำเร็จ: {e}")
    supabase = None


def get_supabase_client() -> Client | None:
    return supabase


def normalize_language(language: str | None) -> str:
    if language in SUPPORTED_LANGUAGES:
        return language
    return DEFAULT_LANGUAGE


def ensure_user_profile(line_uid: str) -> dict | None:
    if not supabase:
        return None

    existing = supabase.table("user_profiles").select("*").eq("line_uid", line_uid).execute()
    if existing.data:
        return existing.data[0]

    payload = {"line_uid": line_uid, "language": DEFAULT_LANGUAGE}
    created = supabase.table("user_profiles").insert(payload).execute()
    if created.data:
        return created.data[0]
    return payload


def get_user_language(line_uid: str) -> str:
    profile = ensure_user_profile(line_uid)
    if not profile:
        return DEFAULT_LANGUAGE
    return normalize_language(profile.get("language"))


def set_user_language(line_uid: str, language: str) -> bool:
    normalized = normalize_language(language)
    if normalized != language or not supabase:
        return False

    ensure_user_profile(line_uid)
    supabase.table("user_profiles").update({"language": normalized}).eq("line_uid", line_uid).execute()
    return True
