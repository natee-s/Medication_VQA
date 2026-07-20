import os
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-key")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

if "pytz" not in sys.modules:
    fake_pytz = types.ModuleType("pytz")
    fake_pytz.timezone = lambda name: None
    sys.modules["pytz"] = fake_pytz


class FakeExecuteResult:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, table, operation="select", payload=None):
        self.table = table
        self.operation = operation
        self.payload = payload
        self.filters = []

    def select(self, columns):
        self.columns = columns
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def insert(self, payload):
        return FakeQuery(self.table, operation="insert", payload=payload)

    def update(self, payload):
        return FakeQuery(self.table, operation="update", payload=payload)

    def execute(self):
        if self.operation == "insert":
            self.table.inserted.append(self.payload)
            return FakeExecuteResult([self.payload])
        if self.operation == "update":
            self.table.updated.append(self.payload)
            return FakeExecuteResult([self.payload])
        return FakeExecuteResult(self.table.next_select_data)


class FakeTable:
    def __init__(self):
        self.next_select_data = []
        self.inserted = []
        self.updated = []

    def query(self):
        return FakeQuery(self)


class FakeSupabase:
    def __init__(self):
        self.user_profiles = FakeTable()

    def table(self, name):
        if name != "user_profiles":
            raise AssertionError(f"unexpected table: {name}")
        return self.user_profiles.query()


class LanguageServiceTests(unittest.TestCase):
    def setUp(self):
        import services.supabase_service as service

        self.service = service
        self.original_supabase = service.supabase
        self.fake = FakeSupabase()
        service.supabase = self.fake

    def tearDown(self):
        self.service.supabase = self.original_supabase

    def test_normalize_language_falls_back_to_thai_for_unknown_values(self):
        self.assertEqual(self.service.normalize_language("en"), "en")
        self.assertEqual(self.service.normalize_language("zh"), "zh")
        self.assertEqual(self.service.normalize_language("xx"), "th")
        self.assertEqual(self.service.normalize_language(None), "th")

    def test_ensure_user_profile_inserts_default_language_when_missing(self):
        self.fake.user_profiles.next_select_data = []

        profile = self.service.ensure_user_profile("U123")

        self.assertEqual(profile["line_uid"], "U123")
        self.assertEqual(profile["language"], "th")
        self.assertEqual(
            self.fake.user_profiles.inserted,
            [{"line_uid": "U123", "language": "th"}],
        )

    def test_get_user_language_reads_existing_profile(self):
        self.fake.user_profiles.next_select_data = [{"line_uid": "U123", "language": "my"}]

        self.assertEqual(self.service.get_user_language("U123"), "my")

    def test_set_user_language_rejects_unknown_language(self):
        self.assertFalse(self.service.set_user_language("U123", "xx"))
        self.assertEqual(self.fake.user_profiles.updated, [])


class MainLanguageHelperTests(unittest.TestCase):
    def test_translation_helper_falls_back_to_thai(self):
        import main

        self.assertEqual(main.t("xx", "language_picker_title"), main.t("th", "language_picker_title"))
        self.assertEqual(main.t("en", "missing_key"), main.t("th", "generic_processing_error"))

    def test_language_picker_contains_five_language_postbacks(self):
        import main

        picker = main.build_language_picker("en")
        body_contents = picker["body"]["contents"]
        postbacks = [
            item["action"]["data"]
            for item in body_contents
            if item.get("type") == "button"
        ]

        self.assertEqual(
            postbacks,
            [
                "action=set_language&lang=th",
                "action=set_language&lang=en",
                "action=set_language&lang=my",
                "action=set_language&lang=lo",
                "action=set_language&lang=zh",
            ],
        )

    def test_ai_language_name_uses_simplified_chinese_for_zh(self):
        import main

        self.assertEqual(main.get_ai_language_name("zh"), "Simplified Chinese")
        self.assertEqual(main.get_ai_language_name("unknown"), "Thai")


class MainLanguageCommandTests(unittest.TestCase):
    def test_language_commands_include_thai_english_and_combined_label(self):
        import main

        self.assertIn("เปลี่ยนภาษา", main.LANGUAGE_COMMANDS)
        self.assertIn("Change Language", main.LANGUAGE_COMMANDS)
        self.assertIn("เปลี่ยนภาษา / Change Language", main.LANGUAGE_COMMANDS)

    def test_rich_menu_language_command_accepts_common_text_variations(self):
        import main

        thai_command = "\u0e40\u0e1b\u0e25\u0e35\u0e48\u0e22\u0e19\u0e20\u0e32\u0e29\u0e32"

        self.assertTrue(main.is_language_command(f"{thai_command} / Change Language"))
        self.assertTrue(main.is_language_command(f"  {thai_command}  /  Change Language  "))
        self.assertTrue(main.is_language_command(f"{thai_command}\nChange Language"))
        self.assertTrue(main.is_language_command(f"{thai_command}／Change Language"))
        self.assertTrue(main.is_language_command("change language"))


class MainPromptLanguageTests(unittest.TestCase):
    def test_prompt_instruction_mentions_selected_language(self):
        import main

        self.assertIn("English", main.build_language_instruction("en"))
        self.assertIn("Simplified Chinese", main.build_language_instruction("zh"))


if __name__ == "__main__":
    unittest.main()
