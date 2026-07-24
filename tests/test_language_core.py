import os
import sys
import types
import unittest
from pathlib import Path

from linebot.exceptions import LineBotApiError


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


class FakeTextResponse:
    def __init__(self, text):
        self.text = text


class FakeGeminiModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.generate_content_calls = []

    def generate_content(self, **kwargs):
        self.generate_content_calls.append(kwargs)
        return FakeTextResponse(self.response_text)


class FakeGeminiClient:
    def __init__(self, response_text):
        self.models = FakeGeminiModels(response_text)


class FakeLineError:
    def __init__(self, message):
        self.message = message


class FakeLineBotApi:
    def __init__(self, error=None):
        self.error = error
        self.replies = []
        self.pushes = []

    def reply_message(self, reply_token, messages):
        self.replies.append((reply_token, messages))
        if self.error:
            raise self.error

    def push_message(self, user_id, messages):
        self.pushes.append((user_id, messages))


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
        self.assertTrue(main.is_language_command("🌐เปลี่ยนภาษา/Language"))
        self.assertTrue(main.is_language_command("เปลี่ยนภาษา Language"))
        self.assertTrue(main.is_language_command("เปลี่ยนภาษา\nLanguage"))

    def test_rich_menu_drug_list_command_accepts_current_label(self):
        import main

        self.assertTrue(main.is_drug_list_command("💊ยาที่ต้องกิน Drug list"))
        self.assertTrue(main.is_drug_list_command("ยาที่ต้องกิน\nDrug list"))
        self.assertTrue(main.is_drug_list_command("ยาที่ต้องกิน/Drug list"))

    def test_rich_menu_alarm_setting_command_accepts_current_label(self):
        import main

        self.assertTrue(main.is_alarm_setting_command("⏰เปลี่ยนเวลาแจ้งเตือน/Alarm setting"))
        self.assertTrue(main.is_alarm_setting_command("⏰เปลี่ยนเวลาแจ้งเตือน/Alrm setting"))
        self.assertTrue(main.is_alarm_setting_command("เวลาเตือน Alarm setting"))
        self.assertTrue(main.is_alarm_setting_command("เวลาแจ้งเตือน Alarm setting"))
        self.assertTrue(main.is_alarm_setting_command("เวลาแจ้งเตือน\nAlarm setting"))
        self.assertTrue(main.is_alarm_setting_command("เปลี่ยนเวลาแจ้งเตือน\nAlarm setting"))


class MainPromptLanguageTests(unittest.TestCase):
    def test_prompt_instruction_mentions_selected_language(self):
        import main

        self.assertIn("English", main.build_language_instruction("en"))
        self.assertIn("Simplified Chinese", main.build_language_instruction("zh"))


class MainDatabaseSearchQueryTests(unittest.TestCase):
    def test_database_search_query_translates_non_thai_input_to_thai(self):
        import main

        fake_client = FakeGeminiClient("ปวดหัว")

        query = main.build_database_search_query(fake_client, "headache", "en")

        self.assertEqual(query, "ปวดหัว")
        self.assertEqual(len(fake_client.models.generate_content_calls), 1)
        prompt = fake_client.models.generate_content_calls[0]["contents"][0]
        self.assertIn("Thai search query", prompt)
        self.assertIn("headache", prompt)

    def test_database_search_query_uses_original_thai_input_without_ai_call(self):
        import main

        fake_client = FakeGeminiClient("ignored")

        query = main.build_database_search_query(fake_client, "ปวดหัว", "th")

        self.assertEqual(query, "ปวดหัว")
        self.assertEqual(fake_client.models.generate_content_calls, [])


class MainRagFlexLocalizationTests(unittest.TestCase):
    def test_rag_flex_uses_selected_language_for_static_labels(self):
        import main

        flex = main.build_rag_flex_reply(
            "en",
            {
                "symptom": "Headache",
                "advice": "Rest and drink enough water.",
                "recommended_drug": "",
                "warning": "",
            },
        )

        header_text = flex["header"]["contents"][0]["text"]
        first_body_text = flex["body"]["contents"][0]["text"]
        button = flex["footer"]["contents"][0]["action"]

        self.assertEqual(header_text, "👩‍⚕️ Banya Sookjai Recommendation")
        self.assertEqual(first_body_text, "🩺 Symptom: Headache")
        self.assertEqual(button["label"], "Contact pharmacist")
        self.assertEqual(button["text"], "Contact pharmacist")
        self.assertNotIn("บ้านยาสุขใจ", header_text)
        self.assertNotIn("อาการ", first_body_text)
        self.assertNotIn("ติดต่อเภสัชกร", button["label"])

    def test_rag_flex_does_not_emit_empty_text_components(self):
        import main

        flex = main.build_rag_flex_reply(
            "zh",
            {
                "symptom": "头痛",
                "advice": "",
                "recommended_drug": "",
                "warning": "",
            },
        )

        def collect_text_values(node):
            if isinstance(node, dict):
                values = []
                if "text" in node:
                    values.append(node["text"])
                for value in node.values():
                    values.extend(collect_text_values(value))
                return values
            if isinstance(node, list):
                values = []
                for item in node:
                    values.extend(collect_text_values(item))
                return values
            return []

        self.assertNotIn("", collect_text_values(flex))


class MainMedicineLabelFlexLocalizationTests(unittest.TestCase):
    def test_medicine_label_flex_uses_selected_language_for_static_labels(self):
        import main

        flex = main.build_medicine_label_flex_reply(
            "zh",
            {
                "trade_name": "Tylenol",
                "generic_name": "Paracetamol",
                "indication": "用于缓解疼痛和退烧",
                "dosage": "每次1片",
                "instruction": "饭后服用",
                "warning": "请勿超过推荐剂量",
            },
            time_payload="morning",
            meal_timing="after",
        )

        header_text = flex["header"]["contents"][0]["text"]
        body_texts = [
            item["text"]
            for item in flex["body"]["contents"]
            if item.get("type") == "text"
        ]
        button_labels = [
            item["action"]["label"]
            for item in flex["footer"]["contents"]
            if item.get("type") == "button"
        ]

        self.assertEqual(header_text, "💊 药品标签信息")
        self.assertIn("商品名: Tylenol", body_texts)
        self.assertIn("药品名称: Paracetamol", body_texts)
        self.assertIn("🎯 适应症: 用于缓解疼痛和退烧", body_texts)
        self.assertIn("⏱️ 用法: 饭后服用", body_texts)
        self.assertEqual(button_labels, ["⏰ 设置用药提醒", "✅ 知道了"])
        self.assertNotIn("ข้อมูลฉลากยา", header_text)
        self.assertFalse(any("วิธีใช้" in text for text in body_texts))

    def test_translate_medicine_label_fields_translates_display_text_for_non_thai_users(self):
        import main

        fake_client = FakeGeminiClient(
            '{"indication":"用于缓解疼痛","dosage":"每次1片","instruction":"饭后服用","warning":"请勿超过推荐剂量"}'
        )
        db_data = {
            "trade_name": "Tylenol",
            "generic_name": "Paracetamol",
            "indication": "บรรเทาอาการปวด",
            "dosage_frequency": "ครั้งละ 1 เม็ด",
            "instruction_time": "หลังอาหาร",
            "precaution": "ห้ามใช้เกินขนาด",
        }

        display = main.build_medicine_label_display_data(fake_client, db_data, "zh")

        self.assertEqual(display["trade_name"], "Tylenol")
        self.assertEqual(display["generic_name"], "Paracetamol")
        self.assertEqual(display["indication"], "用于缓解疼痛")
        self.assertEqual(display["dosage"], "每次1片")
        self.assertEqual(display["instruction"], "饭后服用")
        self.assertEqual(display["warning"], "请勿超过推荐剂量")


class MainReminderLocalizationTests(unittest.TestCase):
    def test_postback_reply_text_uses_selected_language(self):
        import main

        self.assertEqual(
            main.build_acknowledge_reply("zh"),
            "✅ 系统已确认。您可以继续询问此药品，或发送另一张药品标签照片。",
        )
        self.assertEqual(
            main.build_reminder_saved_reply("zh", "PROCTASE-P", "after"),
            "⏰ 已为 PROCTASE-P（饭后）设置用药提醒。",
        )
        self.assertEqual(
            main.build_take_pill_reply("zh", "morning"),
            "✅ 做得好！已记录您早晨的用药。祝您身体健康 💙",
        )
        self.assertEqual(
            main.build_snooze_reply("zh"),
            "💤 已确认。提醒将延后 15 分钟。准备好时请记得服药。",
        )
        self.assertEqual(
            main.build_stop_drug_reply("zh", "PROCTASE-P"),
            "⏹️ 已记录 PROCTASE-P 已用完，并停止此药品的提醒。",
        )

    def test_reminder_alert_flex_uses_selected_language_for_static_labels(self):
        import main

        flex = main.build_reminder_alert_flex(
            "my",
            meal="morning",
            timing="after",
            drugs=[{"drug_name": "PROCTASE-P"}],
        )

        header_text = flex["header"]["contents"][0]["text"]
        meal_text = flex["body"]["contents"][0]["text"]
        drug_button_label = flex["body"]["contents"][2]["contents"][1]["action"]["label"]
        footer_labels = [
            item["action"]["label"]
            for item in flex["footer"]["contents"]
            if item.get("type") == "button"
        ]

        self.assertEqual(header_text, "🔔 ဆေးသောက်ရန် အချိန်ရောက်ပါပြီ!")
        self.assertEqual(meal_text, "မုန့်/အစားအစာ: မနက်စာစားပြီး 🌅")
        self.assertEqual(drug_button_label, "ဆေးကုန်ပြီ")
        self.assertEqual(footer_labels, ["✅ ဆေးအားလုံး သောက်ပြီးပြီ", "💤 15 မိနစ်ရွှေ့မည်"])
        self.assertNotIn("ได้เวลากินยา", header_text)
        self.assertNotIn("มื้อ", meal_text)
        self.assertNotIn("ยาหมด", drug_button_label)


class MainLineReplyFallbackTests(unittest.TestCase):
    def test_reply_or_push_uses_reply_token_when_it_is_valid(self):
        import main

        fake_line_api = FakeLineBotApi()

        main.reply_or_push_message(fake_line_api, "U123", "reply-token", "message")

        self.assertEqual(fake_line_api.replies, [("reply-token", "message")])
        self.assertEqual(fake_line_api.pushes, [])

    def test_reply_or_push_falls_back_to_push_when_reply_token_is_invalid(self):
        import main

        invalid_reply_error = LineBotApiError(
            status_code=400,
            headers={},
            request_id="request-id",
            error=FakeLineError("Invalid reply token"),
        )
        fake_line_api = FakeLineBotApi(error=invalid_reply_error)

        main.reply_or_push_message(fake_line_api, "U123", "expired-reply-token", "message")

        self.assertEqual(fake_line_api.replies, [("expired-reply-token", "message")])
        self.assertEqual(fake_line_api.pushes, [("U123", "message")])


if __name__ == "__main__":
    unittest.main()
