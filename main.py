from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    ImageMessage,
    FlexSendMessage,
    PostbackEvent,
    FollowEvent,
    StickerMessage,
    VideoMessage,
    AudioMessage,
    LocationMessage,
    FileMessage,
)
import os
import cv2
import numpy as np
from google import genai
from google.genai import types
import json
import requests
from services.supabase_service import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    SUPABASE_URL,
    ensure_user_profile,
    get_user_language,
    normalize_language,
    set_user_language,
    supabase,
)
from urllib.parse import parse_qsl
from datetime import datetime
import pytz
import logging


# ==========================================
# 1. ฟังก์ชันสร้างฟังก์ชันด่านหน้า (Gatekeeper)
# ==========================================
def check_image_quality(file_path):
    # 1. ตรวจสอบขนาดไฟล์ (ไม่เกิน 3MB)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"🔍 [TEST] ขนาดไฟล์รูปนี้คือ: {file_size_mb:.1f} MB")
    if file_size_mb > 3.0:
        return False, f"⚠️ รูปภาพมีขนาดใหญ่เกินไป ({file_size_mb:.1f} MB) กรุณาส่งรูปไม่เกิน 3 MB ครับ หรือถ่ายผ่านกล้องของ LINE ได้เลยครับ"

    img = cv2.imread(file_path)
    if img is None:
        return False, "⚠️ ไม่สามารถอ่านไฟล์รูปภาพได้ กรุณาส่งใหม่อีกครั้งครับ"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. ตรวจสอบความสว่าง
    brightness = np.mean(gray)
    print(f"🔍 [TEST] ค่าความสว่างรูปนี้คือ: {brightness}")
    if brightness < 50:
        return False, "⚠️ รูปภาพมืดเกินไป กรุณาถ่ายในที่สว่างแล้วส่งมาใหม่ครับ"
    
    # ตรวจแสงสะท้อน (พิกเซลสว่างจัด > 240 มีมากกว่า 5% ของพื้นที่ภาพ)
    glare_ratio = np.sum(gray > 240) / gray.size
    print(f"🔍 [TEST] ค่าแสงสะท้อนรูปนี้คือ: {glare_ratio}")
    if glare_ratio > 0.05:
        return False, "⚠️ รูปภาพมีแสงแฟลชสะท้อนบังข้อความ กรุณาหลีกเลี่ยงแสงสะท้อนแล้วถ่ายใหม่ครับ"

    # 3. ตรวจสอบความเปรียบต่างสี (Contrast)
    contrast = np.std(gray)
    print(f"🔍 [TEST] ค่าความเปรียบต่างสีรูปนี้คือ: {contrast}")
    if contrast < 20:
        return False, "⚠️ รูปภาพจางหรือสีกลืนกันเกินไป ทำให้ระบบอาจอ่านผิดพลาด กรุณาถ่ายใหม่อีกครั้งครับ"

    # 4. ตรวจสอบความเบลอ (Blurriness)
    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"🔍 [TEST] ค่าความเบลอรูปนี้คือ: {blur_val}")
    if blur_val < 100:
        return False, "⚠️ รูปภาพเบลอเกินไป กรุณาแตะโฟกัสที่กล้องให้ตัวหนังสือคมชัด แล้วถ่ายใหม่ครับ"

    # 5. ตรวจสอบระยะห่าง (Bounding Box Area)
    # เพิ่ม GaussianBlur เพื่อเบลอลายไม้บนโต๊ะและจุดรบกวนก่อนหาขอบ
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # กรองเอาเฉพาะเส้นขอบที่มีขนาดใหญ่กว่า 500 พิกเซล (ลบขยะทิ้ง)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    
    if valid_contours:
        x_min, y_min = img.shape[1], img.shape[0]
        x_max, y_max = 0, 0
        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)
        
        object_area = (x_max - x_min) * (y_max - y_min)
        total_area = img.shape[0] * img.shape[1]
        
        print(f"🔍 [TEST] ค่าพื้นที่วัตถุ: {object_area}, ค่าพื้นที่ภาพรวม: {total_area}, สัดส่วน: {object_area/total_area:.3f}")
        
        #if (object_area / total_area) < 0.15: # ปรับลดเกณฑ์ลงนิดหน่อยเหลือ 15% 
            #return False, "⚠️ รูปภาพอยู่ไกลเกินไป กรุณาถ่ายใกล้ๆ ให้ฉลากยาเต็มกรอบภาพครับ"

    # 6. Auto-Deskew (แก้เอียงอัตโนมัติ 1-15 องศา)
    coords = np.column_stack(np.where(edges > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # ถ้าเอียงนิดหน่อย ให้หมุนภาพเลย
        if 1 < abs(angle) < 15:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(file_path, rotated) # บันทึกทับไฟล์เดิมที่แก้เอียงแล้ว

    return True, "OK"


# ==========================================
# 1. ฟังก์ชันผู้เชี่ยวชาญการล้างภาพ (Image Preprocessing)
# ==========================================
def process_pharmacy_label(input_path, output_path):
    img = cv2.imread(input_path)
    height, width = img.shape[:2]
    if width < 1000:
        scale = 1000 / width
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    balanced = clahe.apply(gray)
    
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(balanced, -1, kernel)
    
    denoised = cv2.medianBlur(sharpened, 3)
    
    processed = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    cv2.imwrite(output_path, processed)
    h, w = processed.shape
    return w, h
# ==========================================
# ⚡ ฟิลเตอร์ซ่อน Log Uvicorn เฉพาะเส้นทาง Cron Job
# ==========================================
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # ถ้าข้อความ Log มีคำว่า /cron/check-reminder ให้ซ่อนไปเลย (return False)
        return record.getMessage().find("/cron/check-reminder") == -1

# นำ Filter ไปติดไว้ที่ระบบ Log ของเซิร์ฟเวอร์
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# ==========================================
# 2. การตั้งค่าเซิร์ฟเวอร์, LINE Bot และ Gemini API
# ==========================================
app = FastAPI()

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN', 'YOUR_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET', 'YOUR_SECRET')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ประกาศเรียกใช้งาน Client ของ Gemini ด้วยรหัสคีย์ที่เราฝากไว้บน Render
ai_client = genai.Client(api_key=GEMINI_API_KEY)


def load_messages():
    locale_path = os.path.join(os.path.dirname(__file__), "locales", "i18n.json")
    with open(locale_path, "r", encoding="utf-8") as file:
        return json.load(file)


MESSAGES = load_messages()

LANGUAGE_OPTIONS = (
    {"code": "th", "label": "🇹🇭 ไทย", "ai_name": "Thai"},
    {"code": "en", "label": "🇬🇧 English", "ai_name": "English"},
    {"code": "my", "label": "🇲🇲 မြန်မာ", "ai_name": "Burmese"},
    {"code": "lo", "label": "🇱🇦 ລາວ", "ai_name": "Lao"},
    {"code": "zh", "label": "🇨🇳 中文", "ai_name": "Simplified Chinese"},
)
AI_LANGUAGE_NAMES = {item["code"]: item["ai_name"] for item in LANGUAGE_OPTIONS}
LANGUAGE_COMMANDS = {"เปลี่ยนภาษา", "Change Language", "เปลี่ยนภาษา / Change Language"}


def reply_or_push_message(line_api, user_id: str, reply_token: str, messages):
    try:
        line_api.reply_message(reply_token, messages)
    except LineBotApiError as e:
        error_message = getattr(getattr(e, "error", None), "message", "")
        if e.status_code == 400 and "Invalid reply token" in error_message:
            print(f"⚠️ LINE reply token หมดอายุสำหรับ {user_id}; ส่ง fallback ด้วย push_message")
            line_api.push_message(user_id, messages)
            return
        raise


def normalize_command_text(text: str) -> str:
    return " ".join((text or "").replace("／", "/").split())


def is_language_command(text: str) -> bool:
    normalized_text = normalize_command_text(text)
    lowered_text = normalized_text.lower()
    lowered_commands = {command.lower() for command in LANGUAGE_COMMANDS}

    return (
        normalized_text in LANGUAGE_COMMANDS
        or lowered_text in lowered_commands
        or ("เปลี่ยนภาษา" in normalized_text and "change language" in lowered_text)
    )


def t(lang: str, key: str, **kwargs) -> str:
    language = normalize_language(lang)
    thai_messages = MESSAGES.get(DEFAULT_LANGUAGE, {})
    message = (
        MESSAGES.get(language, {}).get(key)
        or thai_messages.get(key)
        or thai_messages.get("generic_processing_error", "")
    )
    return message.format(**kwargs)


def get_ai_language_name(lang: str) -> str:
    return AI_LANGUAGE_NAMES.get(normalize_language(lang), "Thai")


def build_language_instruction(lang: str) -> str:
    return f"You must answer the user only in: {get_ai_language_name(lang)}."


def build_database_search_query(client, user_text: str, lang: str) -> str:
    original_text = (user_text or "").strip()
    if not original_text or normalize_language(lang) == DEFAULT_LANGUAGE:
        return original_text

    prompt = f"""
Convert the user's health or medicine question into one concise Thai search query
for searching a Thai pharmacy database.

Rules:
- Return only the Thai search query.
- Keep it short: 1 to 5 Thai words.
- Focus on symptoms, medicine names, or indications.
- Do not include explanations, quotes, markdown, or punctuation.

User language: {get_ai_language_name(lang)}
User message: {original_text}
Thai search query:
""".strip()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
        )
        search_query = (getattr(response, "text", "") or "").strip().strip('"').strip("'")
        return search_query or original_text
    except Exception as e:
        print(f"⚠️ [Search Query Translation] fallback to original text: {e}")
        return original_text


def build_rag_flex_reply(lang: str, ai_data: dict) -> dict:
    def clean_field(key: str) -> str:
        return str(ai_data.get(key) or "").strip()

    symptom_text = clean_field("symptom") or "-"
    advice_text = clean_field("advice")
    recommended_drug_text = clean_field("recommended_drug")
    warning_text = clean_field("warning")

    body_contents = [
        {
            "type": "text",
            "text": f"🩺 {t(lang, 'rag_symptom_label')}: {symptom_text}",
            "weight": "bold",
            "color": "#1DB446",
            "wrap": True,
        },
    ]

    if advice_text:
        body_contents.append({
            "type": "text",
            "text": advice_text,
            "wrap": True,
            "size": "sm",
            "margin": "md",
            "color": "#333333",
        })

    if recommended_drug_text:
        body_contents.append({"type": "separator", "margin": "lg"})
        body_contents.append(
            {
                "type": "text",
                "text": f"💊 {t(lang, 'rag_recommended_drug_label')}",
                "weight": "bold",
                "size": "sm",
                "color": "#009688",
                "margin": "md",
            }
        )
        body_contents.append(
            {
                "type": "text",
                "text": recommended_drug_text,
                "wrap": True,
                "size": "sm",
                "color": "#666666",
            }
        )

    if warning_text:
        body_contents.append({"type": "separator", "margin": "lg"})
        body_contents.append(
            {
                "type": "text",
                "text": f"⚠️ {t(lang, 'rag_warning_label')}",
                "weight": "bold",
                "size": "sm",
                "color": "#F44336",
                "margin": "md",
            }
        )
        body_contents.append(
            {
                "type": "text",
                "text": warning_text,
                "wrap": True,
                "size": "sm",
                "color": "#666666",
            }
        )

    contact_pharmacist_text = t(lang, "contact_pharmacist_button")
    return {
        "type": "bubble",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#1DB446",
            "contents": [
                {
                    "type": "text",
                    "text": f"👩‍⚕️ {t(lang, 'rag_header_title')}",
                    "color": "#FFFFFF",
                    "weight": "bold",
                }
            ],
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": body_contents,
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "action": {
                        "type": "message",
                        "label": contact_pharmacist_text,
                        "text": contact_pharmacist_text,
                    },
                }
            ],
        },
    }


def build_medicine_label_display_data(client, db_data: dict, lang: str) -> dict:
    display_data = {
        "trade_name": db_data.get("trade_name") or t(lang, "not_specified"),
        "generic_name": db_data.get("generic_name") or t(lang, "not_specified"),
        "indication": db_data.get("indication") or t(lang, "not_specified"),
        "dosage": db_data.get("dosage_frequency") or t(lang, "not_specified"),
        "instruction": db_data.get("instruction_time") or t(lang, "not_specified"),
        "warning": db_data.get("precaution") or t(lang, "no_warning"),
    }

    if normalize_language(lang) == DEFAULT_LANGUAGE:
        return display_data

    prompt = f"""
Translate the following medicine label display fields into {get_ai_language_name(lang)}.

Rules:
- Return JSON only.
- Keep trade_name and generic_name unchanged.
- Translate indication, dosage, instruction, and warning.
- Do not add medical advice beyond the source text.

Source JSON:
{json.dumps(display_data, ensure_ascii=False)}

Required JSON keys:
{{
  "indication": "...",
  "dosage": "...",
  "instruction": "...",
  "warning": "..."
}}
""".strip()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config={"response_mime_type": "application/json"},
        )
        translated_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        translated_data = json.loads(translated_text)
        for key in ("indication", "dosage", "instruction", "warning"):
            translated_value = str(translated_data.get(key) or "").strip()
            if translated_value:
                display_data[key] = translated_value
    except Exception as e:
        print(f"⚠️ [Medicine Label Translation] fallback to source language: {e}")

    return display_data


def build_medicine_label_flex_reply(lang: str, display_data: dict, time_payload: str, meal_timing: str) -> dict:
    generic_name = display_data.get("generic_name") or t(lang, "not_specified")
    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#1DB446",
            "contents": [
                {
                    "type": "text",
                    "text": f"💊 {t(lang, 'medicine_label_title')}",
                    "weight": "bold",
                    "size": "lg",
                    "color": "#FFFFFF",
                }
            ],
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                {
                    "type": "text",
                    "text": f"{t(lang, 'medicine_trade_name_label')}: {display_data.get('trade_name') or t(lang, 'not_specified')}",
                    "weight": "bold",
                    "wrap": True,
                },
                {
                    "type": "text",
                    "text": f"{t(lang, 'medicine_generic_name_label')}: {generic_name}",
                    "color": "#666666",
                    "size": "sm",
                    "wrap": True,
                },
                {"type": "separator", "margin": "md"},
                {
                    "type": "text",
                    "text": f"🎯 {t(lang, 'medicine_indication_label')}: {display_data.get('indication') or t(lang, 'not_specified')}",
                    "wrap": True,
                },
                {
                    "type": "text",
                    "text": f"⚖️ {t(lang, 'medicine_dosage_label')}: {display_data.get('dosage') or t(lang, 'not_specified')}",
                    "wrap": True,
                },
                {
                    "type": "text",
                    "text": f"⏱️ {t(lang, 'medicine_instruction_label')}: {display_data.get('instruction') or t(lang, 'not_specified')}",
                    "weight": "bold",
                    "color": "#E03131",
                    "wrap": True,
                },
                {
                    "type": "text",
                    "text": f"⚠️ {t(lang, 'medicine_warning_label')}: {display_data.get('warning') or t(lang, 'no_warning')}",
                    "size": "sm",
                    "color": "#FFA500",
                    "wrap": True,
                },
            ],
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": f"⏰ {t(lang, 'set_reminder_button')}",
                        "data": f"action=set_reminder&drug={generic_name}&time={time_payload}&timing={meal_timing}",
                    },
                },
                {
                    "type": "button",
                    "style": "secondary",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": f"✅ {t(lang, 'acknowledge_button')}",
                        "data": "action=acknowledge",
                    },
                },
            ],
        },
    }


def get_timing_text(lang: str, timing: str) -> str:
    key = "timing_before" if timing == "before" else "timing_after"
    return t(lang, key)


def get_meal_text(lang: str, meal: str) -> str:
    key_by_meal = {
        "morning": "meal_morning",
        "noon": "meal_noon",
        "evening": "meal_evening",
        "bedtime": "meal_bedtime",
    }
    return t(lang, key_by_meal.get(meal, "meal_morning"))


def get_reminder_meal_display(lang: str, meal: str, timing: str) -> str:
    key = f"reminder_meal_{'before' if timing == 'before' else 'after'}_{meal}"
    return t(lang, key)


def build_acknowledge_reply(lang: str) -> str:
    return t(lang, "acknowledge_saved_message")


def build_reminder_saved_reply(lang: str, drug_name: str, timing: str) -> str:
    return t(
        lang,
        "reminder_saved_message",
        drug=drug_name,
        timing=get_timing_text(lang, timing),
    )


def build_stop_drug_reply(lang: str, drug_name: str) -> str:
    return t(lang, "medicine_finished_message", drug=drug_name)


def build_take_pill_reply(lang: str, meal: str) -> str:
    return t(lang, "take_pill_saved_message", meal=get_meal_text(lang, meal))


def build_snooze_reply(lang: str) -> str:
    return t(lang, "snooze_message")


def build_reminder_alert_flex(lang: str, meal: str, timing: str, drugs: list[dict]) -> dict:
    meal_display = get_reminder_meal_display(lang, meal, timing)
    drug_list_contents = []

    for drug in drugs:
        drug_name = drug.get("drug_name", "")
        drug_list_contents.append(
            {
                "type": "box",
                "layout": "horizontal",
                "spacing": "sm",
                "margin": "md",
                "contents": [
                    {
                        "type": "text",
                        "text": f"💊 {drug_name}",
                        "size": "sm",
                        "weight": "bold",
                        "color": "#333333",
                        "gravity": "center",
                        "wrap": True,
                        "flex": 2,
                    },
                    {
                        "type": "button",
                        "style": "secondary",
                        "height": "sm",
                        "flex": 1,
                        "action": {
                            "type": "postback",
                            "label": t(lang, "medicine_finished_button"),
                            "data": f"action=stop_drug&drug={drug_name}",
                        },
                    },
                ],
            }
        )

    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#FFC107",
            "contents": [
                {
                    "type": "text",
                    "text": f"🔔 {t(lang, 'reminder_alert_title')}",
                    "weight": "bold",
                    "size": "lg",
                    "color": "#FFFFFF",
                }
            ],
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                {
                    "type": "text",
                    "text": f"{t(lang, 'reminder_meal_label')}: {meal_display}",
                    "weight": "bold",
                    "size": "md",
                    "color": "#1DB446",
                },
                {"type": "separator", "margin": "md"},
            ]
            + drug_list_contents,
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#1DB446",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": f"✅ {t(lang, 'take_all_button')}",
                        "data": f"action=take_pill&meal={meal}",
                    },
                },
                {
                    "type": "button",
                    "style": "secondary",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": f"💤 {t(lang, 'snooze_button')}",
                        "data": f"action=snooze&meal={meal}",
                    },
                },
            ],
        },
    }


def build_language_picker(lang: str = DEFAULT_LANGUAGE) -> dict:
    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": "#1DB446",
            "contents": [
                {
                    "type": "text",
                    "text": t(lang, "language_picker_title"),
                    "weight": "bold",
                    "color": "#FFFFFF",
                    "size": "lg",
                }
            ],
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "text",
                    "text": t(lang, "language_picker_subtitle"),
                    "wrap": True,
                    "size": "sm",
                    "color": "#555555",
                },
                {"type": "separator", "margin": "md"},
            ]
            + [
                {
                    "type": "button",
                    "style": "secondary",
                    "height": "sm",
                    "action": {
                        "type": "postback",
                        "label": option["label"],
                        "data": f"action=set_language&lang={option['code']}",
                    },
                }
                for option in LANGUAGE_OPTIONS
            ],
        },
    }


@app.get("/")
def root():
    return {"message": "Banya Sookjai AI Server is running!"}

@app.get("/cron/check-reminder")
def check_reminder():
    if not supabase:
        return {"status": "error", "message": "Supabase not connected"}

    # 1. นำเข้า timedelta เพื่อใช้คำนวณเวลา และดึงเวลาปัจจุบัน
    from datetime import timedelta
    bkk_tz = pytz.timezone('Asia/Bangkok')
    now_bkk = datetime.now(bkk_tz)
    current_time_str = now_bkk.strftime("%H:%M") # เช่น 08:00 (ใช้สำหรับเตือน หลังอาหาร)
    
    # คำนวณเวลาล่วงหน้า 30 นาที
    now_plus_30 = now_bkk + timedelta(minutes=30)
    future_30_str = now_plus_30.strftime("%H:%M") # เช่น 08:30 (ใช้สำหรับเตือน ก่อนอาหาร)

    try:
        current_time_db = f"{current_time_str}:00"
        future_30_db = f"{future_30_str}:00"
        
        # 2. ค้นหาผู้ใช้ที่มีเวลาตรงกับปัจจุบัน หรือตรงกับ 30 นาทีข้างหน้า
        or_conditions = [
            f"default_morning.eq.{current_time_db}", f"default_morning.eq.{future_30_db}",
            f"default_noon.eq.{current_time_db}", f"default_noon.eq.{future_30_db}",
            f"default_evening.eq.{current_time_db}", f"default_evening.eq.{future_30_db}",
            f"default_bedtime.eq.{current_time_db}", f"default_bedtime.eq.{future_30_db}"
        ]
        
        # 2.1 ดักจับกรณีผู้ใช้ไม่ได้ตั้งเวลาเอง (ใช้เวลามาตรฐานของร้าน)
        # ตรวจสอบเพื่อเตือน ยาหลังอาหาร
        if current_time_str == "08:00": or_conditions.append("default_morning.is.null")
        if current_time_str == "12:00": or_conditions.append("default_noon.is.null")
        if current_time_str == "18:00": or_conditions.append("default_evening.is.null")
        if current_time_str == "21:00": or_conditions.append("default_bedtime.is.null")
        
        # ตรวจสอบเพื่อเตือน ยาก่อนอาหาร (ลบ 30 นาทีจากเวลามาตรฐาน)
        if current_time_str == "07:30": or_conditions.append("default_morning.is.null")
        if current_time_str == "11:30": or_conditions.append("default_noon.is.null")
        if current_time_str == "17:30": or_conditions.append("default_evening.is.null")
        if current_time_str == "20:30": or_conditions.append("default_bedtime.is.null")
        
        query_string = ",".join(or_conditions)
        users_res = supabase.table("user_profiles").select("*").or_(query_string).execute()
        users = users_res.data
        
        count_messages_sent = 0

        for user in users:
            uid = user.get("line_uid")
            user_language = normalize_language(user.get("language"))
            
            t_morning = str(user.get("default_morning"))[:5] if user.get("default_morning") else "08:00"
            t_noon = str(user.get("default_noon"))[:5] if user.get("default_noon") else "12:00"
            t_evening = str(user.get("default_evening"))[:5] if user.get("default_evening") else "18:00"
            t_bedtime = str(user.get("default_bedtime"))[:5] if user.get("default_bedtime") else "21:00"

            # 3. เตรียมรอบการแจ้งเตือน (แยกตะกร้ายาก่อน/หลังอาหาร อย่างชาญฉลาด)
            meals_to_trigger = []
            
            if current_time_str == t_morning:
                meals_to_trigger.append({"meal": "morning", "timing": "after", "meal_name_th": "หลังอาหารเช้า 🌅"})
            if future_30_str == t_morning:
                meals_to_trigger.append({"meal": "morning", "timing": "before", "meal_name_th": "ก่อนอาหารเช้า 🌅"})
            
            if current_time_str == t_noon:
                meals_to_trigger.append({"meal": "noon", "timing": "after", "meal_name_th": "หลังอาหารกลางวัน ☀️"})
            if future_30_str == t_noon:
                meals_to_trigger.append({"meal": "noon", "timing": "before", "meal_name_th": "ก่อนอาหารกลางวัน ☀️"})
                
            if current_time_str == t_evening:
                meals_to_trigger.append({"meal": "evening", "timing": "after", "meal_name_th": "หลังอาหารเย็น 🌆"})
            if future_30_str == t_evening:
                meals_to_trigger.append({"meal": "evening", "timing": "before", "meal_name_th": "ก่อนอาหารเย็น 🌆"})
                
            if current_time_str == t_bedtime:
                meals_to_trigger.append({"meal": "bedtime", "timing": "after", "meal_name_th": "ก่อนนอน 🌙"})
            if future_30_str == t_bedtime:
                meals_to_trigger.append({"meal": "bedtime", "timing": "before", "meal_name_th": "ก่อนนอน (ล่วงหน้า 30 นาที) 🌙"})

            # 4. วนลูปส่งการแจ้งเตือนเฉพาะยาที่ตรงเงื่อนไข
            for trigger in meals_to_trigger:
                meal_col = trigger["meal"]
                timing = trigger["timing"]
                meal_name_th = trigger["meal_name_th"]
                meal_display = get_reminder_meal_display(user_language, meal_col, timing)
                
                # ค้นหายาที่ผูกกับเวลาและประเภทก่อน/หลังอาหารนี้
                reminders_res = supabase.table("reminder_schedules").select("drug_name")\
                    .eq("line_uid", uid).eq("is_active", True).eq(meal_col, True).eq("meal_timing", timing).execute()
                drugs = reminders_res.data
                
                if drugs:
                    flex_alert = build_reminder_alert_flex(user_language, meal_col, timing, drugs)
                    line_bot_api.push_message(
                        uid,
                        FlexSendMessage(
                            alt_text=t(user_language, "reminder_alt_text", meal=meal_display),
                            contents=flex_alert,
                        )
                    )
                    count_messages_sent += 1
                    continue

                    drug_list_contents = []
                    for d in drugs:
                        drug_name = d["drug_name"]
                        drug_list_contents.append({
                            "type": "box",
                            "layout": "horizontal",
                            "spacing": "sm",
                            "margin": "md",
                            "contents": [
                                {
                                    "type": "text", "text": f"💊 {drug_name}", "size": "sm", "weight": "bold", 
                                    "color": "#333333", "gravity": "center", "wrap": True, "flex": 2
                                },
                                {
                                    "type": "button", "style": "secondary", "height": "sm", "flex": 1,
                                    "action": {"type": "postback", "label": "ยาหมด", "data": f"action=stop_drug&drug={drug_name}"}
                                }
                            ]
                        })

                    flex_alert = {
                        "type": "bubble",
                        "size": "mega",
                        "header": {
                            "type": "box",
                            "layout": "vertical",
                            "backgroundColor": "#FFC107",
                            "contents": [
                                {"type": "text", "text": "🔔 ได้เวลากินยาแล้วครับ!", "weight": "bold", "size": "lg", "color": "#FFFFFF"}
                            ]
                        },
                        "body": {
                            "type": "box",
                            "layout": "vertical",
                            "spacing": "md",
                            "contents": [
                                {"type": "text", "text": f"มื้อ: {meal_name_th}", "weight": "bold", "size": "md", "color": "#1DB446"},
                                {"type": "separator", "margin": "md"}
                            ] + drug_list_contents 
                        },
                        "footer": {
                            "type": "box",
                            "layout": "vertical",
                            "spacing": "sm",
                            "contents": [
                                {
                                    "type": "button", "style": "primary", "color": "#1DB446", "height": "sm",
                                    "action": {"type": "postback", "label": "✅ กินยาทั้งหมดแล้ว", "data": f"action=take_pill&meal={meal_col}"}
                                },
                                {
                                    "type": "button", "style": "secondary", "height": "sm",
                                    "action": {"type": "postback", "label": "💤 เลื่อน 15 นาที", "data": f"action=snooze&meal={meal_col}"}
                                }
                            ]
                        }
                    }

                    line_bot_api.push_message(
                        uid, 
                        FlexSendMessage(alt_text=f"เตือนกินยา: {meal_name_th}", contents=flex_alert)
                    )
                    count_messages_sent += 1

        return {"status": "success", "message": f"เช็กเวลาสำเร็จ ส่งแจ้งเตือนไป {count_messages_sent} รายการ"}

    except Exception as e:
        print(f"❌ Error in cron job: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get('X-Line-Signature', '')
    body = await request.body()
    body_str = body.decode('utf-8')
    
    try:
        handler.handle(body_str, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    return 'OK'


# ==========================================
# ฟังก์ชันค้นหาข้อมูลยา (RAG Search)
# ==========================================
def search_medicine_in_db(drug_name: str):
    """
    ฟังก์ชันนี้จะรับชื่อยา (ที่ AI อ่านได้) มาค้นหาในฐานข้อมูล
    เพื่อดึงข้อมูล Official กลับไปแสดงผล
    """
    if not supabase:
        print("⚠️ สัญญาณการเชื่อมต่อ Supabase ไม่พร้อมใช้งาน")
        return None
        
    try:
        # ใช้ .or_ เพื่อค้นหาชื่อยาจากทั้งคอลัมน์ generic_name และ trade_name
        search_query = f"generic_name.ilike.%{drug_name}%,trade_name.ilike.%{drug_name}%"
        response = supabase.table('Medication_VQA').select('*').or_(search_query).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0] # ส่งข้อมูลแถวแรกที่เจอแจ็กพอตกลับไป
        else:
            return None # ไม่พบข้อมูลในระบบ
            
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการค้นหาข้อมูล: {e}")
        return None

# ==========================================
# 3. ฟังก์ชันหลัก: จัดการเมื่อมีผู้ใช้ส่งรูปภาพเข้ามา
# ==========================================
@handler.add(FollowEvent)
def handle_follow(event):
    user_id = event.source.user_id
    ensure_user_profile(user_id)
    line_bot_api.reply_message(
        event.reply_token,
        FlexSendMessage(
            alt_text=t(DEFAULT_LANGUAGE, "language_picker_alt"),
            contents=build_language_picker(DEFAULT_LANGUAGE),
        ),
    )


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    user_id = event.source.user_id
    user_language = get_user_language(user_id)
    language_instruction = build_language_instruction(user_language)

    # --- ส่งสถานะ "กำลังพิมพ์..." (Loading Animation) ---
    url = "https://api.line.me/v2/bot/chat/loading/start"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    data_loading = {"chatId": user_id, "loadingSeconds": 30}
    requests.post(url, headers=headers, json=data_loading)

    # --- ดาวน์โหลดรูปภาพจาก LINE ---
    message_content = line_bot_api.get_message_content(event.message.id)
    temp_file_path = f"/tmp/{event.message.id}.jpg"
    
    with open(temp_file_path, 'wb') as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)
            
    # ==========================================
    # เฟส 1: ด่านตรวจ QC รูปภาพ
    # ==========================================
    is_good, qc_message = check_image_quality(temp_file_path)
    if not is_good:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=qc_message))
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return

    # ==========================================
    # เฟส 2: ข้ามการประมวลผลล้างรูปภาพ (ส่งภาพสีต้นฉบับให้ AI)
    # ==========================================
    with open(temp_file_path, "rb") as image_file:
        image_bytes = image_file.read()

    # ลบไฟล์ชั่วคราวทิ้งหลังอ่านเสร็จ
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    # ==========================================
    # เฟส 3: เรียกใช้งาน Gemini + ค้นหาข้อมูลจริง (RAG)
    # ==========================================
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                """คุณคือระบบ OCR ดึงคีย์เวิร์ดชื่อยาจากภาพเพื่อนำไปค้นหาในฐานข้อมูล
กฎสำคัญ:
1. หากภาพตะแคงหรือกลับหัว ให้ตอบ error เป็น "rotated"
2. หากตั้งตรงปกติ ให้ดึง "ชื่อยาภาษาอังกฤษ (Generic Name หรือ Trade Name ก็ได้)" ที่เด่นชัดที่สุดในภาพออกมาเพียงชื่อเดียว (ระบุเฉพาะชื่อ ไม่ต้องใส่ขนาดมิลลิกรัม)

รูปแบบ JSON ที่ต้องการเท่านั้น (ห้ามมีอธิบายเพิ่ม):
{
"error": "rotated หรือ null",
"search_keyword": "ชื่อยาภาษาอังกฤษ หรือ null"
}""",
                f"{language_instruction} Keep JSON keys exactly as specified; do not translate search_keyword."
            ]
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith('```json'):
            raw_text = raw_text.replace('```json', '').replace('```', '').strip()
        elif raw_text.startswith('```'):
            raw_text = raw_text.replace('```', '').strip()
            
        try:
            data = json.loads(raw_text)
            
            # 🚨 ดักจับ Error รูปกลับหัว
            if data.get("error") == "rotated":
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=t(user_language, "ocr_rotated_image"))
                )
                return

            search_keyword = data.get("search_keyword")
            if not search_keyword:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=t(user_language, "ocr_unclear_drug_name")))
                return

            # ----------------------------------------------------
            # 🎯 เริ่มกระบวนการ RAG (ค้นหาชื่อยาใน Supabase)
            # ----------------------------------------------------
            db_data = search_medicine_in_db(search_keyword)

            if not db_data:
                line_bot_api.reply_message(
                    event.reply_token, 
                    TextSendMessage(text=t(user_language, "ocr_no_database_match", drug=search_keyword))
                )
                return
            
            # จัดเตรียมข้อมูลใส่ Flex Message
            display_data = build_medicine_label_display_data(ai_client, db_data, user_language)
            generic_name = display_data.get("generic_name") or t(user_language, "not_specified")
            instruction_for_reminder = db_data.get('instruction_time') or ''

            # ----------------------------------------------------
            # 🎯 เพิ่มลอจิกวิเคราะห์เวลากินยาจากข้อความ instruction
            # ----------------------------------------------------
            time_list = []
            if instruction_for_reminder:
                if 'เช้า' in instruction_for_reminder: time_list.append('morning')
                if 'กลางวัน' in instruction_for_reminder or 'เที่ยง' in instruction_for_reminder: time_list.append('noon')
                if 'เย็น' in instruction_for_reminder: time_list.append('evening')
                if 'นอน' in instruction_for_reminder: time_list.append('bedtime')
            
            # รวมเป็น text เช่น "morning,bedtime" ถ้าไม่มีให้ส่ง "none"
            time_payload = ",".join(time_list) if time_list else "none"

            # 👇 [เพิ่มใหม่] ลอจิกตรวจสอบ ก่อนอาหาร หรือ หลังอาหาร 👇
            meal_timing = "after" # ตั้งค่าเริ่มต้นให้เป็น 'หลังอาหาร' ไว้ก่อน
            if 'ก่อนอาหาร' in instruction_for_reminder or 'ก่อน' in instruction_for_reminder:
                meal_timing = "before"
            
            print(f"🔍 [DEBUG] ข้อความวิธีใช้จาก DB: {instruction_for_reminder}")
            print(f"🔍 [DEBUG] Time Payload: {time_payload} | Timing: {meal_timing}")
            # ----------------------------------------------------

            flex_bubble = build_medicine_label_flex_reply(
                user_language,
                display_data,
                time_payload,
                meal_timing,
            )

            reply_or_push_message(
                line_bot_api,
                user_id,
                event.reply_token,
                FlexSendMessage(
                    alt_text=t(user_language, "medicine_label_alt", drug=generic_name),
                    contents=flex_bubble,
                ),
            )

        except json.JSONDecodeError:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=t(user_language, "ai_format_error")))
            
    except Exception as e:
        print(f"⚠️ Error in image processing: {e}")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=t(user_language, "generic_processing_error")))    

@handler.add(PostbackEvent)
def handle_postback(event):
    data = event.postback.data
    user_id = event.source.user_id
    user_language = get_user_language(user_id)

    if data.startswith("action=set_language"):
        postback_dict = dict(parse_qsl(data))
        requested_language = postback_dict.get("lang")
        selected_language = normalize_language(requested_language)
        if requested_language in SUPPORTED_LANGUAGES and set_user_language(user_id, selected_language):
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=t(selected_language, "language_saved")),
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=t(DEFAULT_LANGUAGE, "generic_processing_error")),
            )
        return

    # ----------------------------------------
    # กรณีที่ 1: ผู้ใช้กดปุ่ม "✅ รับทราบ"
    # ----------------------------------------
    if data == "action=acknowledge":
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=build_acknowledge_reply(user_language))
        )
        return

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="✅ ระบบรับทราบเรียบร้อยครับ คุณสามารถพิมพ์สอบถามข้อมูลเกี่ยวกับยานี้เพิ่มเติมได้เลยครับ หรือหากต้องการให้อ่านฉลากยาตัวอื่น สามารถส่งรูปมาได้เลยครับ")
        )

    # ----------------------------------------
    # กรณีที่ 2: ผู้ใช้กดปุ่ม "⏰ ตั้งเตือนกินยา"
    # ----------------------------------------
    elif data.startswith("action=set_reminder"):
        postback_dict = dict(parse_qsl(data))
        
        drug_name = postback_dict.get("drug", "ยาของคุณ")
        time_str = postback_dict.get("time", "")
        # 👇 รับค่าก่อน/หลังอาหารที่แอบส่งมา (ถ้าไม่มีให้เป็น after)
        meal_timing = postback_dict.get("timing", "after") 

        print(f"เตรียมบันทึกข้อมูลลง DB: User={user_id}, Drug={drug_name}, Time={time_str}, Timing={meal_timing}")

        is_morning = "morning" in time_str
        is_noon = "noon" in time_str
        is_evening = "evening" in time_str
        is_bedtime = "bedtime" in time_str

        try:
            user_check = supabase.table("user_profiles").select("line_uid").eq("line_uid", user_id).execute()
            if not user_check.data:
                supabase.table("user_profiles").insert({"line_uid": user_id}).execute()

            # 👇 เพิ่มคอลัมน์ meal_timing เข้าไปในข้อมูลที่จะบันทึก
            reminder_payload = {
                "line_uid": user_id,
                "drug_name": drug_name,
                "is_active": True,
                "morning": is_morning,
                "noon": is_noon,
                "evening": is_evening,
                "bedtime": is_bedtime,
                "meal_timing": meal_timing # 👈 บันทึกลง Supabase ตรงนี้
            }
            supabase.table("reminder_schedules").insert(reminder_payload).execute()

            reply_text = build_reminder_saved_reply(user_language, drug_name, meal_timing)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            return

            # สร้างข้อความแจ้งลูกค้าให้ชัดเจนขึ้น
            timing_th = "ก่อนอาหาร" if meal_timing == "before" else "หลังอาหาร"
            reply_text = f"⏰ ตั้งเวลาเตือนสำหรับยา {drug_name} ({timing_th}) ลงในระบบเรียบร้อยครับ!"
            
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            
        except Exception as e:
            print(f"Error saving reminder: {e}")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=t(user_language, "reminder_save_error")))
            return

            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ เกิดข้อผิดพลาดในการบันทึกข้อมูลลงฐานข้อมูล กรุณาลองใหม่อีกครั้ง"))
            
    # ----------------------------------------
    # กรณีที่ 3: ผู้ใช้กดปุ่มจาก Flex Message แจ้งเตือนกินยา
    # ----------------------------------------
    elif data.startswith("action=take_pill") or data.startswith("action=snooze") or data.startswith("action=stop_drug"):
        postback_dict = dict(parse_qsl(data))
        action = postback_dict.get("action")

        if action == "stop_drug":
            # 🎯 ลอจิก: ปิดการแจ้งเตือนยาทีละตัว (ทุกมื้อ)
            drug_name = postback_dict.get("drug", "")
            try:
                # อัปเดตให้ยาตัวนี้ is_active = False ในฐานข้อมูล
                supabase.table("reminder_schedules").update({"is_active": False}).eq("line_uid", user_id).eq("drug_name", drug_name).execute()
                
                reply_text = build_stop_drug_reply(user_language, drug_name)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
                print(f"✅ ยกเลิกการแจ้งเตือนยา {drug_name} ให้ผู้ใช้ {user_id} สำเร็จ")
                return

                reply_text = f"⏹️ ระบบได้บันทึกว่า {drug_name} หมดแล้ว และจะหยุดการแจ้งเตือนยารายการนี้ครับ"
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
                print(f"✅ ยกเลิกการแจ้งเตือนยา {drug_name} ให้ผู้ใช้ {user_id} สำเร็จ")
            except Exception as e:
                print(f"❌ Error stopping drug reminder: {e}")
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=t(user_language, "medicine_finished_error")))
                return

                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ เกิดข้อผิดพลาดในการยกเลิกแจ้งเตือนครับ"))

        elif action == "take_pill":
            # 🎯 ลอจิก: ตอบรับเมื่อกดกินยาทั้งหมด
            meal = postback_dict.get("meal", "")
            reply_text = build_take_pill_reply(user_language, meal)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            return

            meal_th = {"morning": "เช้า", "noon": "กลางวัน", "evening": "เย็น", "bedtime": "ก่อนนอน"}.get(meal, "")
            
            reply_text = f"✅ ยอดเยี่ยมมากครับ! บันทึกการทานยามื้อ{meal_th} เรียบร้อยแล้ว ขอให้สุขภาพแข็งแรงนะครับ 💙"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            
        elif action == "snooze":
            # 🎯 ลอจิก: ตอบรับการเลื่อน (เฟสนี้ใช้ข้อความตอบรับไปก่อน)
            reply_text = build_snooze_reply(user_language)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            return

            reply_text = f"💤 รับทราบครับ เลื่อนการแจ้งเตือนออกไป 15 นาที ถ้าพร้อมทานยาแล้ว อย่าลืมหยิบมาทานนะครับ"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

    # ----------------------------------------
    # กรณีที่ 4: ผู้ใช้กดเปลี่ยนเวลาจาก Datetime Picker
    # ----------------------------------------
    elif data.startswith("action=update_time"):
        postback_dict = dict(parse_qsl(data))
        meal = postback_dict.get("meal")
        
        # ดึงเวลาที่ User เลื่อนเลือกมาจาก params ของ LINE (จะได้ออกมาเป็น 'HH:MM' เช่น '08:30')
        selected_time = event.postback.params.get('time') if event.postback.params else None
        
        if selected_time:
            meal_col = f"default_{meal}" # แปลงเป็นชื่อคอลัมน์ เช่น default_morning
            meal_th = {"morning": "มื้อเช้า", "noon": "มื้อกลางวัน", "evening": "มื้อเย็น", "bedtime": "ก่อนนอน"}.get(meal, "")
            
            try:
                # เติมวินาทีให้ครบฟอร์แมต time ของ DB (HH:MM:SS)
                db_time = f"{selected_time}:00"
                
                # เช็คก่อนว่ามี Profile User คนนี้หรือยัง
                user_check = supabase.table("user_profiles").select("line_uid").eq("line_uid", user_id).execute()
                if not user_check.data:
                    # ถ้ายังไม่มีให้สร้างใหม่พร้อมเวลาที่เลือก
                    supabase.table("user_profiles").insert({"line_uid": user_id, meal_col: db_time}).execute()
                else:
                    # ถ้ามีแล้วให้อัปเดตเวลาทับลงไป
                    supabase.table("user_profiles").update({meal_col: db_time}).eq("line_uid", user_id).execute()

                reply_text = f"✅ บันทึกเวลาแจ้งเตือน {meal_th} เป็นเวลา {selected_time} น. เรียบร้อยครับ\nระบบจะใช้เวลานี้แจ้งเตือนคุณทุกวันครับ"
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
                print(f"✅ อัปเดตเวลา {meal_th} ให้ {user_id} เป็น {db_time}")

            except Exception as e:
                print(f"❌ Error updating time: {e}")
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ เกิดข้อผิดพลาดในการบันทึกเวลา กรุณาลองใหม่อีกครั้งครับ"))

# ==========================================
# เส้นทางสำหรับทดสอบ Database โดยเฉพาะ (ฉบับ Debug)
# ==========================================
@app.get("/test-db/{drug_name}")
def test_database_connection(drug_name: str):
    # 1. ปริ้นท์ค่า URL ออกมาดูใน Log ของ Render
    print(f"🔗 [DEBUG] SUPABASE_URL ของคุณคือ: '{SUPABASE_URL}'")
    
    if not supabase:
        return {"status": "error", "message": "ไม่ได้เชื่อมต่อ Supabase Client"}

    try:
        # 2. บังคับใช้ชื่อตาราง Medication_VQA
        print(f"🔍 [DEBUG] กำลังค้นหา: {drug_name} ในตาราง Medication_VQA")
        response = supabase.table('Medication_VQA').select('*').ilike('generic_name', f"%{drug_name}%").execute()
        
        if response.data and len(response.data) > 0:
            return {
                "status": "success", 
                "message": "เย้! ดึงข้อมูลสำเร็จแล้ว",
                "data": response.data[0]
            }
        else:
            return {
                "status": "not_found", 
                "message": f"เชื่อมต่อสำเร็จ แต่ไม่พบข้อมูลของยา '{drug_name}'"
            }
            
    except Exception as e:
        print(f"❌ [DEBUG] ERROR DETAIL: {str(e)}")
        return {"status": "error", "message": str(e)}

# ----------------------------------------
# ฟังก์ชันรับข้อความ (Text) และวิเคราะห์ Intent (NLP)
# ----------------------------------------
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_text = event.message.text
    user_id = event.source.user_id
    user_language = get_user_language(user_id)

    print(f"💬 ได้รับข้อความจาก {user_id}: {user_text}")

    if is_language_command(user_text):
        print(f"🌐 เปิดตัวเลือกภาษาให้ {user_id} จากข้อความ: {user_text!r}")
        line_bot_api.reply_message(
            event.reply_token,
            FlexSendMessage(
                alt_text=t(user_language, "language_picker_alt"),
                contents=build_language_picker(user_language),
            ),
        )
        return

    # 👇 [เพิ่มใหม่] ส่งสถานะ "กำลังพิมพ์..." ให้ฝั่งข้อความ
    url = "https://api.line.me/v2/bot/chat/loading/start"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    # สำหรับข้อความมักจะประมวลผลเร็วกว่ารูป ตั้งเผื่อไว้ที่ 20 วินาทีครับ
    data_loading = {"chatId": user_id, "loadingSeconds": 20}
    requests.post(url, headers=headers, json=data_loading)
    # 👆 สิ้นสุดส่วนที่เพิ่มใหม่

    # ==========================================
    # ⚡ [ดักจับพิเศษ] คำสั่งจาก Rich Menu
    # ==========================================
    if user_text == "เช็กรายการยา":
        try:
            # ดึงข้อมูลยาที่ยัง Active อยู่ของลูกค้ารายนี้
            res = supabase.table("reminder_schedules").select("drug_name, morning, noon, evening, bedtime").eq("line_uid", user_id).eq("is_active", True).execute()
            if res.data:
                reply_text = "💊 รายการยาที่คุณต้องทานปัจจุบันมีดังนี้ครับ:\n\n"
                for item in res.data:
                    meals = []
                    if item.get("morning"): meals.append("เช้า")
                    if item.get("noon"): meals.append("กลางวัน")
                    if item.get("evening"): meals.append("เย็น")
                    if item.get("bedtime"): meals.append("ก่อนนอน")
                    
                    meal_str = ", ".join(meals) if meals else "ไม่ระบุมื้อ"
                    reply_text += f"🔹 {item['drug_name']}\n   (มื้อ: {meal_str})\n"
                
                reply_text += "\nขอให้สุขภาพแข็งแรงนะครับ 💙"
            else:
                reply_text = "ตอนนี้คุณไม่มีรายการยาที่ตั้งเตือนไว้ครับ หากต้องการตั้งเตือนสามารถถ่ายรูปฉลากยาส่งมาได้เลยครับ 📸"
            
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        except Exception as e:
            print(f"❌ Error checking meds: {e}")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="ขออภัยครับ ไม่สามารถดึงข้อมูลรายการยาได้ในขณะนี้"))
        return # หยุดการทำงานตรงนี้ ไม่ต้องส่งไปหา AI

    elif user_text == "เปลี่ยนเวลาแจ้งเตือน":
        # สร้าง Flex Message ดึง Widget นาฬิกาของ LINE ขึ้นมาให้เลือก
        flex_time_picker = {
            "type": "bubble",
            "size": "kilo",
            "header": {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": "#1DB446",
                "contents": [
                    {"type": "text", "text": "⏰ ตั้งเวลาแจ้งเตือนใหม่", "weight": "bold", "color": "#FFFFFF", "size": "md"}
                ]
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "contents": [
                    {"type": "text", "text": "เลือกช่วงเวลาที่คุณต้องการให้ระบบเตือนกินยาครับ", "wrap": True, "size": "sm", "color": "#666666"},
                    {"type": "separator", "margin": "md"},
                    {
                        "type": "button",
                        "style": "secondary",
                        "height": "sm",
                        "margin": "sm",
                        "action": {
                            "type": "datetimepicker",
                            "label": "🌅 มื้อเช้า",
                            "data": "action=update_time&meal=morning",
                            "mode": "time"
                        }
                    },
                    {
                        "type": "button",
                        "style": "secondary",
                        "height": "sm",
                        "margin": "sm",
                        "action": {
                            "type": "datetimepicker",
                            "label": "☀️ มื้อกลางวัน",
                            "data": "action=update_time&meal=noon",
                            "mode": "time"
                        }
                    },
                    {
                        "type": "button",
                        "style": "secondary",
                        "height": "sm",
                        "margin": "sm",
                        "action": {
                            "type": "datetimepicker",
                            "label": "🌆 มื้อเย็น",
                            "data": "action=update_time&meal=evening",
                            "mode": "time"
                        }
                    },
                    {
                        "type": "button",
                        "style": "secondary",
                        "height": "sm",
                        "margin": "sm",
                        "action": {
                            "type": "datetimepicker",
                            "label": "🌙 ก่อนนอน",
                            "data": "action=update_time&meal=bedtime",
                            "mode": "time"
                        }
                    }
                ]
            }
        }
        line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text="ตั้งเวลาแจ้งเตือน", contents=flex_time_picker))
        return # หยุดการทำงานตรงนี้
    # ==========================================

    # 1. 🎯 สร้าง Prompt ให้ Gemini ช่วยแยกแยะเจตนา (Intent Classification)
    system_prompt = """
    คุณคือ AI ผู้ช่วยเภสัชกรประจำร้าน 'บ้านยาสุขใจ' 
    จงวิเคราะห์ข้อความของผู้ใช้และแยกแยะเจตนา (Intent) ออกมาเป็น 1 ใน 3 หมวดหมู่นี้เท่านั้น:
    1. MED_QUERY : คำถามเกี่ยวกับยา สุขภาพ อาการป่วย
    2. STORE_INFO : คำถามเกี่ยวกับร้าน เช่น เวลาเปิด-ปิด ที่อยู่ ติดต่อ
    3. GENERAL : การทักทายทั่วไป หรือเรื่องอื่นๆ ที่ไม่เกี่ยวกับข้างต้น
    
    กฎเหล็ก: ตอบกลับมาแค่ชื่อหมวดหมู่ภาษาอังกฤษ (เช่น MED_QUERY) ห้ามมีข้อความอื่นปนเด็ดขาด
    """
    
    try:
        # สร้างตัว client ขึ้นมาใหม่ โดยดึง API Key จาก Environment Variable
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        # 2. 🧠 เรียกใช้ Gemini Model แบบ Text
        # (ตรวจสอบให้แน่ใจว่าได้ประกาศ genai.configure(api_key=...) ไว้ด้านบนแล้ว)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[system_prompt, f"ข้อความผู้ใช้: {user_text}"]
        )
        
        # ตัดช่องว่างเผื่อ AI ตอบติด whitespace
        intent = response.text.strip().upper() 
        print(f"🧠 [NLP] วิเคราะห์ข้อความ -> Intent: {intent}")
        
        # 3. 🔀 Router: ส่งข้อความตอบกลับเบื้องต้นตาม Intent
        if "MED_QUERY" in intent:
            # =======================================================
            # ✂️ โค้ดชุดใหม่: Vector Search (Semantic RAG) ✂️
            # =======================================================
            print(f"🔍 [Vector Search] กำลังวิเคราะห์อาการ: {user_text}")
            
            try:
                database_search_query = build_database_search_query(client, user_text, user_language)
                print(f"🔎 [Vector Search] คำค้นสำหรับฐานข้อมูล: {database_search_query}")

                # 3.1 แปลงประโยคของลูกค้าให้เป็น Vector (อัปเดตโมเดลเป็น gemini-embedding-001)
                embed_res = client.models.embed_content(
                    model='gemini-embedding-001',
                    contents=database_search_query,
                    config=types.EmbedContentConfig(output_dimensionality=768) # 👈 บังคับให้เหลือ 768 มิติ
                )
                query_vector = embed_res.embeddings[0].values

                # 3.2 นำ Vector ไปค้นหาใน Supabase ผ่าน RPC ฟังก์ชันที่เราสร้างไว้
                db_res = supabase.rpc(
                    "match_symptoms", 
                    {
                        "query_embedding": query_vector,
                        "match_threshold": 0.4, # ปรับจูนได้: ค่ายิ่งใกล้ 1 ยิ่งต้องเหมือนเป๊ะ (แนะนำ 0.3 - 0.5)
                        "match_count": 3        # ดึงยาที่ตรงกับอาการมากที่สุดมา 3 อันดับแรก
                    }
                ).execute()
                
                records = db_res.data
                print(f"✅ [Vector Search] ดึงข้อมูลยาที่เกี่ยวข้องมาได้ {len(records)} รายการ")
                
                # 👈 เพิ่มบล็อกนี้เพื่อแอบดูคะแนนความเหมือนที่แท้จริง
                if records:
                    for r in records:
                        print(f"   -> [DEBUG] เจอข้อความของยา: {r.get('trade_name')} | ได้คะแนนความเหมือน: {r.get('similarity'):.4f}")
                        
            except Exception as e:
                print(f"❌ [Vector Search] Error: {e}")
                records = []
            # =======================================================
            
            # 👇 โค้ดส่วน 3.3 ด้านล่างนี้ (การสร้าง context_text) ปล่อยไว้เหมือนเดิมได้เลยครับ
            context_text = "ไม่พบข้อมูลยาที่ตรงกับคำถามในฐานข้อมูลร้าน"
            if records:
                context_texts = [f"- {r['trade_name']}: {r['rag_text']}" for r in records]
                context_text = "\n".join(context_texts)

            # 👇 เริ่มก๊อปปี้จากตรงนี้ไปวางทับ 👇
            # 3.3 นำข้อมูลที่เจอมาสร้าง Context ส่งให้ Gemini สรุปคำตอบ
            
            context_text = "ไม่พบข้อมูลยาที่ตรงกับคำถามในฐานข้อมูลร้าน"
            if records:
                context_texts = [f"- {r['trade_name']}: {r['rag_text']}" for r in records]
                context_text = "\n".join(context_texts)
                
            # สเต็ปที่ 1: สร้าง Prompt บังคับโครงสร้าง JSON
            language_instruction = build_language_instruction(user_language)
            final_prompt = f"""
            จากข้อมูลร้านยาต่อไปนี้: {context_text}
            จงตอบคำถามของลูกค้า: {user_text}
            {language_instruction}
            ข้อมูลจากฐานข้อมูลอาจเป็นภาษาไทย ให้แปลและสรุปเป็นภาษาของผู้ใช้ตามคำสั่งด้านบน
            ห้ามแปลชื่อยา trade name หรือ generic name แบบเดาสุ่ม

            กรุณาตอบกลับในรูปแบบ JSON เท่านั้น โดยใช้โครงสร้างดังนี้:
            {{
              "symptom": "สรุปอาการสั้นๆ (เช่น ปวดหัวจากความเครียด)",
              "advice": "คำแนะนำเบื้องต้น",
              "recommended_drug": "แสดงรายชื่อยาที่แนะนำ 'ทั้งหมด' จากข้อมูลร้านยา ห้ามตัดทิ้ง (ให้จัดเรียงเป็นข้อ 1. 2. 3. พร้อมบอกสรรพคุณสั้นๆ ในแต่ละข้อ, ถ้าไม่มีข้อมูลให้ใส่ค่าว่าง '')",
              "warning": "สรุปข้อควรระวังรวมของยาทั้งหมดที่แนะนำ (ถ้าไม่มีให้ใส่ค่าว่าง '')"
            }}
            """

            # สเต็ปที่ 2: สั่ง Gemini ให้ตอบกลับมาเป็น JSON
            final_res = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=[final_prompt],
                config={"response_mime_type": "application/json"}
            )

            # สเต็ปที่ 3: ประกอบร่าง Flex Message แบบ Dynamic
            try:
                # เพิ่มระบบเคลียร์ Markdown เผื่อ AI แถมมา
                clean_json_text = final_res.text.strip().replace("```json", "").replace("```", "").strip()
                ai_data = json.loads(clean_json_text)
                flex_rag_reply = build_rag_flex_reply(user_language, ai_data)
                
                reply_or_push_message(
                    line_bot_api,
                    user_id,
                    event.reply_token,
                    FlexSendMessage(alt_text=t(user_language, "rag_alt_text"), contents=flex_rag_reply),
                )

            except Exception as e:
                print(f"❌ Error parsing JSON or building Flex: {e}")
                reply_or_push_message(
                    line_bot_api,
                    user_id,
                    event.reply_token,
                    TextSendMessage(text=t(user_language, "ai_format_error")),
                )

    # 👇 การย่อหน้าตรงนี้แหละครับที่ถูกต้อง! มันต้องออกมาระดับเดียวกับ try ด้านบนสุด
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in text message NLP: {error_msg}")
        
        if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg:
            reply_text = t(user_language, "generic_processing_error")
        else:
            reply_text = t(user_language, "generic_processing_error")
            
        reply_or_push_message(
            line_bot_api,
            user_id,
            event.reply_token,
            TextSendMessage(text=reply_text),
        )
# ==========================================
# ⚡ ดักจับข้อความประเภทอื่นๆ (Edge Cases & Error Handling)
# ==========================================
@handler.add(MessageEvent, message=(StickerMessage, VideoMessage, AudioMessage, LocationMessage, FileMessage))
def handle_other_messages(event):
    # กำหนดข้อความตอบกลับเมื่อลูกค้าส่งสิ่งที่ไม่รองรับเข้ามา
    user_language = get_user_language(event.source.user_id)
    reply_text = t(user_language, "unsupported_message_type")
    
    try:
        line_bot_api.reply_message(
            event.reply_token, 
            TextSendMessage(text=reply_text)
        )
        print("✅ [EDGE CASE] ตอบกลับข้อความที่ไม่รองรับสำเร็จ")
    except Exception as e:
        print(f"❌ [EDGE CASE] Error: {e}")
