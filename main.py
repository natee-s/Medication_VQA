from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    ImageMessage,
    FlexSendMessage,
    PostbackEvent,
)
import os
import cv2
import numpy as np
from google import genai
from google.genai import types
import json
import requests
from supabase import create_client, Client
from urllib.parse import parse_qsl

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

@app.get("/")
def root():
    return {"message": "Banya Sookjai AI Server is running!"}

@app.get("/cron/check-reminder")
def check_reminder():
    # 📌 ในอนาคตเราจะเขียนโค้ดตรงนี้เพื่อ:
    # 1. ค้นหา Database ว่ามีผู้ใช้คนไหนถึงเวลากินยาหรือยัง
    # 2. ใช้ line_bot_api.push_message() ส่งแจ้งเตือน
    
    print("🔔 Cron job is triggered! Server is awake & Checking reminders...")
    
    return {"status": "success", "message": "Cron job executed successfully. Server is active."}

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
# 2.1. การตั้งค่าเชื่อมต่อฐานข้อมูล Supabase
# ==========================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# ตรวจสอบเบื้องต้นว่ามีการตั้งค่า Key หรือยัง
if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ Warning: SUPABASE_URL หรือ SUPABASE_KEY ยังไม่ได้ตั้งค่าใน Environment Variables")

# สร้าง Client สำหรับเชื่อมต่อฐานข้อมูล
try:
    # จะเริ่มสร้าง client ก็ต่อเมื่อมีค่าครบทั้งสองตัว
    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ เชื่อมต่อ Supabase สำเร็จ!")
    else:
        supabase = None
except Exception as e:
    print(f"❌ เชื่อมต่อ Supabase ไม่สำเร็จ: {e}")
    supabase = None

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
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # --- ส่งสถานะ "กำลังพิมพ์..." (Loading Animation) ---
    url = "https://api.line.me/v2/bot/chat/loading/start"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    data_loading = {"chatId": event.source.user_id, "loadingSeconds": 10}
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
}"""
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
                    TextSendMessage(text="📸 รูปภาพตะแคงหรือกลับหัวครับ กรุณาถ่ายฉลากยาให้ตั้งตรงแล้วส่งมาใหม่อีกครั้งนะครับ")
                )
                return

            search_keyword = data.get("search_keyword")
            if not search_keyword:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="⚠️ ระบบอ่านชื่อยาบนฉลากไม่ชัดเจน กรุณาถ่ายใหม่อีกครั้งครับ"))
                return

            # ----------------------------------------------------
            # 🎯 เริ่มกระบวนการ RAG (ค้นหาชื่อยาใน Supabase)
            # ----------------------------------------------------
            db_data = search_medicine_in_db(search_keyword)

            if not db_data:
                line_bot_api.reply_message(
                    event.reply_token, 
                    TextSendMessage(text=f"🔍 ระบบสกัดชื่อยาได้ว่า '{search_keyword}' แต่ไม่พบข้อมูลนี้ในฐานข้อมูลครับ")
                )
                return
            
            # จัดเตรียมข้อมูลใส่ Flex Message
            trade_name = db_data.get('trade_name') or 'ไม่ระบุ'
            generic_name = db_data.get('generic_name') or 'ไม่ระบุ'
            indication = db_data.get('indication') or 'ไม่ระบุ'
            dosage = db_data.get('dosage_frequency') or 'ไม่ระบุ'
            instruction = db_data.get('instruction_time') or 'ไม่ระบุ'
            warning = db_data.get('precaution') or 'ไม่มี'

            flex_bubble = {
                "type": "bubble",
                "size": "mega",
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": "#1DB446",
                    "contents": [
                        {"type": "text", "text": "💊 ข้อมูลฉลากยา", "weight": "bold", "size": "lg", "color": "#FFFFFF"}
                    ]
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "spacing": "md",
                    "contents": [
                        {"type": "text", "text": f"ชื่อการค้า: {trade_name}", "weight": "bold", "wrap": True},
                        {"type": "text", "text": f"ชื่อยา: {generic_name}", "color": "#666666", "size": "sm", "wrap": True},
                        {"type": "separator", "margin": "md"},
                        {"type": "text", "text": f"🎯 ข้อบ่งใช้: {indication}", "wrap": True},
                        {"type": "text", "text": f"⚖️ ขนาดยา: {dosage}", "wrap": True},
                        {"type": "text", "text": f"⏱️ วิธีใช้: {instruction}", "weight": "bold", "color": "#E03131", "wrap": True},
                        {"type": "text", "text": f"⚠️ คำเตือน: {warning}", "size": "sm", "color": "#FFA500", "wrap": True}
                    ]
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
                            "action": {"type": "postback", "label": "⏰ ตั้งเตือนกินยา", "data": f"action=set_reminder&drug={generic_name}"}
                        },
                        {
                            "type": "button",
                            "style": "secondary",
                            "height": "sm",
                            "action": {"type": "postback", "label": "✅ รับทราบ", "data": "action=acknowledge"}
                        }
                    ]
                }
            }

            line_bot_api.reply_message(
                event.reply_token,
                FlexSendMessage(alt_text=f"ข้อมูลยา: {generic_name}", contents=flex_bubble)
            )

        except json.JSONDecodeError:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"พบปัญหาในการจัดรูปแบบข้อมูลจาก AI:\n{raw_text}"))
            
    except Exception as e:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"⚠️ เกิดข้อผิดพลาดในระบบประมวลผล: {str(e)}"))    

@handler.add(PostbackEvent)
def handle_postback(event):
    data = event.postback.data
    user_id = event.source.user_id

    # ----------------------------------------
    # กรณีที่ 1: ผู้ใช้กดปุ่ม "✅ รับทราบ"
    # ----------------------------------------
    if data == "action=acknowledge":
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="✅ ระบบรับทราบเรียบร้อยครับ คุณสามารถพิมพ์สอบถามข้อมูลเกี่ยวกับยานี้เพิ่มเติมได้เลยครับ หรือหากต้องการให้อ่านฉลากยาตัวอื่น สามารถส่งรูปมาได้เลยครับ")
        )

    # ----------------------------------------
    # กรณีที่ 2: ผู้ใช้กดปุ่ม "⏰ ตั้งเตือนกินยา"
    # ----------------------------------------
    elif data.startswith("action=set_reminder"):
        # สกัดเอาชื่อยาออกมาจาก payload
        parts = data.split("&drug=")
        drug_name = parts[1] if len(parts) > 1 else "ยาของคุณ"

        print(f"เตรียมบันทึกข้อมูลลง DB: User={user_id}, Drug={drug_name}")

        try:
            # 1. เช็คก่อนว่ามี User คนนี้ในตาราง user_profiles หรือยัง
            user_check = supabase.table("user_profiles").select("line_uid").eq("line_uid", user_id).execute()
            if not user_check.data:
                # ถ้ายังไม่มี ให้สร้างโปรไฟล์ใหม่ (เวลา default จะถูกสร้างให้อัตโนมัติใน DB)
                supabase.table("user_profiles").insert({"line_uid": user_id}).execute()

            # 2. บันทึกข้อมูลการตั้งเตือนลงตาราง reminder_schedules
            reminder_payload = {
                "line_uid": user_id,
                "drug_name": drug_name,
                "is_active": True
            }
            supabase.table("reminder_schedules").insert(reminder_payload).execute()

            # 3. ตอบกลับผู้ใช้เมื่อบันทึกสำเร็จ
            reply_text = f"⏰ ตั้งเวลาเตือนสำหรับยา {drug_name} ลงในระบบเรียบร้อยครับ!"
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=reply_text)
            )
            
        except Exception as e:
            print(f"Error saving reminder: {e}")
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="❌ เกิดข้อผิดพลาดในการบันทึกข้อมูลลงฐานข้อมูล กรุณาลองใหม่อีกครั้ง")
            )

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

# ==========================================
# 4. ฟังก์ชันจัดการเมื่อผู้ใช้ส่งข้อความ (Text) เข้ามา
# ==========================================
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_text = event.message.text
    
    # ดักจับข้อความพื้นฐานเบื้องต้น
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text=f"คุณพิมพ์มาว่า: '{user_text}'\n\n📸 ขณะนี้ระบบบอทยังอยู่ในช่วงทดสอบ ฟังก์ชันหลักคือการอ่าน 'รูปภาพฉลากยา' รบกวนส่งรูปฉลากยามาให้ผมวิเคราะห์ได้เลยนะครับ!"
        )
    )