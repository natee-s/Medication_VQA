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

@app.get("/")
def root():
    return {"message": "Banya Sookjai AI Server is running!"}

@app.get("/cron/check-reminder")
def check_reminder():
    if not supabase:
        return {"status": "error", "message": "Supabase not connected"}

    # 1. ดึงเวลาปัจจุบันของประเทศไทย (UTC+7)
    bkk_tz = pytz.timezone('Asia/Bangkok')
    now_bkk = datetime.now(bkk_tz)
    current_time = now_bkk.strftime("%H:%M") # จะได้ออกมาเป็นข้อความ เช่น '08:30'
    #print(f"⏰ [CRON] รันระบบตรวจสอบแจ้งเตือนเวลา: {current_time} น.")

    try:
        # 2. ค้นหาเฉพาะลูกค้าที่ถึงเวลากินยาในนาทีนี้ (ลดภาระ Database)
        current_time_db = f"{current_time}:00" # เติมวินาทีให้ตรงกับฟอร์แมตเวลาใน DB (เช่น 08:00:00)
        
        # 2.1 สร้างเงื่อนไขค้นหาเวลาที่ระบุไว้
        or_conditions = [
            f"default_morning.eq.{current_time_db}",
            f"default_noon.eq.{current_time_db}",
            f"default_evening.eq.{current_time_db}",
            f"default_bedtime.eq.{current_time_db}"
        ]
        
        # 2.2 ดักจับคนที่เป็นค่าว่าง (NULL) ให้ใช้เวลามาตรฐานของร้าน
        if current_time == "08:00": or_conditions.append("default_morning.is.null")
        if current_time == "12:00": or_conditions.append("default_noon.is.null")
        if current_time == "18:00": or_conditions.append("default_evening.is.null")
        if current_time == "21:00": or_conditions.append("default_bedtime.is.null")
        
        # 2.3 รวมเงื่อนไขทั้งหมดเข้าด้วยกัน
        query_string = ",".join(or_conditions)
        
        # 2.4 ยิงคำสั่งให้ Supabase กรองมาให้เลย
        users_res = supabase.table("user_profiles").select("*").or_(query_string).execute()
        users = users_res.data
        
        count_messages_sent = 0

        for user in users:
            uid = user.get("line_uid")
            
            # แปลงเวลาจาก DB (เช่น '08:00:00') ให้เหลือแค่ '08:00' เพื่อเอามาเทียบ
            t_morning = str(user.get("default_morning"))[:5] if user.get("default_morning") else "08:00"
            t_noon = str(user.get("default_noon"))[:5] if user.get("default_noon") else "12:00"
            t_evening = str(user.get("default_evening"))[:5] if user.get("default_evening") else "18:00"
            t_bedtime = str(user.get("default_bedtime"))[:5] if user.get("default_bedtime") else "21:00"

            meal_to_take = None
            meal_name_th = ""

            # 3. ตรวจสอบว่าเวลาปัจจุบัน ตรงกับมื้อไหนของผู้ใช้คนนี้หรือไม่?
            if current_time == t_morning:
                meal_to_take = "morning"
                meal_name_th = "หลังอาหารเช้า 🌅"
            elif current_time == t_noon:
                meal_to_take = "noon"
                meal_name_th = "หลังอาหารกลางวัน ☀️"
            elif current_time == t_evening:
                meal_to_take = "evening"
                meal_name_th = "หลังอาหารเย็น 🌆"
            elif current_time == t_bedtime:
                meal_to_take = "bedtime"
                meal_name_th = "ก่อนนอน 🌙"

            # 4. ถ้าเวลาตรงกับมื้อยา ให้ดึงรายชื่อยาที่ต้องกินออกมารวมกัน
            if meal_to_take:
                print(f"🔍 พบผู้ใช้ {uid} ถึงเวลากินมื้อ {meal_to_take}")
                
                # ค้นหายาที่เป็น Active และตรงกับมื้อนั้นๆ
                reminders_res = supabase.table("reminder_schedules").select("drug_name").eq("line_uid", uid).eq("is_active", True).eq(meal_to_take, True).execute()
                drugs = reminders_res.data
                
                if drugs:
                    # 🎯 สร้างกล่องรายการยาแบบไดนามิก (มีปุ่ม "ยาหมด" แต่ละตัว)
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
                                    "type": "text",
                                    "text": f"💊 {drug_name}",
                                    "size": "sm",
                                    "weight": "bold",
                                    "color": "#333333",
                                    "gravity": "center",
                                    "wrap": True,
                                    "flex": 2 # ให้พื้นที่ชื่อยา 2 ส่วน
                                },
                                {
                                    "type": "button",
                                    "style": "secondary",
                                    "height": "sm",
                                    "flex": 1, # ให้พื้นที่ปุ่ม 1 ส่วน
                                    "action": {
                                        "type": "postback", 
                                        "label": "ยาหมด", 
                                        # ส่งข้อมูลไปว่าต้องการหยุดยาตัวไหน
                                        "data": f"action=stop_drug&drug={drug_name}"
                                    }
                                }
                            ]
                        })

                    # 5. ส่งแจ้งเตือนแบบ Flex Message โฉมใหม่
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
                            # นำส่วนหัว มาบวกกับ รายการยาที่วนลูปไว้ด้านบน
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
                                    "type": "button",
                                    "style": "primary",
                                    "color": "#1DB446",
                                    "height": "sm",
                                    "action": {"type": "postback", "label": "✅ กินยาทั้งหมดแล้ว", "data": f"action=take_pill&meal={meal_to_take}"}
                                },
                                {
                                    "type": "button",
                                    "style": "secondary",
                                    "height": "sm",
                                    "action": {"type": "postback", "label": "💤 เลื่อน 15 นาที", "data": f"action=snooze&meal={meal_to_take}"}
                                }
                            ]
                        }
                    }

                    line_bot_api.push_message(
                        uid, 
                        FlexSendMessage(alt_text=f"เตือนกินยา: {meal_name_th}", contents=flex_alert)
                    )
                    print(f"✅ ส่ง Flex Message แจ้งเตือนให้ {uid} สำเร็จ (ยา {len(drugs)} รายการ)")
                    count_messages_sent += 1
        if count_messages_sent > 0:
            print(f"🎉 [CRON-SUCCESS] เวลา {current_time} น. | ส่งแจ้งเตือนกินยาสำเร็จ {count_messages_sent} รายการ")

        return {"status": "success", "message": f"เช็กเวลา {current_time} น. สำเร็จ ส่งแจ้งเตือนไป {count_messages_sent} รายการ"}

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

            # ----------------------------------------------------
            # 🎯 เพิ่มลอจิกวิเคราะห์เวลากินยาจากข้อความ instruction
            # ----------------------------------------------------
            time_list = []
            if instruction != 'ไม่ระบุ':
                if 'เช้า' in instruction: time_list.append('morning')
                if 'กลางวัน' in instruction or 'เที่ยง' in instruction: time_list.append('noon')
                if 'เย็น' in instruction: time_list.append('evening')
                if 'นอน' in instruction: time_list.append('bedtime')
            
            # รวมเป็น text เช่น "morning,bedtime" ถ้าไม่มีให้ส่ง "none"
            time_payload = ",".join(time_list) if time_list else "none"

            # 👇 [เพิ่มใหม่] ลอจิกตรวจสอบ ก่อนอาหาร หรือ หลังอาหาร 👇
            meal_timing = "after" # ตั้งค่าเริ่มต้นให้เป็น 'หลังอาหาร' ไว้ก่อน
            if 'ก่อนอาหาร' in instruction or 'ก่อน' in instruction:
                meal_timing = "before"
            
            print(f"🔍 [DEBUG] ข้อความวิธีใช้จาก DB: {instruction}")
            print(f"🔍 [DEBUG] Time Payload: {time_payload} | Timing: {meal_timing}")
            # ----------------------------------------------------

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
                            "action": {"type": "postback", "label": "⏰ ตั้งเตือนกินยา", "data": f"action=set_reminder&drug={generic_name}&time={time_payload}&timing={meal_timing}"}
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

            # สร้างข้อความแจ้งลูกค้าให้ชัดเจนขึ้น
            timing_th = "ก่อนอาหาร" if meal_timing == "before" else "หลังอาหาร"
            reply_text = f"⏰ ตั้งเวลาเตือนสำหรับยา {drug_name} ({timing_th}) ลงในระบบเรียบร้อยครับ!"
            
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            
        except Exception as e:
            print(f"Error saving reminder: {e}")
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
                
                reply_text = f"⏹️ ระบบได้บันทึกว่า {drug_name} หมดแล้ว และจะหยุดการแจ้งเตือนยารายการนี้ครับ"
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
                print(f"✅ ยกเลิกการแจ้งเตือนยา {drug_name} ให้ผู้ใช้ {user_id} สำเร็จ")
            except Exception as e:
                print(f"❌ Error stopping drug reminder: {e}")
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ เกิดข้อผิดพลาดในการยกเลิกแจ้งเตือนครับ"))

        elif action == "take_pill":
            # 🎯 ลอจิก: ตอบรับเมื่อกดกินยาทั้งหมด
            meal = postback_dict.get("meal", "")
            meal_th = {"morning": "เช้า", "noon": "กลางวัน", "evening": "เย็น", "bedtime": "ก่อนนอน"}.get(meal, "")
            
            reply_text = f"✅ ยอดเยี่ยมมากครับ! บันทึกการทานยามื้อ{meal_th} เรียบร้อยแล้ว ขอให้สุขภาพแข็งแรงนะครับ 💙"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            
        elif action == "snooze":
            # 🎯 ลอจิก: ตอบรับการเลื่อน (เฟสนี้ใช้ข้อความตอบรับไปก่อน)
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

    print(f"💬 ได้รับข้อความจาก {user_id}: {user_text}")

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
            # 3.1 สกัดคำค้นหา (Keyword) แบบกว้าง (Broad Match)
            extract_prompt = """
            จงดึง 'คำหลักสั้นๆ' ที่เป็นอาการป่วย หรือ ชื่อยา จากข้อความของผู้ใช้ เพื่อนำไปค้นหาในฐานข้อมูล
            
            ข้อกำหนดสำคัญ (ต้องทำตามอย่างเคร่งครัด):
            1. หมวดอาการปวด: ไม่ว่าผู้ใช้จะพิมพ์ว่า ปวดหัว, ปวดเข่า, ปวดท้อง, ปวดฟัน, ปวดหลัง ให้สกัดเหลือแค่คำว่า "ปวด" คำเดียวเท่านั้น
            2. หมวดภูมิแพ้: คัดจมูก, น้ำมูกไหล ให้สกัดเหลือแค่ "น้ำมูก"
            3. หมวดไข้: ตัวร้อน, เป็นไข้ ให้สกัดเหลือแค่ "ไข้"
            4. หมวดไอ: ไอแห้ง, มีเสมหะ ให้สกัดเหลือแค่ "ไอ" หรือ "เสมหะ"
            
            กฎเหล็ก: ตอบแค่คำหลัก 'เพียงคำเดียว' สั้นๆ ห้ามมีประโยคอื่น ห้ามมีเครื่องหมาย
            """
            keyword_res = client.models.generate_content(
                model='gemini-2.5-flash', # ⚠️ แก้เป็น 2.5 ให้ตรงกับระบบหลัก
                contents=[extract_prompt, f"ข้อความ: {user_text}"]
            )
            # เคลียร์เครื่องหมายหรือช่องว่างแปลกๆ ที่ AI อาจแถมมา
            keyword = keyword_res.text.strip().replace("*", "").replace('"', "").replace("'", "")
            print(f"🔍 [RAG] Keyword สำหรับค้นหา: {keyword}")

            # 3.2 ค้นหาข้อมูลควบทั้งคอลัมน์ rag_text และ indication ด้วยคำสั่ง .or_()
            db_res = (
                supabase.table("Medication_VQA")
                .select("trade_name, rag_text")
                .or_(f"rag_text.ilike.%{keyword}%,indication.ilike.%{keyword}%")
                .execute()
            )
            records = db_res.data

            if records:
                # 3.3 นำข้อมูลที่เจอมาสร้าง Context ส่งให้ Gemini สรุปคำตอบ
                context_texts = [f"- {r['trade_name']}: {r['rag_text']}" for r in records]
                context_str = "\n".join(context_texts)
                
                final_prompt = f"""
                คุณคือ AI ผู้ช่วยเภสัชกรประจำร้าน 'บ้านยาสุขใจ'
                จงตอบคำถามลูกค้าโดยอ้างอิงจากข้อมูล (Context) ด้านล่างนี้เท่านั้น
                
                ข้อมูลอ้างอิงจากร้านยา:
                {context_str}
                
                คำถามของลูกค้า: {user_text}
                
                กฎเหล็ก: 
                - ตอบด้วยความสุภาพ เป็นกันเอง สั้นกระชับเข้าใจง่าย
                - ห้ามแนะนำยาหรือวิธีการรักษาที่ 'ไม่มี' ในข้อมูลอ้างอิงเด็ดขาด ถ้าข้อมูลไม่พอให้บอกว่าแนะนำให้ปรึกษาเภสัชกรหน้าร้าน
                """
                
                final_res = client.models.generate_content(
                    model='gemini-2.5-flash', # ⚠️ แก้เป็น 2.5 ให้ตรงกับระบบหลัก
                    contents=[final_prompt]
                )
                reply_text = final_res.text.strip()
                print("✅ [RAG] สร้างคำตอบจากฐานข้อมูลสำเร็จ!")
            else:
                reply_text = f"ขออภัยครับคุณลูกค้า จากข้อมูลของร้านบ้านยาสุขใจ ตอนนี้ผมยังไม่พบข้อมูลที่เกี่ยวกับ '{keyword}' ครับ รบกวนทักสอบถามเภสัชกรที่หน้าร้านได้เลยนะครับ 👨‍⚕️"
            
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            
        elif "STORE_INFO" in intent:
            reply_text = "🏠 ร้านบ้านยาสุขใจ ตั้งอยู่ที่ อ.หนองแค จ.สระบุรี เปิดให้บริการทุกวันครับ สอบถามเส้นทางเพิ่มเติมแจ้งได้เลยครับ"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            
        else:
            reply_text = "สวัสดีครับ บ้านยาสุขใจยินดีให้บริการครับ วันนี้มีอะไรให้ผมช่วยดูแลไหมครับ? 😊"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
        
    except Exception as e:
        print(f"❌ Error in text message NLP: {e}")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="ขออภัยครับ ตอนนี้ระบบคัดกรองข้อความขัดข้องชั่วคราวครับ"))