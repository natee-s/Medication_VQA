from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage 
import os
import cv2
import numpy as np
from google import genai
from google.genai import types
import json

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

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ประกาศเรียกใช้งาน Client ของ Gemini ด้วยรหัสคีย์ที่เราฝากไว้บน Render
ai_client = genai.Client(api_key=GEMINI_API_KEY)

@app.get("/")
def root():
    return {"message": "Banya Sookjai AI Server is running!"}

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
# 3. ระบบจัดการข้อความ (Text & Image)
# ==========================================

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    reply_text = f"คุณพิมพ์มาว่า: {event.message.text}\n(ขณะนี้ระบบกำลังทดสอบฟังก์ชันอ่านภาพฉลากยาด้วย Gemini 1.5 Flash)"
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id
    
    # 3.1 รับและล้างภาพด้วย OpenCV
    image_content = line_bot_api.get_message_content(message_id)
    file_path = f"/tmp/{message_id}.jpg"
    processed_path = f"/tmp/processed_{message_id}.jpg"
    
    with open(file_path, 'wb') as fd:
        for chunk in image_content.iter_content():
            fd.write(chunk)
            
    width, height = process_pharmacy_label(file_path, processed_path)
    
    # 3.2 อ่านไฟล์ภาพที่ล้างเสร็จแล้วเข้ามาในรูปแบบ Binary เพื่อส่งให้ Gemini
    with open(processed_path, "rb") as f:
        image_bytes = f.read()
        
    # 3.3 เรียกใช้งาน Gemini (ยิงตรงผ่าน SDK รวดเร็วและเสถียร)
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                """นี่คือภาพฉลากยา กรุณาดึงข้อมูลและส่งกลับมาเป็นรูปแบบ JSON (ไม่ต้องมี Markdown หรือ Code block คร่อม) โดยใช้โครงสร้างดังนี้:
            {
            "trade_name": "ชื่อทางการค้า หรือ ระบุ null หากไม่พบ",
            "generic_name": "ชื่อยา หรือ ระบุ null หากไม่พบ",
            "indication": "ข้อบ่งใช้ หรือ สรรพคุณ หรือ ระบุ null หากไม่พบ", 
            "dosage": "ขนาดยา หรือ ระบุ null หากไม่พบ",
            "instruction": "วิธีรับประทาน หรือ ระบุ null หากไม่พบ",
            "warning": "คำเตือน หรือ ระบุ null หากไม่พบ"
            }
            ห้ามมีข้อความอธิบายใดๆ เพิ่มเติม นอกเหนือจาก JSON object นี้"""
            ]
        )
        
        # 1. รับข้อความผลลัพธ์จาก Gemini
        raw_text = response.text.strip()
        
        # 2. ดักจับและลบ Markdown Code Block (เผื่อ AI แอบใส่ ```json มาคลุม)
        if raw_text.startswith('```json'):
            raw_text = raw_text.replace('```json', '').replace('```', '').strip()
        elif raw_text.startswith('```'):
            raw_text = raw_text.replace('```', '').strip()
            
        try:
            # 3. แปลงข้อความ JSON ให้กลายเป็น Python Dictionary
            data = json.loads(raw_text)
            
            # 4. จัดรูปแบบข้อความใหม่ให้สวยงามสำหรับแสดงใน LINE
            trade_name = data.get('trade_name') or 'ไม่ระบุ'
            generic_name = data.get('generic_name') or 'ไม่ระบุ'
            
            # (แก้) ปรับลบ slash (/) ที่เกินมาตรง trade_name ออก
            reply_message = f"""1. **ชื่อทางการค้า:**
    * {trade_name}
    
2. **ชื่อยา:**
    * {generic_name}

3. **ข้อบ่งใช้:**
    * {data.get('indication', 'ไม่ระบุ')}

4. **ขนาดยา:**
    * {data.get('dosage', 'ไม่ระบุ')}

5. **วิธีรับประทาน:**
    * {data.get('instruction', 'ไม่ระบุ')}

6. **คำเตือน:**
    * {data.get('warning', 'ไม่มี')}"""

        except json.JSONDecodeError:
            # กรณี Error ไม่เป็น JSON
            reply_message = f"พบปัญหาในการจัดรูปแบบข้อมูล:\n{raw_text}"
            
    except Exception as e:
        reply_message = f"⚠️ ไม่สามารถเชื่อมต่อสมอง AI Gemini ได้: {str(e)}"
    
    # 3.4 ส่งคำตอบกลับไปแสดงผลบนหน้าจอ LINE
    # (แก้) เปลี่ยนมารับค่าจาก reply_message
    final_reply = f"✅ ประมวลผลภาพเสร็จสิ้น (Pipeline B - Temporary Mode)\n\nผลการวิเคราะห์จาก Gemini:\n{reply_message}"
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=final_reply)
    )