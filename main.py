from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage 
import os
import cv2
import numpy as np
import base64
import requests
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
# 2. การตั้งค่าเซิร์ฟเวอร์, LINE Bot และ OpenRouter
# ==========================================
app = FastAPI()

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN', 'YOUR_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET', 'YOUR_SECRET')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', 'YOUR_OPENROUTER_KEY')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

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
    reply_text = f"คุณพิมพ์มาว่า: {event.message.text}\n(ขณะนี้ระบบกำลังทดสอบฟังก์ชันอ่านภาพฉลากยา ลองส่งรูปเข้ามาได้เลยครับ)"
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id
    
    # 3.1 รับและล้างภาพ
    image_content = line_bot_api.get_message_content(message_id)
    file_path = f"/tmp/{message_id}.jpg"
    processed_path = f"/tmp/processed_{message_id}.jpg"
    
    with open(file_path, 'wb') as fd:
        for chunk in image_content.iter_content():
            fd.write(chunk)
            
    width, height = process_pharmacy_label(file_path, processed_path)
    
    # 3.2 แปลงภาพที่ล้างแล้วเป็น Base64 เพื่อส่งให้ OpenRouter
    with open(processed_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 3.3 เตรียมคำสั่ง (Prompt) และเรียกใช้งาน Qwen-VL ผ่าน OpenRouter
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "qwen/qwen2.5-vl-72b-instruct", 
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "นี่คือภาพฉลากยา กรุณาดึงข้อมูลและสรุปออกมาเป็น: 1. ชื่อยา 2. ขนาดยา 3. วิธีรับประทาน 4. คำเตือน (ถ้ามี)"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_string}"
                        }
                    }
                ]
            }
        ]
    }
    
    # 3.4 ยิงข้อมูลไปให้ AI และรับผลลัพธ์
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        
        # ตรวจสอบว่ามี Error จาก OpenRouter หรือไม่
        if "error" in response_json:
            draft_answer = f"⚠️ เกิดข้อผิดพลาดจาก OpenRouter: {response_json['error']['message']}"
        else:
            draft_answer = response_json['choices'][0]['message']['content']
            
    except Exception as e:
        draft_answer = f"⚠️ ไม่สามารถเชื่อมต่อ AI ได้: {str(e)}"
    
    # 3.5 ส่งคำตอบ (Draft Answer) กลับไปที่ LINE
    final_reply = f"✅ ประมวลผลภาพเสร็จสิ้น (Pipeline B)\n\nผลการอ่านจาก Qwen-VL:\n{draft_answer}"
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=final_reply)
    )