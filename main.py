from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
import os
import cv2 # Image Preprocessing
import numpy as np #สำหรับการทำคณิตศาสตร์เมทริกซ์ภาพ

# ==========================================
# 1. ฟังก์ชันผู้เชี่ยวชาญการล้างภาพ (Image Preprocessing)
# ==========================================
def process_pharmacy_label(input_path, output_path):
    # อ่านภาพต้นฉบับ
    img = cv2.imread(input_path)
    
    # แก้ปัญหา "ถ่ายห่าง": ขยายภาพ (Upscale) ถ้ารูปมีขนาดเล็กไป
    height, width = img.shape[:2]
    if width < 1000:
        scale = 1000 / width
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
    # แปลงภาพเป็นสีขาวดำ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # แก้ปัญหา "แสงน้อย + รอยยับ": ใช้เทคนิค CLAHE ปรับสมดุลแสงเงาเฉพาะจุด
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    balanced = clahe.apply(gray)
    
    # แก้ปัญหา "รูปเบลอ": ใช้ Kernel ดึงขอบตัวหนังสือให้คมชัด
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(balanced, -1, kernel)
    
    # ลดจุดรบกวน (Noise) ที่เกิดจากการดึงความคมชัด
    denoised = cv2.medianBlur(sharpened, 3)
    
    # ตัดพื้นหลังทิ้ง (Adaptive Thresholding) ให้เหลือแต่หมึกดำบนกระดาษขาว
    processed = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # บันทึกภาพที่ล้างเสร็จแล้ว
    cv2.imwrite(output_path, processed)
    
    # ส่งคืนค่ากว้าง/ยาวเพื่อเอาไปทำรายงาน
    h, w = processed.shape
    return w, h


# ==========================================
# 2. การตั้งค่าเซิร์ฟเวอร์และ LINE Bot
# ==========================================
app = FastAPI()

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN', 'YOUR_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET', 'YOUR_SECRET')

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

# โค้ดส่วนจัดการข้อความ Text
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    reply_text = f"คุณพิมพ์มาว่า: {event.message.text}\n(ระบบ ฉลากฉลาด กำลังทดสอบระบบอยู่ เตรียมตัวพบกับของดีได้เลย)"
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

# โค้ดส่วนจัดการรูปภาพ Image
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id
    
    # ดึงไฟล์รูปภาพจากเซิร์ฟเวอร์ LINE
    image_content = line_bot_api.get_message_content(message_id)
    file_path = f"/tmp/{message_id}.jpg"
    processed_path = f"/tmp/processed_{message_id}.jpg"
    
    # บันทึกรูปต้นฉบับ
    with open(file_path, 'wb') as fd:
        for chunk in image_content.iter_content():
            fd.write(chunk)
            
    # เรียกใช้ฟังก์ชันล้างภาพที่เราเขียนไว้
    width, height = process_pharmacy_label(file_path, processed_path)
    
    # ตอบกลับผู้ใช้
    reply_text = f"✅ ล้างภาพฉลากยาขั้นสูงสำเร็จ!\nปรับแก้ความเบลอ แสงสะท้อน และขยายภาพเรียบร้อย (ขนาดใหม่: {width}x{height} px)\nเตรียมพร้อมส่งต่อให้ AI วิเคราะห์ข้อมูลครับ"
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )