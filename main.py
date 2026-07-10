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
from linebot.models import MessageEvent, ImageMessage, TextSendMessage, FlexSendMessage, PostbackEvent

# ==========================================
# 1. ฟังก์ชันสร้างฟังก์ชันด่านหน้า (Gatekeeper)
# ==========================================
def check_image_quality(file_path):
    # 1. ตรวจสอบขนาดไฟล์ (ไม่เกิน 3MB)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 3.0:
        return False, f"⚠️ รูปภาพมีขนาดใหญ่เกินไป ({file_size_mb:.1f} MB) กรุณาส่งรูปไม่เกิน 3 MB ครับ หรือถ่ายผ่านกล้องของ LINE ได้เลยครับ"

    img = cv2.imread(file_path)
    if img is None:
        return False, "⚠️ ไม่สามารถอ่านไฟล์รูปภาพได้ กรุณาส่งใหม่อีกครั้งครับ"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. ตรวจสอบความสว่าง
    brightness = np.mean(gray)
    if brightness < 50:
        return False, "⚠️ รูปภาพมืดเกินไป กรุณาถ่ายในที่สว่างแล้วส่งมาใหม่ครับ"
    
    # ตรวจแสงสะท้อน (พิกเซลสว่างจัด > 240 มีมากกว่า 5% ของพื้นที่ภาพ)
    glare_ratio = np.sum(gray > 240) / gray.size
    if glare_ratio > 0.05:
        return False, "⚠️ รูปภาพมีแสงแฟลชสะท้อนบังข้อความ กรุณาหลีกเลี่ยงแสงสะท้อนแล้วถ่ายใหม่ครับ"

    # 3. ตรวจสอบความเปรียบต่างสี (Contrast)
    contrast = np.std(gray)
    if contrast < 20:
        return False, "⚠️ รูปภาพจางหรือสีกลืนกันเกินไป ทำให้ระบบอาจอ่านผิดพลาด กรุณาถ่ายใหม่อีกครั้งครับ"

    # 4. ตรวจสอบความเบลอ (Blurriness)
    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_val < 100:
        return False, "⚠️ รูปภาพเบลอเกินไป กรุณาแตะโฟกัสที่กล้องให้ตัวหนังสือคมชัด แล้วถ่ายใหม่ครับ"

    # 5. ตรวจสอบระยะห่าง (Bounding Box Area)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x_min, y_min = img.shape[1], img.shape[0]
        x_max, y_max = 0, 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)
        
        object_area = (x_max - x_min) * (y_max - y_min)
        total_area = img.shape[0] * img.shape[1]
        if (object_area / total_area) < 0.2:
            return False, "⚠️ รูปภาพอยู่ไกลเกินไป กรุณาถ่ายใกล้ๆ ให้ฉลากยาเต็มกรอบภาพครับ"

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
    
    # ==========================================
    # 0. แสดง Loading Animation (สถานะกำลังพิมพ์...)
    # ==========================================
    try:
        url = "https://api.line.me/v2/bot/chat/loading/start"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {line_bot_api.http_client.headers['Authorization'].split(' ')[1]}" 
        }
        # ตั้งเวลาให้จุดไข่ปลาแสดงสูงสุด 5 วินาที (ครอบคลุมเวลาที่ AI คิดพอดี)
        data = {
            "chatId": event.source.user_id,
            "loadingSeconds": 5
        }
        requests.post(url, headers=headers, json=data)
    except Exception as e:
        print(f"Loading Animation Error: {e}")

    # 3.1 รับภาพจากผู้ใช้
    image_content = line_bot_api.get_message_content(message_id)
    file_path = f"/tmp/{message_id}.jpg"
    processed_path = f"/tmp/processed_{message_id}.jpg"
    
    with open(file_path, 'wb') as fd:
        for chunk in image_content.iter_content():
            fd.write(chunk)
            
    # ==========================================
    # เฟส 1: ตรวจสอบคุณภาพรูป (QC Gatekeeper)
    # ==========================================
    is_valid, error_msg = check_image_quality(file_path)
    if not is_valid:
        # ถ้าไม่ผ่านเกณฑ์ เด้งแจ้งเตือนแล้วหยุดทำงานทันที (Fail-Fast)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=error_msg))
        return

    # ==========================================
    # เฟส 2: ล้างและตกแต่งภาพ
    # ==========================================
    width, height = process_pharmacy_label(file_path, processed_path)
    
    with open(processed_path, "rb") as f:
        image_bytes = f.read()
        
    # ==========================================
    # เฟส 3: เรียกใช้งาน Gemini + ยามเฝ้าประตู (Bouncer)
    # ==========================================
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                """นี่คือภาพฉลากยา 
กฎสำคัญ:
1. หากพบว่าตัวหนังสือในภาพตะแคงซ้าย/ขวา (90, 270 องศา) หรือ กลับหัว (180 องศา) ให้ตอบกลับมาเป็น JSON แบบนี้เท่านั้น: {"error": "rotated"} และห้ามตอบอย่างอื่น
2. หากภาพตั้งตรงปกติ (0 องศา) กรุณาดึงข้อมูลและส่งกลับมาเป็นรูปแบบ JSON (ไม่ต้องมี Markdown หรือ Code block คร่อม) โดยใช้โครงสร้างดังนี้:
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
        
        raw_text = response.text.strip()
        if raw_text.startswith('```json'):
            raw_text = raw_text.replace('```json', '').replace('```', '').strip()
        elif raw_text.startswith('```'):
            raw_text = raw_text.replace('```', '').strip()
            
        try:
            data = json.loads(raw_text)
            
            # 🚨 ดักจับ Error รูปกลับหัวจาก Gemini
            if data.get("error") == "rotated":
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="📸 รูปภาพตะแคงหรือกลับหัวครับ กรุณาถ่ายฉลากยาให้ตั้งตรงแล้วส่งมาใหม่อีกครั้งนะครับ")
                )
                return

            # ถ้าข้อมูลปกติ ดำเนินการสร้าง Flex Message ต่อ
            trade_name = data.get('trade_name') or 'ไม่ระบุ'
            generic_name = data.get('generic_name') or 'ไม่ระบุ'
            indication = data.get('indication') or 'ไม่ระบุ'
            dosage = data.get('dosage') or 'ไม่ระบุ'
            instruction = data.get('instruction') or 'ไม่ระบุ'
            warning = data.get('warning') or 'ไม่มี'

            flex_bubble = {
                "type": "bubble",
                "size": "mega",
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": "#1DB446",
                    "contents": [
                        {"type": "text", "text": "💊 ข้อมูลฉลากยา (บ้านยาสุขใจ)", "weight": "bold", "size": "lg", "color": "#FFFFFF"}
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
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"พบปัญหาในการจัดรูปแบบข้อมูล:\n{raw_text}"))
            
    except Exception as e:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"⚠️ ไม่สามารถเชื่อมต่อสมอง AI ได้: {str(e)}"))

@handler.add(PostbackEvent)
def handle_postback(event):reminder
    data = event.postback.data
    
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
        
        # ดึง User ID เตรียมเอาไปผูกกับตาราง Database
        user_id = event.source.user_id
        
        print(f"เตรียมบันทึกข้อมูลลง DB: User={user_id}, Drug={drug_name}")
        
        # ตอบกลับผู้ใช้
        reply_text = f"⏰ ตั้งเวลาเตือนสำหรับยา {drug_name} เรียบร้อยครับ!\n(ระบบจะเชื่อมต่อฐานข้อมูลในเฟสถัดไป)"
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )