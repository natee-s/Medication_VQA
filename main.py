from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
import os

app = FastAPI()

# ดึงค่ากุญแจจาก Environment Variables (เดี๋ยวเราไปใส่ค่านี้ในเว็บ Render)
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN', 'YOUR_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET', 'YOUR_SECRET')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# หน้าสำหรับทดสอบว่าเซิร์ฟเวอร์ทำงานปกติไหม
@app.get("/")
def root():
    return {"message": "Medication_VQA Server is running!"}

# เส้นทาง Webhook ที่ LINE จะส่งข้อมูลมาให้
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

# เมื่อมีคนพิมพ์ข้อความเข้ามา ให้บอทตอบกลับเบื้องต้นไปก่อน
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    reply_text = f"คุณพิมพ์มาว่า: {event.message.text}\n(ระบบ ฉลากฉลาด กำลังทดสอบระบบอยู่ เตรียมตัวพบกับของดีได้เลย)"
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

# โค้ดส่วนนี้จะทำงานเมื่อมีคนส่ง "รูปภาพ" เข้ามา
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id
    
    # 1. ดึงไฟล์รูปภาพจากเซิร์ฟเวอร์ LINE
    image_content = line_bot_api.get_message_content(message_id)
    
    # 2. ตั้งชื่อไฟล์และกำหนดโฟลเดอร์ชั่วคราว (/tmp/ รองรับการเขียนไฟล์บน Render)
    file_path = f"/tmp/{message_id}.jpg"
    
    # 3. บันทึกรูปลงเซิร์ฟเวอร์
    with open(file_path, 'wb') as fd:
        for chunk in image_content.iter_content():
            fd.write(chunk)
            
    # 4. ตอบกลับผู้ใช้เบื้องต้น
    reply_text = "ระบบได้รับรูปฉลากยาแล้วครับ กำลังเตรียมปรับความคมชัดภาพ..."
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )