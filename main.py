from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
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