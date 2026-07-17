import os
from supabase import create_client, Client
from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

# ==========================================
# 1. ใส่ API Keys ของคุณแมนตรงนี้ครับ
# ==========================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ตรวจสอบความปลอดภัยเบื้องต้น
if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    raise ValueError("⚠️ หา API Key ไม่พบ! กรุณาตรวจสอบไฟล์ .env")

# เชื่อมต่อระบบ
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

def generate_embeddings():
    print("⏳ เริ่มต้นกระบวนการแปลงข้อมูลเป็น Vector...")
    
    # 1. ดึงข้อมูลยาที่คอลัมน์ embedding ยังเป็นค่าว่าง (null)
    # หมายเหตุ: หาก Primary Key ของคุณแมนไม่ได้ชื่อ 'id' ให้แก้ตรง "id, trade_name..." ด้วยนะครับ
    response = supabase.table("Medication_VQA").select("source_row_number, trade_name, rag_text, indication").is_("embedding", "null").execute()
    records = response.data

    if not records:
        print("✅ ข้อมูลยาทั้งหมดมี Vector ครบถ้วนแล้วครับ ไม่มีอะไรต้องอัปเดต")
        return

    print(f"🔍 พบข้อมูลที่ต้องแปลง Vector จำนวน {len(records)} รายการ\n")

    for record in records:
        # 2. นำข้อมูล 'ชื่อยา' 'สรรพคุณ' และ 'ข้อบ่งใช้' มามัดรวมกันเป็นประโยคเดียว
        # เพื่อให้ AI เข้าใจบริบทรวมๆ ของยาตัวนี้ได้ดีที่สุด
        text_to_embed = f"ชื่อยา: {record.get('trade_name', '')} สรรพคุณและอาการ: {record.get('rag_text', '')} {record.get('indication', '')}"
        
        try:
            # 3. เรียกใช้โมเดล text-embedding-004 เพื่อแปลงข้อความเป็น Vector
            result = ai_client.models.embed_content(
                model='gemini-embedding-001',
                contents=text_to_embed,
                config=types.EmbedContentConfig(output_dimensionality=768) # 👈 เพิ่มตรงนี้
            )
            
            # ดึงค่าชุดตัวเลข (768 มิติ) ออกมา
            embedding_vector = result.embeddings[0].values

            # 4. อัปเดตชุดตัวเลขนี้ กลับลงไปในแถวเดิมของตาราง
            supabase.table("Medication_VQA").update({"embedding": embedding_vector}).eq("source_row_number", record["source_row_number"]).execute()
            
            print(f"✅ แปลงและบันทึก Vector สำเร็จ: {record['trade_name']}")
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดกับยา {record['trade_name']}: {e}")

if __name__ == "__main__":
    generate_embeddings()