# ใช้ Python 3.10 เป็นระบบพื้นฐาน
FROM python:3.10-slim

# อัปเดตระบบ และติดตั้งโปรแกรม Tesseract พร้อมแพ็กเกจภาษาไทย + ตัวช่วย OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-tha \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ตั้งค่าโฟลเดอร์ทำงานในเซิร์ฟเวอร์
WORKDIR /app

# ก๊อปปี้ไฟล์รายชื่อไลบรารีและสั่งติดตั้ง
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ก๊อปปี้ไฟล์โค้ดทั้งหมด (เช่น main.py) ลงเซิร์ฟเวอร์
COPY . .

# คำสั่งเปิดเซิร์ฟเวอร์เมื่อติดตั้งเสร็จ
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]