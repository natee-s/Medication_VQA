# Production Load Testing

เอกสารนี้ใช้ทดสอบว่า Medication_VQA production รองรับ load ได้แค่ไหนในระดับ HTTP/read-only

## หลักสำคัญ

สคริปต์นี้ตั้งใจไม่ยิง endpoint ที่มี side effect เป็นค่าเริ่มต้น:

- ไม่ยิง `/webhook`
- ไม่ยิง `/liff/upload-label`
- ไม่ยิง `/cron/check-reminder`

เพราะ endpoint เหล่านี้อาจเรียก LINE, Gemini, YOLO หรือ reminder จริง

ค่าเริ่มต้นจะทดสอบเฉพาะ:

- `/`
- `/liff/camera`
- `/liff/config`
- `/test-db/{drug}`

## ค่าที่ควรดู

- `error_rate`: ควรใกล้ `0%`
- `p95 latency`: 95% ของ request เร็วกว่าค่านี้
- `p99 latency`: 99% ของ request เร็วกว่าค่านี้
- `rps`: requests per second

เกณฑ์อ่านผลเบื้องต้น:

- ดี: error `0%`, p95 ต่ำกว่า `1000 ms`
- พอใช้: error ต่ำกว่า `1%`, p95 ต่ำกว่า `2000 ms`
- ควรหยุดเพิ่ม load: error มากกว่า `1%`, timeout, 502/504, หรือ container restart

## Run บน Ubuntu

เปิด VS Code SSH: GinyaAI แล้วรัน:

```bash
cd ~/apps/Medication_VQA
git pull origin main
```

### 1. Smoke Test เบาๆ

```bash
python3 tools/load_test_production.py \
  --base-url https://ginya.v89tech.com \
  --profile production-read \
  --concurrency 3 \
  --duration-seconds 30 \
  --ramp-up-seconds 5 \
  --output-json logs/load_test_smoke.json \
  --output-csv logs/load_test_smoke.csv
```

ถ้า error เป็น `0%` ค่อยไปขั้นถัดไป

### 2. Baseline Test

```bash
python3 tools/load_test_production.py \
  --base-url https://ginya.v89tech.com \
  --profile production-read \
  --concurrency 10 \
  --duration-seconds 120 \
  --ramp-up-seconds 20 \
  --output-json logs/load_test_baseline.json \
  --output-csv logs/load_test_baseline.csv
```

### 3. Step Load Test

รันเพิ่มทีละระดับ:

```bash
python3 tools/load_test_production.py \
  --base-url https://ginya.v89tech.com \
  --profile production-read \
  --concurrency 20 \
  --duration-seconds 120 \
  --ramp-up-seconds 30 \
  --output-json logs/load_test_c20.json \
  --output-csv logs/load_test_c20.csv
```

```bash
python3 tools/load_test_production.py \
  --base-url https://ginya.v89tech.com \
  --profile production-read \
  --concurrency 50 \
  --duration-seconds 180 \
  --ramp-up-seconds 60 \
  --output-json logs/load_test_c50.json \
  --output-csv logs/load_test_c50.csv
```

หยุดเพิ่มทันทีถ้าเห็น:

- error rate มากกว่า `1%`
- p95 เกิน `2000 ms`
- มี `502`, `504`, timeout
- container restart

## ดู resource ระหว่างทดสอบ

เปิด terminal อีกแท็บบน Ubuntu:

```bash
docker stats medication-vqa-main medication-vqa-pdpa-masker medication-vqa-postgres
```

อีกแท็บดู log:

```bash
docker compose -f docker-compose.ubuntu.yml logs -f --tail=100 main
```

อีกแท็บดู PostgreSQL:

```bash
docker compose -f docker-compose.postgres.yml logs -f --tail=100 db
```

## ประเมินจำนวน user แบบคร่าวๆ

สคริปต์มีค่า:

```bash
--expected-requests-per-user-per-minute 2
```

แปลว่า 1 active user โดยเฉลี่ยยิง request ประมาณ 2 ครั้งต่อนาที

ตัวอย่าง:

ถ้าระบบรับได้ `20 rps` อย่างเสถียร:

```text
20 requests/second = 1200 requests/minute
1200 / 2 = ประมาณ 600 active users
```

แต่นี่เป็นการประเมินเฉพาะ endpoint อ่านข้อมูล ไม่ใช่ capacity สุดท้ายของ flow ที่มี Gemini/YOLO/LINE

## ข้อจำกัด

สคริปต์นี้ตอบคำถามได้ว่า:

- หน้าเว็บ/LIFF endpoint รับ concurrent requests ได้แค่ไหน
- database lookup ผ่าน `/test-db` เร็วไหม
- domain/reverse proxy มีปัญหา 502/timeout ไหม

สคริปต์นี้ยังไม่ตอบเต็มๆ ว่า:

- Gemini รองรับกี่ request พร้อมกัน
- LINE push/reply limit เป็นเท่าไหร่
- YOLO masking รองรับรูปจริงพร้อมกันกี่รูป

ถ้าจะวัด image/Gemini/YOLO จริง ต้องทำ test แยกแบบควบคุม quota และจำนวนรูปอย่างระมัดระวัง
