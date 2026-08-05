# Production Config Checklist

เอกสารนี้ใช้เช็กค่าบน Ubuntu Server ก่อน deploy หรือหลังแก้ `.env`

เอกสารที่เกี่ยวข้อง:

- `docs/postgres_migration_runbook.md`
- `docs/supabase_after_migration.md`

## 1. ไฟล์ที่ห้าม commit

ไฟล์เหล่านี้ต้องอยู่เฉพาะบนเครื่อง/server และห้าม commit ขึ้น GitHub:

- `.env`
- ไฟล์ backup ใน `postgres/backups/`
- ไฟล์ debug image
- ไฟล์ model เช่น `.pt`, `.onnx`
- logs เช่น `.log`

ตรวจสถานะ Git:

```bash
git status --short
```

ถ้าเห็น `.env`, `postgres/backups/`, หรือไฟล์ model `.pt` อยู่ในรายการที่จะ commit ให้หยุดก่อน

## 2. Required Environment Variables

เปิด `.env` บน Ubuntu:

```bash
cd ~/apps/Medication_VQA
nano .env
```

ค่าที่ต้องมี:

```env
LINE_CHANNEL_ACCESS_TOKEN=...
LINE_CHANNEL_SECRET=...
LIFF_ID=...

GEMINI_API_KEY=...

DB_BACKEND=postgres
POSTGRES_DB=medication_vqa
POSTGRES_USER=medication_vqa
POSTGRES_PASSWORD=...
DATABASE_URL=postgresql://medication_vqa:<POSTGRES_PASSWORD>@db:5432/medication_vqa

SUPABASE_URL=...
SUPABASE_KEY=...

PDPA_MASKING_SERVICE_URL=http://pdpa-masker:17081/mask
PDPA_MASKING_SERVICE_TOKEN=...
LOCAL_PDPA_SERVICE_TOKEN=...
PDPA_MASKING_SERVICE_TIMEOUT_SECONDS=60

YOLO_OBB_ENABLED=true
YOLO_OBB_MODEL_PATH=/app/models/yolo_obb/best_round3.pt
YOLO_OBB_CONFIDENCE=0.45
YOLO_OBB_IMAGE_SIZE=1024
```

หมายเหตุ:

- `DB_BACKEND=postgres` คือ production ปัจจุบัน
- `SUPABASE_URL` และ `SUPABASE_KEY` ยังควรเก็บไว้ช่วงแรกเพื่อ rollback
- `PDPA_MASKING_SERVICE_TOKEN` และ `LOCAL_PDPA_SERVICE_TOKEN` ต้องเป็นค่าเดียวกัน
- `DATABASE_URL` ใน Docker ต้องใช้ host เป็น `db` ไม่ใช่ `127.0.0.1`

## 3. Debug Image Config

ช่วงทดสอบ PDPA masking อาจเปิดไว้:

```env
SAVE_LOCAL_PDPA_DEBUG_IMAGES=true
LOCAL_PDPA_DEBUG_DIR=/app/test/local_pdpa_debug
```

เมื่อระบบนิ่งแล้ว แนะนำให้ปิด:

```env
SAVE_LOCAL_PDPA_DEBUG_IMAGES=false
```

เหตุผล: ลดการเก็บรูปผู้ใช้บน server และลดการใช้ disk

## 4. Service Health Check

หลังแก้ `.env` หรือหลัง deploy ให้ restart:

```bash
cd ~/apps/Medication_VQA
docker compose -f docker-compose.postgres.yml up -d
docker compose -f docker-compose.ubuntu.yml up -d --build
```

เช็ก container:

```bash
docker compose -f docker-compose.postgres.yml ps
docker compose -f docker-compose.ubuntu.yml ps
```

เช็ก endpoint:

```bash
curl https://ginya.v89tech.com/
curl http://127.0.0.1:17081/health
curl https://ginya.v89tech.com/test-db/AMITRIPTYLINE
```

## 5. PostgreSQL Check

เช็กว่า database พร้อม:

```bash
docker compose -f docker-compose.postgres.yml exec db pg_isready -U medication_vqa -d medication_vqa
```

เช็กจำนวนข้อมูลยา:

```bash
docker compose -f docker-compose.postgres.yml exec db psql -U medication_vqa -d medication_vqa -c 'select count(*) from public."Medication_VQA";'
```

เช็ก embedding:

```bash
docker compose -f docker-compose.postgres.yml exec db psql -U medication_vqa -d medication_vqa -c 'select count(*) from public."Medication_VQA" where embedding is not null;'
```

## 6. Rollback Switch

ถ้า PostgreSQL มีปัญหา ให้เปลี่ยน:

```env
DB_BACKEND=supabase
```

แล้ว restart main:

```bash
docker compose -f docker-compose.ubuntu.yml up -d --build main
```

เมื่อแก้ PostgreSQL เสร็จค่อยเปลี่ยนกลับ:

```env
DB_BACKEND=postgres
```

แล้ว restart main อีกครั้ง

## 7. LINE Regression Checklist

หลัง deploy ให้ทดสอบใน LINE:

- พิมพ์อาการ เช่น `ปวดหัว`
- ส่งรูปฉลากยาจากมือถือ
- ถ่ายผ่าน LIFF Camera
- กด Rich menu: `ยาที่ต้องกิน`
- กด Rich menu: `เวลาแจ้งเตือน`
- กด Rich menu: `เปลี่ยนภาษา`
- กดปุ่มใน Flex Message เช่น ตั้งเตือน/รับทราบ/ยาหมด

ทุกข้อควรตอบกลับเป็นภาษาที่ user เลือกไว้
