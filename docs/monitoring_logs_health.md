# Monitoring / Logs / Health Check

เอกสารนี้ใช้สำหรับดูแลระบบบน Ubuntu Server หลังย้าย production มาอยู่ที่ `https://ginya.v89tech.com/`

เอกสารที่เกี่ยวข้อง:

- `docs/postgres_migration_runbook.md`
- `docs/production_config_checklist.md`
- `docs/maintenance_routine.md`

เป้าหมายของขั้นตอนนี้คือ:

- รู้ว่าระบบยังทำงานปกติหรือไม่
- รู้ว่าต้องดู log ตรงไหนเมื่อเกิดปัญหา
- แยกให้ได้ว่าปัญหาอยู่ที่ main app, PDPA masker, PostgreSQL, domain/HTTPS, หรือ LINE
- มีคำสั่งชุดเดียวสำหรับตรวจสุขภาพระบบแบบรวดเร็ว

## ภาพรวม Service

```text
LINE OA / LIFF / User
        |
        v
https://ginya.v89tech.com
        |
        v
main app container
port 17080
        |
        +--> PostgreSQL container
        |    internal database
        |
        +--> PDPA masker container
             port 17081 inside server only
```

Service หลักมี 3 ตัว:

| Service | Container | หน้าที่ |
| --- | --- | --- |
| Main app | `medication-vqa-main` | รับ LINE webhook, LIFF Camera, สร้าง Flex Message, ค้นข้อมูลยา |
| PDPA masker | `medication-vqa-pdpa-masker` | ใช้ YOLO-OBB ปิดข้อมูลส่วนบุคคลก่อนส่งเข้า AI |
| PostgreSQL | `medication-vqa-postgres` | เก็บข้อมูลยา, user profile, reminder schedule |

## Quick Health Check

ใช้คำสั่งนี้หลัง deploy, หลังแก้ `.env`, หลัง server restart, หรือเมื่อ LINE เริ่มตอบแปลกๆ:

```bash
cd ~/apps/Medication_VQA
bash tools/ubuntu_health_check.sh
```

ถ้าระบบปกติควรเห็นท้ายคำสั่งประมาณนี้:

```text
All health checks passed.
```

ถ้าเจอ `FAIL` ให้ดูหัวข้อที่ fail ก่อน เช่น:

- `HTTP Health` fail = endpoint/domain อาจมีปัญหา
- `PostgreSQL` fail = database อาจไม่พร้อมหรือ query ไม่ได้
- `Docker Containers` fail = container อาจล่มหรือ unhealthy
- `Recent Logs` มี error = ดูข้อความ error นั้นต่อ

ปรับ domain หรือชื่อยาที่ใช้ test ได้ด้วย environment variable:

```bash
PUBLIC_BASE_URL=https://ginya.v89tech.com TEST_DRUG=AMITRIPTYLINE bash tools/ubuntu_health_check.sh
```

## Manual Health Check

ถ้าต้องการเช็กเองทีละส่วน ให้ใช้คำสั่งด้านล่าง

### 1. เช็ก Container

```bash
cd ~/apps/Medication_VQA
docker compose -f docker-compose.postgres.yml ps
docker compose -f docker-compose.ubuntu.yml ps
```

สถานะที่อยากเห็น:

- `Up`
- `healthy`

ถ้าเห็น `Exited`, `Restarting`, หรือ `unhealthy` ให้ดู log ของ service นั้นทันที

### 2. เช็ก Main App ใน Server

```bash
curl http://127.0.0.1:17080/
```

ผลที่ควรได้:

```json
{"message":"Banya Sookjai AI Server is running!"}
```

### 3. เช็ก Main App ผ่าน Domain

```bash
curl https://ginya.v89tech.com/
```

ผลที่ควรได้เหมือนข้อก่อนหน้า

ถ้า local ผ่าน แต่ domain ไม่ผ่าน ปัญหามักอยู่ที่ reverse proxy/HTTPS/domain routing ให้เก็บผลลัพธ์แล้วส่งให้ mentor ตรวจฝั่ง proxy

### 4. เช็ก LIFF Camera

```bash
curl -I https://ginya.v89tech.com/liff/camera
```

ผลที่ควรได้:

```text
HTTP/2 200
```

ถ้าได้ `405 Method Not Allowed` ตอนใช้ `curl -I` แปลว่า endpoint ไม่รับ HEAD method ให้ลองเปิดใน browser หรือใช้:

```bash
curl https://ginya.v89tech.com/liff/camera | head
```

### 5. เช็ก PDPA Masker

```bash
curl http://127.0.0.1:17081/health
```

ผลที่ควรมี:

```json
{"ok":true,"model_exists":true,"yolo_enabled":true}
```

ถ้า `model_exists=false` ให้เช็กว่าไฟล์ model อยู่จริง:

```bash
ls -lh ~/apps/Medication_VQA/models/yolo_obb/best_round3.pt
```

### 6. เช็ก PostgreSQL

```bash
docker compose -f docker-compose.postgres.yml exec db pg_isready -U medication_vqa -d medication_vqa
```

ผลที่ควรได้:

```text
accepting connections
```

เช็กจำนวนข้อมูลยา:

```bash
docker compose -f docker-compose.postgres.yml exec db psql -U medication_vqa -d medication_vqa -c 'select count(*) from public."Medication_VQA";'
```

เช็กจำนวน embedding:

```bash
docker compose -f docker-compose.postgres.yml exec db psql -U medication_vqa -d medication_vqa -c 'select count(*) from public."Medication_VQA" where embedding is not null;'
```

เช็ก vector search:

```bash
docker compose -f docker-compose.postgres.yml exec db psql -U medication_vqa -d medication_vqa -c 'select source_row_number, trade_name, similarity from public.match_symptoms((select embedding from public."Medication_VQA" where embedding is not null limit 1), 0.1, 3);'
```

## Logs

### ดู Log Main App

```bash
docker compose -f docker-compose.ubuntu.yml logs -f main
```

ใช้เมื่อ:

- LINE ไม่ตอบ
- Flex Message ผิด
- ค้นยาไม่เจอ
- Gemini/OCR error
- database query error

### ดู Log PDPA Masker

```bash
docker compose -f docker-compose.ubuntu.yml logs -f pdpa-masker
```

ใช้เมื่อ:

- ส่งรูป upload แล้วระบบบอก PDPA masking ขัดข้อง
- YOLO ไม่เจอฉลาก
- mask แล้วแปลก
- `/health` ของ port 17081 fail

### ดู Log PostgreSQL

```bash
docker compose -f docker-compose.postgres.yml logs -f db
```

ใช้เมื่อ:

- database ไม่พร้อม
- query ช้า/ล้มเหลว
- import/backup/restore มีปัญหา

### ออกจากหน้า Log

กด:

```text
Ctrl + C
```

## คำสั่งดู Error แบบเร็ว

Main app:

```bash
docker compose -f docker-compose.ubuntu.yml logs --tail=200 main | grep -Ei "error|exception|failed|timeout|traceback|unauthorized|ขัดข้อง"
```

PDPA masker:

```bash
docker compose -f docker-compose.ubuntu.yml logs --tail=200 pdpa-masker | grep -Ei "error|exception|failed|timeout|traceback|unauthorized"
```

PostgreSQL:

```bash
docker compose -f docker-compose.postgres.yml logs --tail=200 db | grep -Ei "error|fatal|panic|warning"
```

## Restart แบบปลอดภัย

Restart เฉพาะ main app:

```bash
docker compose -f docker-compose.ubuntu.yml restart main
```

Restart เฉพาะ PDPA masker:

```bash
docker compose -f docker-compose.ubuntu.yml restart pdpa-masker
```

Restart เฉพาะ PostgreSQL:

```bash
docker compose -f docker-compose.postgres.yml restart db
```

Restart ทั้งระบบ:

```bash
docker compose -f docker-compose.postgres.yml up -d
docker compose -f docker-compose.ubuntu.yml up -d --build
```

## Disk / Storage Check

เช็กพื้นที่ disk:

```bash
df -h
```

เช็กขนาด backup:

```bash
du -sh ~/apps/Medication_VQA/postgres/backups
```

เช็กขนาด debug image:

```bash
du -sh ~/apps/Medication_VQA/test/local_pdpa_debug
```

ถ้า debug image โตเกินไป และระบบเสถียรแล้ว ให้ปิดใน `.env`:

```env
SAVE_LOCAL_PDPA_DEBUG_IMAGES=false
```

แล้ว restart PDPA masker:

```bash
docker compose -f docker-compose.ubuntu.yml up -d --build pdpa-masker
```

## LINE Regression Check

หลัง deploy หรือ restart ครั้งใหญ่ ให้ทดสอบใน LINE:

1. พิมพ์อาการ เช่น `ปวดหัว`
2. ส่งรูปฉลากยาจากมือถือ
3. ถ่ายผ่าน LIFF Camera
4. กด Rich menu `ยาที่ต้องกิน / Drug list`
5. กด Rich menu `เวลาแจ้งเตือน / Alarm setting`
6. กด Rich menu `เปลี่ยนภาษา / Language`
7. กดปุ่มใน Flex Message เช่น `ตั้งเตือนกินยา`, `รับทราบ`, `กินยาทั้งหมดแล้ว`, `เลื่อน 15 นาที`

ถ้าทุกข้อทำงานและตอบเป็นภาษาที่เลือกไว้ แปลว่าระบบหลักยังปกติ

## อาการเสียที่พบบ่อย

### เปิดเว็บแล้ว 502 Bad Gateway

เช็ก:

```bash
curl http://127.0.0.1:17080/
curl https://ginya.v89tech.com/
docker compose -f docker-compose.ubuntu.yml ps
```

ถ้า `127.0.0.1:17080` ผ่าน แต่ domain fail ให้ส่งข้อมูลให้ mentor เช็ก reverse proxy/HTTPS

### LINE มี animation 3 จุดแล้วเงียบ

ดู main log:

```bash
docker compose -f docker-compose.ubuntu.yml logs -f main
```

สาเหตุที่เป็นไปได้:

- LINE reply token หมดอายุ
- Gemini/OCR timeout
- database query error
- PDPA service timeout

### Upload รูปแล้วระบบบอก PDPA ขัดข้อง

เช็ก:

```bash
curl http://127.0.0.1:17081/health
docker compose -f docker-compose.ubuntu.yml logs -f pdpa-masker
```

สาเหตุที่เป็นไปได้:

- YOLO model หาย
- PDPA token ไม่ตรงกัน
- PDPA masker ล่ม
- รูปใหญ่หรือประมวลผลนานเกิน timeout

### ค้นยาไม่เจอทั้งที่ควรเจอ

เช็ก database:

```bash
docker compose -f docker-compose.postgres.yml exec db psql -U medication_vqa -d medication_vqa -c 'select count(*) from public."Medication_VQA";'
docker compose -f docker-compose.postgres.yml exec db psql -U medication_vqa -d medication_vqa -c 'select count(*) from public."Medication_VQA" where embedding is not null;'
curl https://ginya.v89tech.com/test-db/AMITRIPTYLINE
```

ถ้า row count เป็น 0 หรือ embedding เป็น 0 ต้องกลับไปเช็กขั้นตอน import PostgreSQL

## Routine

แนะนำรอบการเช็ก:

- หลัง deploy ทุกครั้ง: รัน `bash tools/ubuntu_health_check.sh`
- ทุกวันช่วงทดสอบ production แรกๆ: รัน quick health check 1 ครั้ง
- ทุกสัปดาห์: ทำ PostgreSQL backup และ safe restore test
- หลังระบบนิ่ง: ปิด debug image และเก็บ Supabase ไว้เป็น fallback/cold backup ตามเอกสาร `docs/supabase_after_migration.md`
