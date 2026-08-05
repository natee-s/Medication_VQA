# PostgreSQL Migration Runbook

เอกสารนี้ใช้สำหรับดูแลระบบหลังย้ายฐานข้อมูลจาก Supabase Database มาเป็น PostgreSQL บน Ubuntu Server โดยยังเก็บ Supabase ไว้เป็น backup และ rollback ได้ผ่านค่า `.env`

เอกสารที่เกี่ยวข้อง:

- `docs/production_config_checklist.md`
- `docs/supabase_after_migration.md`
- `docs/monitoring_logs_health.md`
- `docs/maintenance_routine.md`

## 1. ภาพรวมระบบ

ระบบปัจจุบันบน Ubuntu ใช้ Docker แยกเป็น 3 ส่วนหลัก:

| ส่วน | Container | Port | หน้าที่ |
| --- | --- | --- | --- |
| Main app | `medication-vqa-main` | `17080` | รับ LINE webhook, LIFF Camera, Gemini OCR/RAG, Flex Message |
| PDPA masker | `medication-vqa-pdpa-masker` | `17081` ภายในเครื่อง | ใช้ YOLO-OBB ปิดข้อมูลส่วนบุคคลก่อนส่งเข้า AI |
| PostgreSQL | `medication-vqa-postgres` | `15432` ภายในเครื่อง | เก็บข้อมูลยา, user profile, reminder schedule |

Domain production:

```text
https://ginya.v89tech.com/
```

ไฟล์สำคัญ:

```text
~/apps/Medication_VQA/.env
~/apps/Medication_VQA/docker-compose.ubuntu.yml
~/apps/Medication_VQA/docker-compose.postgres.yml
~/apps/Medication_VQA/postgres/init/001_schema.sql
```

## 2. ค่า `.env` ที่เกี่ยวกับฐานข้อมูล

เปิดไฟล์ `.env` บน Ubuntu:

```bash
cd ~/apps/Medication_VQA
nano .env
```

ถ้าต้องการใช้ PostgreSQL:

```env
DB_BACKEND=postgres
DATABASE_URL=postgresql://medication_vqa:<POSTGRES_PASSWORD>@db:5432/medication_vqa
POSTGRES_DB=medication_vqa
POSTGRES_USER=medication_vqa
POSTGRES_PASSWORD=<POSTGRES_PASSWORD>
```

ถ้าต้องการ rollback กลับ Supabase:

```env
DB_BACKEND=supabase
SUPABASE_URL=<SUPABASE_URL>
SUPABASE_KEY=<SUPABASE_KEY>
```

ห้าม commit ค่า `.env` จริงขึ้น GitHub เพราะมี secret/API key อยู่ในไฟล์นี้

## 3. Start / Restart Services

เข้าโฟลเดอร์โปรเจกต์ก่อนทุกครั้ง:

```bash
cd ~/apps/Medication_VQA
```

เริ่มหรือ restart PostgreSQL:

```bash
docker compose -f docker-compose.postgres.yml up -d
```

เริ่มหรือ restart main app และ PDPA masker:

```bash
docker compose -f docker-compose.ubuntu.yml up -d --build
```

ถ้าต้องการ rebuild เฉพาะ main app:

```bash
docker compose -f docker-compose.ubuntu.yml up -d --build main
```

ถ้าต้องการ rebuild เฉพาะ PDPA masker:

```bash
docker compose -f docker-compose.ubuntu.yml up -d --build pdpa-masker
```

## 4. Health Check

Quick health check หลัง deploy หรือเมื่อสงสัยว่าระบบมีปัญหา:

```bash
cd ~/apps/Medication_VQA
bash tools/ubuntu_health_check.sh
```

ถ้าระบบปกติควรเห็น:

```text
All health checks passed.
```

ดูสถานะ container:

```bash
docker compose -f docker-compose.postgres.yml ps
docker compose -f docker-compose.ubuntu.yml ps
```

สถานะที่ต้องการคือ `Up` หรือ `healthy`

เช็ก main app ภายใน server:

```bash
curl http://127.0.0.1:17080/
```

ผลลัพธ์ที่ควรได้:

```json
{"message":"Banya Sookjai AI Server is running!"}
```

เช็ก main app ผ่าน domain:

```bash
curl https://ginya.v89tech.com/
```

เช็ก PDPA masker:

```bash
curl http://127.0.0.1:17081/health
```

เช็ก PostgreSQL:

```bash
docker compose -f docker-compose.postgres.yml exec db pg_isready -U medication_vqa -d medication_vqa
```

## 5. ตรวจว่าข้อมูลยาอยู่ใน PostgreSQL

นับจำนวนแถวข้อมูลยา:

```bash
docker compose -f docker-compose.postgres.yml exec db psql -U medication_vqa -d medication_vqa -c 'select count(*) from public."Medication_VQA";'
```

นับจำนวน embedding ที่ใช้งานได้:

```bash
docker compose -f docker-compose.postgres.yml exec db psql -U medication_vqa -d medication_vqa -c 'select count(*) from public."Medication_VQA" where embedding is not null;'
```

ทดสอบ vector search:

```bash
docker compose -f docker-compose.postgres.yml exec db psql -U medication_vqa -d medication_vqa -c 'select source_row_number, trade_name, similarity from public.match_symptoms((select embedding from public."Medication_VQA" where embedding is not null limit 1), 0.1, 3);'
```

## 6. Backup / Restore PostgreSQL

Backup จะถูกเก็บไว้ใน:

```text
~/apps/Medication_VQA/postgres/backups/
```

ไฟล์ backup จริงถูก ignore ด้วย `.gitignore` แล้ว ห้าม commit ไฟล์ backup ขึ้น GitHub

สร้าง backup:

```bash
cd ~/apps/Medication_VQA
bash postgres/scripts/backup_postgres.sh
```

ผลลัพธ์ที่ควรเห็น:

```text
Backup complete.
File: /home/v89dev/apps/Medication_VQA/postgres/backups/medication_vqa_YYYYMMDD_HHMMSS.dump
Checksum: /home/v89dev/apps/Medication_VQA/postgres/backups/medication_vqa_YYYYMMDD_HHMMSS.dump.sha256
```

ตั้ง retention กี่วันก่อนลบ backup เก่า:

```bash
RETENTION_DAYS=30 bash postgres/scripts/backup_postgres.sh
```

ดูไฟล์ backup ทั้งหมด:

```bash
ls -lh postgres/backups/
```

เช็ก checksum:

```bash
sha256sum -c postgres/backups/<backup-file.dump>.sha256
```

Restore จาก backup:

```bash
CONFIRM_RESTORE=YES bash postgres/scripts/restore_postgres.sh postgres/backups/<backup-file.dump>
```

คำเตือน: restore เป็นคำสั่งที่มีผลกับข้อมูลใน database ปัจจุบัน ให้ใช้เมื่อจำเป็นหรือเมื่อตั้งใจทดสอบ restore เท่านั้น

ทดสอบ restore แบบปลอดภัย โดยกู้ไฟล์ backup เข้า PostgreSQL container ชั่วคราว ไม่กระทบ database จริง:

```bash
bash postgres/scripts/test_restore_postgres.sh postgres/backups/<backup-file.dump>
```

ถ้าสำเร็จจะเห็นข้อความ:

```text
Safe restore test completed successfully.
```

สคริปต์นี้จะลบ container/volume ทดสอบให้อัตโนมัติหลังจบงาน ถ้าต้องการเก็บ container ทดสอบไว้ตรวจเอง ให้ใช้:

```bash
KEEP_TEST_RESTORE=YES bash postgres/scripts/test_restore_postgres.sh postgres/backups/<backup-file.dump>
```

หลัง restore ให้เช็กจำนวนข้อมูลยา:

```bash
docker compose -f docker-compose.postgres.yml exec db psql -U medication_vqa -d medication_vqa -c 'select count(*) from public."Medication_VQA";'
```

## 7. ทดสอบผ่าน Main App

หลังตั้ง `DB_BACKEND=postgres` แล้ว restart main app:

```bash
docker compose -f docker-compose.ubuntu.yml up -d --build main
```

ทดสอบ endpoint ค้นหายา:

```bash
curl https://ginya.v89tech.com/test-db/AMITRIPTYLINE
```

ถ้าขึ้น `"status":"success"` แปลว่า main app อ่านข้อมูลจาก database backend ได้

ทดสอบผ่าน LINE:

1. พิมพ์อาการ เช่น `ปวดหัว`
2. ส่งรูปฉลากยาจากมือถือ
3. ถ่ายผ่าน LIFF Camera
4. กด Rich menu: `ยาที่ต้องกิน`, `เวลาแจ้งเตือน`, `เปลี่ยนภาษา`

## 8. ดู Logs

ดู log main app:

```bash
docker compose -f docker-compose.ubuntu.yml logs -f main
```

ดู log PDPA masker:

```bash
docker compose -f docker-compose.ubuntu.yml logs -f pdpa-masker
```

ดู log PostgreSQL:

```bash
docker compose -f docker-compose.postgres.yml logs -f db
```

ออกจากหน้า log:

```text
Ctrl + C
```

## 9. Rollback กลับ Supabase

ใช้เมื่อ PostgreSQL มีปัญหาและต้องการให้ระบบกลับไปใช้ Supabase ชั่วคราว

1. เปิด `.env`

```bash
cd ~/apps/Medication_VQA
nano .env
```

2. เปลี่ยนค่า:

```env
DB_BACKEND=supabase
```

3. บันทึกไฟล์:

```text
Ctrl + O -> Enter -> Ctrl + X
```

4. Restart main app:

```bash
docker compose -f docker-compose.ubuntu.yml up -d --build main
```

5. ทดสอบ:

```bash
curl https://ginya.v89tech.com/
curl https://ginya.v89tech.com/test-db/AMITRIPTYLINE
```

## 10. Deploy Code Update

เมื่อมีการ push code จากเครื่อง local ขึ้น GitHub แล้ว ให้ทำบน Ubuntu:

```bash
cd ~/apps/Medication_VQA
git pull
docker compose -f docker-compose.ubuntu.yml up -d --build
```

ถ้ามีการเปลี่ยน schema หรือ PostgreSQL init script ต้องวางแผนแยกก่อน เพราะ `postgres/init/*.sql` จะรันอัตโนมัติเฉพาะตอนสร้าง database volume ใหม่เท่านั้น ไม่ได้รันซ้ำทุกครั้งที่ restart container

## 11. จุดที่ต้องระวัง

- อย่า commit `.env` จริงขึ้น GitHub
- อย่าลบ Supabase จนกว่าจะมี backup/restore PostgreSQL ที่ทดสอบแล้ว
- อย่า commit ไฟล์ใน `postgres/backups/` ขึ้น GitHub
- อย่าลบ Docker volume `postgres_data` ถ้ายังไม่ได้ backup เพราะข้อมูล PostgreSQL อยู่ใน volume นี้
- ถ้าแก้ `POSTGRES_PASSWORD` หลังสร้าง database แล้ว อาจต้องจัดการ user/password ใน PostgreSQL เพิ่ม ไม่ใช่เปลี่ยน `.env` อย่างเดียว
- ถ้าเปลี่ยน YOLO model path ต้องเช็กว่าไฟล์ model อยู่ใน `models/yolo_obb/` บน server แล้ว

## 12. คำสั่งฉุกเฉิน

ดู container ทั้งหมด:

```bash
docker ps
```

Restart main app:

```bash
docker compose -f docker-compose.ubuntu.yml restart main
```

Restart PDPA masker:

```bash
docker compose -f docker-compose.ubuntu.yml restart pdpa-masker
```

Restart PostgreSQL:

```bash
docker compose -f docker-compose.postgres.yml restart db
```

หยุด service ทั้งชุด main/PDPA:

```bash
docker compose -f docker-compose.ubuntu.yml down
```

หยุด PostgreSQL:

```bash
docker compose -f docker-compose.postgres.yml down
```

คำเตือน: คำสั่ง `down` ไม่ลบ volume โดยอัตโนมัติ แต่ห้ามใช้ `down -v` ถ้ายังไม่ได้ backup เพราะจะลบข้อมูล database
