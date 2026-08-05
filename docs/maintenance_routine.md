# Maintenance Routine

เอกสารนี้คือ routine การดูแลระบบ Medication VQA หลังย้าย production มาอยู่บน Ubuntu Server แล้ว

ใช้ร่วมกับ:

- `docs/postgres_migration_runbook.md`
- `docs/monitoring_logs_health.md`
- `docs/production_config_checklist.md`
- `docs/supabase_after_migration.md`

## หลักคิด

Routine การดูแลระบบไม่ใช่การแก้ระบบทุกวัน แต่คือการเช็กสั้นๆ อย่างสม่ำเสมอ เพื่อให้รู้ปัญหาก่อน user เจอ และมี backup พร้อมถ้าต้องกู้ระบบ

สิ่งที่ต้องดูแลหลักๆ มี 5 อย่าง:

1. ระบบยังตอบไหม
2. Docker containers ยัง healthy ไหม
3. PostgreSQL ยัง query ได้ไหม
4. Backup ยังสร้างและ restore ได้ไหม
5. Disk ไม่เต็มเพราะ backup/debug image/logs ใช่ไหม

## Daily Routine

ทำวันละ 1 ครั้งในช่วงที่ระบบเพิ่งย้าย server ใหม่ หรือช่วงที่ยังทดสอบ production แรกๆ

ใช้เวลาประมาณ 2-5 นาที

```bash
cd ~/apps/Medication_VQA
bash tools/ubuntu_health_check.sh
```

ผลที่ต้องการ:

```text
All health checks passed.
```

ถ้าไม่ผ่าน:

1. ดูว่าหัวข้อไหนขึ้น `FAIL`
2. เปิด `docs/monitoring_logs_health.md`
3. ดู log ของ service ที่ fail
4. ถ้ายังแก้ไม่ได้ ให้จด error แล้วค่อยถาม mentor หรือกลับมาถามต่อ

## After Deploy Routine

ทำทุกครั้งหลัง `git pull`, rebuild, restart, หรือ deploy code ใหม่

### 1. Pull code ล่าสุด

```bash
cd ~/apps/Medication_VQA
git pull
```

### 2. Restart service

ถ้าแก้ code main app หรือไฟล์ frontend/LIFF:

```bash
docker compose -f docker-compose.ubuntu.yml up -d --build main
```

ถ้าแก้ YOLO/PDPA masker:

```bash
docker compose -f docker-compose.ubuntu.yml up -d --build pdpa-masker
```

ถ้าไม่แน่ใจว่าแก้อะไรบ้าง ใช้ทั้งชุด:

```bash
docker compose -f docker-compose.postgres.yml up -d
docker compose -f docker-compose.ubuntu.yml up -d --build
```

### 3. Run health check

```bash
bash tools/ubuntu_health_check.sh
```

### 4. Test ใน LINE

ทดสอบ workflow หลัก:

1. พิมพ์อาการ เช่น `ปวดหัว`
2. ส่งรูปฉลากยาจากมือถือ
3. ถ่ายผ่าน LIFF Camera
4. กด Rich menu `ยาที่ต้องกิน / Drug list`
5. กด Rich menu `เวลาแจ้งเตือน / Alarm setting`
6. กด Rich menu `เปลี่ยนภาษา / Language`
7. กดปุ่มใน Flex Message เช่น `ตั้งเตือนกินยา`, `รับทราบ`, `กินยาทั้งหมดแล้ว`, `เลื่อน 15 นาที`

ถ้าทั้งหมดผ่าน ค่อยถือว่า deploy รอบนั้นสมบูรณ์

## Weekly Routine

ทำสัปดาห์ละ 1 ครั้ง

เป้าหมายคือยืนยันว่า backup ใช้ได้จริง ไม่ใช่แค่มีไฟล์ backup เฉยๆ

### 1. สร้าง PostgreSQL backup

```bash
cd ~/apps/Medication_VQA
bash postgres/scripts/backup_postgres.sh
```

### 2. เช็ก checksum

ดูชื่อไฟล์ backup ล่าสุด:

```bash
ls -lt postgres/backups/ | head
```

เช็กไฟล์ `.sha256` ของ backup ล่าสุด:

```bash
sha256sum -c postgres/backups/<backup-file.dump>.sha256
```

ผลที่ต้องการ:

```text
postgres/backups/<backup-file.dump>: OK
```

### 3. Safe restore test

ทดสอบ restore เข้า temporary container เพื่อไม่กระทบ database จริง:

```bash
bash postgres/scripts/test_restore_postgres.sh postgres/backups/<backup-file.dump>
```

ผลที่ต้องการ:

```text
Safe restore test completed successfully.
```

### 4. เช็ก disk usage

```bash
df -h
du -sh postgres/backups
du -sh test/local_pdpa_debug 2>/dev/null || true
docker system df
```

ถ้า disk เริ่มเต็ม ให้ดูหัวข้อ `Disk Cleanup Routine`

## Monthly Routine

ทำเดือนละ 1 ครั้ง

### 1. ตรวจ backup เก่า

```bash
ls -lh postgres/backups/
```

โดย default สคริปต์ backup จะลบไฟล์ที่เก่ากว่า 14 วัน ถ้าต้องการเก็บ 30 วัน:

```bash
RETENTION_DAYS=30 bash postgres/scripts/backup_postgres.sh
```

### 2. ตรวจว่า debug image ยังจำเป็นไหม

เปิด `.env`:

```bash
nano .env
```

ถ้าระบบนิ่งแล้ว แนะนำให้ปิด:

```env
SAVE_LOCAL_PDPA_DEBUG_IMAGES=false
```

แล้ว restart PDPA masker:

```bash
docker compose -f docker-compose.ubuntu.yml up -d --build pdpa-masker
```

เหตุผล: ลดการเก็บรูปผู้ใช้บน server และลดการใช้ disk

### 3. ทบทวน Supabase fallback

ดูเอกสาร:

```text
docs/supabase_after_migration.md
```

ถ้า PostgreSQL ใช้งาน production เสถียรครบ 2-4 สัปดาห์, backup/restore ผ่าน, และ mentor เห็นด้วย ค่อยตัดสินใจว่าจะเก็บ Supabase เป็น cold backup หรือปิด project

## Disk Cleanup Routine

ใช้เมื่อ disk เริ่มเต็ม หรือ health check แจ้งว่า storage สูงผิดปกติ

### 1. ดูพื้นที่รวม

```bash
df -h
```

### 2. ดูขนาดโฟลเดอร์ที่มักโต

```bash
du -sh postgres/backups
du -sh test/local_pdpa_debug 2>/dev/null || true
docker system df
```

### 3. ลบ Docker build cache ที่ไม่ใช้แล้ว

คำสั่งนี้ลบเฉพาะ cache ที่ Docker สร้างไว้ ไม่ลบ database volume:

```bash
docker builder prune
```

ถ้าต้องการลบ cache แบบไม่ถามซ้ำ:

```bash
docker builder prune -f
```

คำเตือน: ห้ามใช้ `docker volume prune` ถ้ายังไม่เข้าใจ เพราะอาจลบ volume database ได้

## Incident Routine

ใช้เมื่อ user แจ้งว่าระบบมีปัญหา หรือ LINE ไม่ตอบ

### 1. อย่าเพิ่ง restart ทันที

ก่อน restart ให้เก็บข้อมูลก่อน เพราะถ้า restart เลย log บางอย่างอาจหายหรือไล่ยากขึ้น

### 2. Run health check

```bash
cd ~/apps/Medication_VQA
bash tools/ubuntu_health_check.sh
```

### 3. ดู log ที่เกี่ยวข้อง

Main app:

```bash
docker compose -f docker-compose.ubuntu.yml logs --tail=200 main
```

PDPA masker:

```bash
docker compose -f docker-compose.ubuntu.yml logs --tail=200 pdpa-masker
```

PostgreSQL:

```bash
docker compose -f docker-compose.postgres.yml logs --tail=200 db
```

### 4. จด incident note

ใช้ template นี้:

```text
Date/time:
User action:
Expected result:
Actual result:
Health check result:
Main app log:
PDPA masker log:
PostgreSQL log:
Action taken:
Result after action:
```

### 5. Restart เฉพาะ service ที่มีปัญหา

Main app:

```bash
docker compose -f docker-compose.ubuntu.yml restart main
```

PDPA masker:

```bash
docker compose -f docker-compose.ubuntu.yml restart pdpa-masker
```

PostgreSQL:

```bash
docker compose -f docker-compose.postgres.yml restart db
```

หลัง restart ให้รัน:

```bash
bash tools/ubuntu_health_check.sh
```

## Optional Cron Schedule

ยังไม่แนะนำให้เปิด cron อัตโนมัติทันทีจนกว่า mentor จะยืนยันเวลา backup ที่เหมาะสม

ถ้าต้องการตั้ง backup อัตโนมัติภายหลัง แนวคิดคือ:

- backup วันละครั้งช่วงกลางคืน
- retention 14-30 วัน
- เขียน log ไว้ในไฟล์
- ยังต้องทำ safe restore test เองอย่างน้อยสัปดาห์ละครั้ง

ตัวอย่าง cron สำหรับ backup ทุกวันเวลา 02:30:

```cron
30 2 * * * cd /home/v89dev/apps/Medication_VQA && RETENTION_DAYS=30 bash postgres/scripts/backup_postgres.sh >> /home/v89dev/apps/Medication_VQA/postgres/backups/backup.log 2>&1
```

วิธีเปิด crontab:

```bash
crontab -e
```

หลังตั้ง cron ให้เช็กว่าไฟล์ backup ถูกสร้างจริงในวันถัดไป:

```bash
ls -lt ~/apps/Medication_VQA/postgres/backups/ | head
```

## Routine Summary

| รอบเวลา | สิ่งที่ทำ | คำสั่งหลัก |
| --- | --- | --- |
| หลัง deploy | health check + test LINE | `bash tools/ubuntu_health_check.sh` |
| ทุกวันช่วงแรก | quick health check | `bash tools/ubuntu_health_check.sh` |
| ทุกสัปดาห์ | backup + checksum + safe restore | `backup_postgres.sh`, `test_restore_postgres.sh` |
| ทุกเดือน | disk cleanup review + debug image review + Supabase fallback review | `df -h`, `du -sh`, อ่านเอกสาร fallback |
| เมื่อเกิดปัญหา | health check + logs + incident note | ดู `docs/monitoring_logs_health.md` |

