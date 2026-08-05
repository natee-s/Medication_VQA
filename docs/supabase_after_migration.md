# Supabase After Migration

เอกสารนี้สรุปวิธีจัดการ Supabase หลังระบบ production ย้ายมาใช้ PostgreSQL บน Ubuntu แล้ว

เอกสารที่เกี่ยวข้อง:

- `docs/postgres_migration_runbook.md`
- `docs/maintenance_routine.md`

## สถานะปัจจุบัน

ตอนนี้ production ใช้:

```env
DB_BACKEND=postgres
```

Supabase ยังไม่ควรลบ เพราะยังมีประโยชน์ 3 อย่าง:

1. เป็น fallback ถ้า PostgreSQL มีปัญหา
2. เป็น backup อ้างอิงของข้อมูลเดิมช่วงแรกหลัง migration
3. ใช้เปรียบเทียบข้อมูล/พฤติกรรม ถ้าเจอ bug ในระบบใหม่

## สิ่งที่ควรทำตอนนี้

- เก็บ Supabase project ไว้ก่อน
- เก็บ `SUPABASE_URL` และ `SUPABASE_KEY` ใน `.env` บน Ubuntu ต่อไป
- ห้าม commit `SUPABASE_KEY` จริงขึ้น GitHub
- ใช้ PostgreSQL เป็น backend หลักต่อไป
- ทำ PostgreSQL backup เป็นประจำ
- ถ้ามี bug เรื่อง database ให้แก้ PostgreSQL ก่อน แล้วค่อย rollback เฉพาะกรณีจำเป็น

## สิ่งที่ยังไม่ควรทำ

- ยังไม่ควรลบ Supabase project
- ยังไม่ควรลบ table หรือ function ใน Supabase
- ยังไม่ควรลบ `SUPABASE_URL` / `SUPABASE_KEY` จาก `.env`
- ยังไม่ควรปิด Supabase ทันทีหลัง migration วันแรก ๆ

## วิธี Rollback กลับ Supabase

ใช้เมื่อ PostgreSQL มีปัญหาจนระบบใช้งานไม่ได้ เช่น:

- PostgreSQL container ล่ม
- ข้อมูลยาใน PostgreSQL หาย
- function `match_symptoms()` พัง
- user ใช้งาน flow สำคัญไม่ได้

ขั้นตอน:

```bash
cd ~/apps/Medication_VQA
nano .env
```

เปลี่ยน:

```env
DB_BACKEND=postgres
```

เป็น:

```env
DB_BACKEND=supabase
```

บันทึกไฟล์ แล้ว restart main app:

```bash
docker compose -f docker-compose.ubuntu.yml up -d --build main
```

ทดสอบ:

```bash
curl https://ginya.v89tech.com/
curl https://ginya.v89tech.com/test-db/AMITRIPTYLINE
```

จากนั้นทดสอบใน LINE:

- พิมพ์อาการ
- ส่งรูปฉลากยา
- ถ่ายผ่าน LIFF Camera
- กด Rich menu

## วิธี Switch กลับ PostgreSQL

หลังแก้ PostgreSQL แล้ว:

```bash
cd ~/apps/Medication_VQA
nano .env
```

เปลี่ยนกลับ:

```env
DB_BACKEND=postgres
```

restart main:

```bash
docker compose -f docker-compose.ubuntu.yml up -d --build main
```

เช็ก PostgreSQL:

```bash
docker compose -f docker-compose.postgres.yml exec db pg_isready -U medication_vqa -d medication_vqa
docker compose -f docker-compose.postgres.yml exec db psql -U medication_vqa -d medication_vqa -c 'select count(*) from public."Medication_VQA";'
```

## เงื่อนไขก่อนพิจารณาปิด Supabase

ควรรอให้ครบทุกข้อก่อน:

- PostgreSQL ใช้งาน production ต่อเนื่องอย่างน้อย 2-4 สัปดาห์
- Backup PostgreSQL ทำงานได้จริง
- Safe restore test ผ่านแล้ว
- reminder cron ทำงานจริงตามเวลา
- image upload และ LIFF Camera ใช้งานได้ต่อเนื่อง
- ไม่มี bug สำคัญที่ต้อง rollback ไป Supabase
- mentor เห็นด้วยกับแผนปิด fallback

## ทางเลือกหลังระบบนิ่ง

### ทางเลือก A: เก็บ Supabase ต่อเป็น cold backup

ข้อดี:

- rollback ง่ายที่สุด
- เหมาะกับช่วงฝึกงาน/โปรเจกต์ที่ยังพัฒนาอยู่

ข้อเสีย:

- ต้องดูแล secret เพิ่ม
- อาจมี cost หรือ policy retention ในอนาคต

### ทางเลือก B: Export Supabase เก็บไว้ แล้วปิด project

ข้อดี:

- ลดระบบที่ต้องดูแล
- ลดความสับสนว่า source of truth อยู่ที่ไหน

ข้อเสีย:

- rollback ไม่เร็วเท่าเดิม
- ต้องมั่นใจว่า PostgreSQL backup/restore พร้อมจริง

## Recommendation

ตอนนี้แนะนำเลือกทางเลือก A ก่อน:

```text
PostgreSQL = primary production database
Supabase = temporary fallback / cold backup
```

หลัง production PostgreSQL เสถียร 2-4 สัปดาห์ ค่อยกลับมาประเมินว่าจะปิด Supabase หรือเก็บไว้เป็น backup ระยะยาว
