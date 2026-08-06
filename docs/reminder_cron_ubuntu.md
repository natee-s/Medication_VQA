# Reminder Cron On Ubuntu

ระบบเตือนกินยาจะไม่ทำงานเองตลอดเวลา แต่ต้องมีตัวเรียก endpoint นี้ซ้ำ ๆ:

```text
http://127.0.0.1:17080/cron/check-reminder
```

หลังย้ายจาก Render/cron-job.org มา Ubuntu แนะนำให้ให้ Ubuntu cron เป็นตัวเรียกเองทุก 1 นาที

## Install

รันบน Ubuntu Server:

```bash
cd ~/apps/Medication_VQA
bash tools/install_reminder_cron.sh
```

ค่า default:

- เรียกทุก 1 นาที: `* * * * *`
- ใช้ URL ภายใน server: `http://127.0.0.1:17080/cron/check-reminder`
- เก็บ log ที่ `logs/reminder_cron.log`

## Check

```bash
cd ~/apps/Medication_VQA
bash tools/check_reminder_cron.sh
```

คำสั่งนี้จะแสดง:

- เวลา server ปัจจุบัน
- cron block ที่ติดตั้งไว้
- ผลการยิง endpoint แบบ manual
- log ล่าสุดของ reminder cron

## Manual Test

ถ้าต้องการยิงเตือน 1 ครั้งทันที:

```bash
cd ~/apps/Medication_VQA
bash tools/run_reminder_cron.sh
```

ถ้า endpoint ใช้งานได้ ควรเห็น JSON จาก `/cron/check-reminder`

## Important

หลังติดตั้ง Ubuntu cron แล้ว ให้ปิด job เดิมใน cron-job.org เพื่อกันการยิงซ้ำและส่ง reminder ซ้ำ

อย่าเปิดทั้ง 2 ตัวพร้อมกัน:

- cron-job.org
- Ubuntu cron

ให้เหลือ Ubuntu cron เป็นตัวหลักตัวเดียว
