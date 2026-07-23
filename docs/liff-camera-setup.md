# LIFF Camera Guideline MVP

ฟีเจอร์นี้เพิ่มหน้าเว็บกล้องสำหรับเปิดใน LINE LIFF เพื่อช่วยให้ผู้ใช้ถ่ายรูปฉลากยาให้อยู่ในกรอบเดียวกันก่อนส่งเข้า backend

## URL ที่ใช้ทดสอบ

หลัง deploy บน Render แล้ว หน้า camera จะอยู่ที่:

```text
https://<your-render-service-url>/liff/camera
```

ตัวอย่างถ้า service คือ `https://medication-docker.onrender.com`:

```text
https://medication-docker.onrender.com/liff/camera
```

## วิธีตั้งค่าใน LINE Developers Console

1. เข้า LINE Developers Console
2. เข้า Provider และ Channel ของ LINE Login หรือสร้าง LINE Login Channel ใหม่
3. ไปที่เมนู LIFF
4. กด Add LIFF app
5. ตั้งค่า Size เป็น `Full`
6. ใส่ Endpoint URL เป็น:

```text
https://<your-render-service-url>/liff/camera
```

7. กด Save
8. LINE จะให้ LIFF URL รูปแบบนี้:

```text
https://liff.line.me/<LIFF_ID>
```

9. นำ LIFF URL นี้ไปตั้งเป็น action ของปุ่ม Rich Menu เช่นปุ่ม `ถ่ายฉลากยา`

## วิธีทดสอบบนมือถือ

1. เปิด LINE OA บนมือถือ
2. กดปุ่ม Rich Menu ที่ตั้งเป็น LIFF URL
3. ระบบจะเปิดหน้ากล้อง
4. อนุญาตสิทธิ์กล้อง
5. วางฉลากยาให้อยู่ในกรอบแนวนอน
6. ให้เส้นคั่นบนฉลากอยู่ใกล้เส้น guideline ด้านในกรอบ
7. กด `ถ่ายรูป`
8. ตรวจ preview
9. ถ้ารูปไม่ดี กด `ถ่ายใหม่`
10. ถ้ารูปดี กด `ส่งรูป`

## ไฟล์รูปที่ upload ไปอยู่ที่ไหน

ตอนนี้เป็น MVP สำหรับทดสอบการถ่ายและส่งรูปก่อน ระบบจะบันทึกรูปที่ upload ไว้ใน debug folder ของ server

ค่า default:

```text
/tmp/liff_uploads
```

ถ้าต้องการกำหนดเอง ให้ตั้ง Environment Variable:

```text
LIFF_UPLOAD_DEBUG_DIR=/tmp/liff_uploads
```

ระบบจะบันทึกไฟล์ metadata คู่กับรูปด้วย เช่น:

```text
liff_label_20260723123000_ab12cd34.jpg
liff_label_20260723123000_ab12cd34.json
```

ไฟล์ `.json` จะเก็บข้อมูล เช่น `line_user_id`, ขนาดไฟล์ และเวลาที่ upload เพื่อใช้เชื่อมต่อกับขั้นตอน push message กลับไปหา user ใน phase ถัดไป

## Environment Variables สำหรับ LIFF

เพิ่มตัวนี้ใน Render:

```text
LIFF_ID=<LIFF_ID ที่ได้จาก LINE Developers Console>
```

ถ้าไม่ใส่ `LIFF_ID` หน้าเว็บยังเปิดกล้องและ upload debug ได้ แต่จะไม่สามารถดึง `line_user_id` จาก LINE profile ได้

## ค่าของกรอบถ่ายรูป

```text
Guide aspect ratio: 1.344
Header divider: 25% จากขอบบน
Output image size: 1344 x 1000
Camera: กล้องหลัง
Upload format: JPEG
```

## หมายเหตุ

MVP นี้ยังไม่ได้เชื่อมรูปที่ upload จาก LIFF เข้า OCR/RAG/Flex Message โดยตรง จุดประสงค์รอบนี้คือให้ทดสอบก่อนว่าเปิดกล้องได้ กรอบใช้งานได้ crop ได้ และ backend รับรูปได้ถูกต้อง
