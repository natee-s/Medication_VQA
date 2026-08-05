# Repository Cleanup Audit

เอกสารนี้สรุปสถานะ Git/Repo หลังย้ายระบบขึ้น Ubuntu Server และ PostgreSQL แล้ว

วันที่ audit: 2026-08-05

## สรุปสั้น

Repo ยังใช้งานได้ปกติ แต่มีไฟล์ค้างบางกลุ่มที่ควรจัดการก่อน development รอบถัดไป เพื่อให้ production repo สะอาดและไม่เผลอ commit ไฟล์ใหญ่หรือไฟล์ debug

## สถานะที่พบ

### ไฟล์ที่ถูกแก้/ลบใน Git

```text
D models/yolo_obb/best.pt
M tests/test_debug_pdpa_image.py
```

ความหมาย:

- `models/yolo_obb/best.pt` เป็น model file ที่เคยถูก track ใน Git มาก่อน แต่ตอนนี้ local ไม่มีไฟล์นี้แล้ว
- ปัจจุบัน production ใช้ `best_round3.pt` ผ่าน `YOLO_OBB_MODEL_PATH`
- `.gitignore` กัน `*.pt` แล้ว แต่ไฟล์ที่เคยถูก track มาก่อนยังคงโผล่ใน Git ได้
- `tests/test_debug_pdpa_image.py` มีการแก้ไขจากงานก่อนหน้า ควร review ก่อน commit

### ไฟล์ untracked ที่พบ

```text
docs/superpowers/plans/2026-07-20-minimal-project-organization.md
docs/superpowers/plans/2026-07-20-multi-language-core-flow.md
docs/superpowers/plans/2026-07-21-light-image-preprocessing-pdpa.md
sample/
tests/test_label.jpg
tools/augment_yolo_obb_dataset.py
```

การตีความ:

- `tools/augment_yolo_obb_dataset.py` เป็น tool ที่มีประโยชน์สำหรับ YOLO dataset augmentation และควรพิจารณา commit ถ้าจะใช้ต่อ
- `tests/test_label.jpg` เป็น test asset ที่เคยถูกอ้างถึงในแผน project organization แต่ยังไม่ถูก track
- `sample/` เป็นรูปตัวอย่าง/ภาพ screenshot จำนวนมาก ไม่ควร commit เข้า production repo โดยไม่จำเป็น
- `docs/superpowers/plans/*.md` เป็น plan เก่าจาก workflow ก่อนหน้า ควรตัดสินใจว่าจะเก็บเป็น documentation หรือ ignore

### ไฟล์สำคัญที่ถูก ignore แล้ว

`.gitignore` กันไฟล์กลุ่มนี้แล้ว:

- `.env`, `.env.*`
- `.venv/`
- `datasets/`
- `debug_pdpa/`
- `debug_yolo_local/`
- `hf_pdpa_masking_service/`
- `postgres/import/`
- `postgres/backups/`
- `test/local_pdpa_debug/`
- `runs/`
- `*.pt`
- `*.onnx`
- `*.log`

จาก audit ไม่พบว่า `.env`, backup, debug output, dataset, หรือ model รอบใหม่ถูกเตรียม commit

## คำแนะนำ Cleanup

### 1. เอา legacy model ออกจาก Git

แนะนำให้ stage deletion ของ `models/yolo_obb/best.pt` เพื่อให้ model file ไม่อยู่ใน repo อีกต่อไป

```bash
git add -u models/yolo_obb/best.pt
```

หมายเหตุ: คำสั่งนี้ไม่ได้ลบ `best_round3.pt` บนเครื่องหรือบน server เพราะไฟล์นั้นถูก ignore และอยู่นอก Git

### 2. Review test ที่แก้ไว้

ดู diff:

```bash
git diff -- tests/test_debug_pdpa_image.py
```

ถ้าเป็น test ที่ต้องการเก็บ:

```bash
git add tests/test_debug_pdpa_image.py
```

ถ้าไม่ต้องการเก็บ ต้องตัดสินใจเองก่อน revert เพราะอาจเป็นงานจากรอบก่อนหน้า

### 3. ตัดสินใจ `tools/augment_yolo_obb_dataset.py`

ถ้าจะใช้ต่อสำหรับ train YOLO รอบถัดไป แนะนำให้ commit:

```bash
git add tools/augment_yolo_obb_dataset.py
```

ถ้าไม่ใช้ต่อ ให้เก็บไว้นอก repo หรือ ignore ภายหลัง

### 4. ตัดสินใจ `tests/test_label.jpg`

ถ้า test หรือ manual debug ยังต้องใช้ image นี้:

```bash
git add tests/test_label.jpg
```

ถ้าไม่ใช้แล้ว ไม่ควร commit รูปนี้

### 5. ไม่ควร commit `sample/`

`sample/` เป็นภาพ screenshot/debug sample จำนวนมาก แนะนำให้ไม่ commit เข้า production repo

ถ้าต้องการกันไม่ให้โผล่ใน `git status` อีก ให้เพิ่มใน `.gitignore`:

```gitignore
sample/
```

### 6. Review `docs/superpowers/plans/`

ถ้าต้องการเก็บ history การวางแผนไว้ใน repo สามารถ commit ได้

ถ้าไม่ต้องการ ควรไม่ commit และอาจ ignore plan generated files ภายหลัง

## Commit ชุดที่แนะนำ

ถ้าต้องการ cleanup แบบ conservative:

```bash
git add docs/repo_cleanup_audit.md
git add -u models/yolo_obb/best.pt
git add tools/augment_yolo_obb_dataset.py
git commit -m "Document repository cleanup audit"
```

อย่าเพิ่ง add `sample/`, `.env`, `datasets/`, `postgres/backups/`, `debug_*`, หรือ model `.pt` รอบใหม่

