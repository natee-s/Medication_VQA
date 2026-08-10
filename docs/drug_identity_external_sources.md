# Drug Identity External Sources

This project uses external drug datasets only for identity, alias, and registration reference.
External sources must not directly replace the pharmacy label instructions shown to users.

## Concept

```text
TMT / Thai FDA CSV
  -> staging_drug_identity_imports
  -> drug_identity / drug_aliases / drug_identity_sources
  -> better OCR/name matching
  -> final user-facing medication details still come from Medication_VQA
```

## 1. Run Migrations

Run these on the Ubuntu server from the project folder:

```bash
cd ~/apps/Medication_VQA

docker compose -f docker-compose.postgres.yml exec -T db psql \
  -U medication_vqa \
  -d medication_vqa \
  < postgres/migrations/202608100001_add_drug_identity_layer.sql

docker compose -f docker-compose.postgres.yml exec -T db psql \
  -U medication_vqa \
  -d medication_vqa \
  < postgres/migrations/202608100002_add_external_drug_identity_sources.sql
```

## 2. Prepare Source Files

Use CSV/TSV files for import. If the source file is XLS/XLSX, open it and export/save as CSV first.

Recommended source names:

- `TMT`
- `ThaiFDA`

For the first TMT import, convert:

```text
MasterTMT_20260615.xls -> postgres/import/tmt/MasterTMT_20260615.csv
```

The `postgres/import/` folder is ignored by Git, so raw external datasets are not committed.

## 3. Dry Run Import

Use dry run first to check column mapping without changing the database:

```bash
python postgres/scripts/import_external_drug_identity_csv.py \
  --file postgres/import/tmt/MasterTMT_20260615.csv \
  --source-name TMT \
  --source-version TMTRF20260615 \
  --source-url "https://this.or.th/service/tmt/" \
  --dry-run
```

Expected MasterTMT mapping:

```text
active_ingredient <= ActiveIngredient
dosage_form <= Dosageform
generic_name <= ActiveIngredient
manufacturer <= Manufacturer
registration_status <= Status
source_key <= TPUCode
strength <= Strength
tmt_code <= TPUCode
trade_name <= TradeName
```

If column names are not detected correctly, pass mappings manually:

```bash
python postgres/scripts/import_external_drug_identity_csv.py \
  --file /path/to/thai_fda.csv \
  --source-name ThaiFDA \
  --source-version FDA_YYYYMMDD \
  --map trade_name=ชื่อผลิตภัณฑ์ \
  --map generic_name=ตัวยาสำคัญ \
  --map fda_registration_no=เลขทะเบียนตำรับยา \
  --map manufacturer=ผู้รับอนุญาต \
  --dry-run
```

Supported target columns:

- `source_key`
- `tmt_code`
- `fda_registration_no`
- `trade_name`
- `generic_name`
- `active_ingredient`
- `strength`
- `dosage_form`
- `manufacturer`
- `registration_status`

## 4. Real Import

Remove `--dry-run` after the mapping looks correct:

```bash
python postgres/scripts/import_external_drug_identity_csv.py \
  --file postgres/import/tmt/MasterTMT_20260615.csv \
  --source-name TMT \
  --source-version TMTRF20260615 \
  --source-url "https://this.or.th/service/tmt/"
```

The script will:

1. Normalize the CSV into staging format.
2. Copy rows into `staging_drug_identity_imports`.
3. Run `sync_staging_drug_identity_imports(batch_id)`.
4. Create/update identity rows and aliases.

## 5. Verify

```bash
docker compose -f docker-compose.postgres.yml exec db psql \
  -U medication_vqa \
  -d medication_vqa \
  -c "select source_name, count(*) from public.drug_identity group by source_name order by source_name;"

docker compose -f docker-compose.postgres.yml exec db psql \
  -U medication_vqa \
  -d medication_vqa \
  -c "select source_name, count(*) from public.drug_aliases group by source_name order by source_name;"

docker compose -f docker-compose.postgres.yml exec db psql \
  -U medication_vqa \
  -d medication_vqa \
  -c "select tmt_code, trade_name, active_ingredient, strength, dosage_form from public.drug_identity where source_name = 'TMT' limit 5;"
```

## Safety Rule

External rows from TMT/Thai FDA should improve matching only. The bot should still build medication Flex Messages from `Medication_VQA`, because that table contains the pharmacy-specific label instructions.
