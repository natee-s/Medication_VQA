import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


STAGING_COLUMNS = [
    "import_batch_id",
    "source_name",
    "source_version",
    "source_url",
    "source_key",
    "tmt_code",
    "fda_registration_no",
    "trade_name",
    "generic_name",
    "active_ingredient",
    "strength",
    "dosage_form",
    "manufacturer",
    "registration_status",
    "raw_payload",
]


AUTO_COLUMN_CANDIDATES = {
    "source_key": (
        "source_key",
        "id",
        "code",
        "tmtid",
        "tmt_id",
        "tmt_code",
        "tpucode",
        "gpu_code",
        "gp_code",
        "vtm_code",
        "register_no",
        "registration_no",
        "license_no",
    ),
    "tmt_code": (
        "tmt_code",
        "tmtid",
        "tmt_id",
        "tmt code",
        "tpucode",
        "gpu_code",
        "gp_code",
        "vtm_code",
        "รหัส tmt",
        "รหัสยา",
    ),
    "fda_registration_no": (
        "fda_registration_no",
        "registration_no",
        "register_no",
        "reg_no",
        "เลขทะเบียน",
        "เลขทะเบียนยา",
        "เลขทะเบียนตำรับยา",
    ),
    "trade_name": (
        "trade_name",
        "brand_name",
        "product_name",
        "tradename",
        "ชื่อการค้า",
        "ชื่อผลิตภัณฑ์",
        "ชื่อยา",
    ),
    "generic_name": (
        "generic_name",
        "generic",
        "generic name",
        "ingredient",
        "active ingredient",
        "ชื่อสามัญ",
        "ตัวยาสำคัญ",
    ),
    "active_ingredient": (
        "active_ingredient",
        "active ingredient",
        "ingredient",
        "ตัวยาสำคัญ",
        "สารสำคัญ",
    ),
    "strength": ("strength", "ความแรง", "ขนาด", "ขนาดยา"),
    "dosage_form": ("dosage_form", "dose_form", "form", "รูปแบบยา", "รูปแบบ"),
    "manufacturer": ("manufacturer", "producer", "licensee", "ผู้ผลิต", "ผู้รับอนุญาต"),
    "registration_status": ("registration_status", "status", "สถานะ", "สถานะทะเบียน"),
}


def normalize_header(value: str) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum() or "\u0e00" <= ch <= "\u0e7f")


def sniff_dialect(path: Path, encoding: str) -> csv.Dialect:
    sample = path.read_text(encoding=encoding, errors="replace")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        return csv.excel


def parse_mapping(values: list[str]) -> dict[str, str]:
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid mapping '{value}'. Use target_column=source_column")
        target, source = value.split("=", 1)
        target = target.strip()
        source = source.strip()
        if target not in STAGING_COLUMNS:
            raise ValueError(f"Unsupported target column '{target}'")
        mapping[target] = source
    return mapping


def detect_columns(fieldnames: list[str], explicit_mapping: dict[str, str]) -> dict[str, str]:
    normalized_lookup = {normalize_header(name): name for name in fieldnames}
    mapping = dict(explicit_mapping)

    for target, candidates in AUTO_COLUMN_CANDIDATES.items():
        if target in mapping:
            continue
        for candidate in candidates:
            match = normalized_lookup.get(normalize_header(candidate))
            if match:
                mapping[target] = match
                break

    return mapping


def clean_value(value: str | None) -> str:
    value = str(value or "").strip()
    if value.lower() in {"nan", "none", "null"}:
        return ""
    return value


def row_value(row: dict[str, str], mapping: dict[str, str], target: str) -> str:
    source_column = mapping.get(target)
    if not source_column:
        return ""
    return clean_value(row.get(source_column))


def sanitize_raw_payload(row: dict) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in row.items():
        if key is None:
            if value:
                payload["_extra_values"] = value
            continue
        payload[str(key)] = clean_value(value)
    return payload


def build_source_key(source_name: str, normalized: dict[str, str], raw_row: dict[str, str]) -> str:
    for key in ("source_key", "tmt_code", "fda_registration_no"):
        value = normalized.get(key)
        if value:
            return value

    stable_text = json.dumps(
        {
            "source_name": source_name,
            "trade_name": normalized.get("trade_name", ""),
            "generic_name": normalized.get("generic_name", ""),
            "active_ingredient": normalized.get("active_ingredient", ""),
            "strength": normalized.get("strength", ""),
            "dosage_form": normalized.get("dosage_form", ""),
            "raw": sanitize_raw_payload(raw_row),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(stable_text.encode("utf-8")).hexdigest()[:32]


def normalize_csv(
    source_path: Path,
    output_path: Path,
    source_name: str,
    source_version: str,
    source_url: str,
    batch_id: str,
    explicit_mapping: dict[str, str],
    encoding: str,
) -> tuple[int, dict[str, str]]:
    dialect = sniff_dialect(source_path, encoding)
    row_count = 0

    with source_path.open("r", encoding=encoding, newline="") as source:
        reader = csv.DictReader(source, dialect=dialect)
        fieldnames = reader.fieldnames or []
        mapping = detect_columns(fieldnames, explicit_mapping)

        with output_path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=STAGING_COLUMNS)
            writer.writeheader()

            for row in reader:
                normalized = {
                    "source_key": row_value(row, mapping, "source_key"),
                    "tmt_code": row_value(row, mapping, "tmt_code"),
                    "fda_registration_no": row_value(row, mapping, "fda_registration_no"),
                    "trade_name": row_value(row, mapping, "trade_name"),
                    "generic_name": row_value(row, mapping, "generic_name"),
                    "active_ingredient": row_value(row, mapping, "active_ingredient"),
                    "strength": row_value(row, mapping, "strength"),
                    "dosage_form": row_value(row, mapping, "dosage_form"),
                    "manufacturer": row_value(row, mapping, "manufacturer"),
                    "registration_status": row_value(row, mapping, "registration_status"),
                }
                normalized["source_key"] = build_source_key(source_name, normalized, row)

                writer.writerow(
                    {
                        "import_batch_id": batch_id,
                        "source_name": source_name,
                        "source_version": source_version,
                        "source_url": source_url,
                        **normalized,
                        "raw_payload": json.dumps(sanitize_raw_payload(row), ensure_ascii=False, sort_keys=True),
                    }
                )
                row_count += 1

    return row_count, mapping


def run_docker_psql(compose_file: str, sql: str, stdin_path: Path | None = None) -> None:
    command = [
        "docker",
        "compose",
        "-f",
        compose_file,
        "exec",
        "-T",
        "db",
        "psql",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        "medication_vqa",
        "-d",
        "medication_vqa",
        "-c",
        sql,
    ]

    if stdin_path:
        with stdin_path.open("rb") as stdin_file:
            subprocess.run(command, stdin=stdin_file, check=True)
    else:
        subprocess.run(command, check=True)


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def import_to_postgres(compose_file: str, normalized_csv_path: Path, batch_id: str) -> None:
    copy_sql = (
        "\\copy public.staging_drug_identity_imports "
        "(" + ", ".join(STAGING_COLUMNS) + ") "
        "from stdin with (format csv, header true, null '')"
    )
    run_docker_psql(compose_file, copy_sql, normalized_csv_path)
    run_docker_psql(
        compose_file,
        f"select * from public.sync_staging_drug_identity_imports({sql_literal(batch_id)});",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import TMT/Thai FDA CSV into staging tables, then sync names/codes into Drug Identity aliases."
    )
    parser.add_argument("--file", required=True, help="Path to source CSV/TSV file")
    parser.add_argument("--source-name", required=True, help="Source name, e.g. TMT or ThaiFDA")
    parser.add_argument("--source-version", default="", help="Source version/date, e.g. TMT_202608")
    parser.add_argument("--source-url", default="", help="Original source URL for audit/reference")
    parser.add_argument("--batch-id", default="", help="Optional import batch id")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="Column mapping as target_column=source_column. Can be repeated.",
    )
    parser.add_argument("--compose-file", default="docker-compose.postgres.yml")
    parser.add_argument("--encoding", default="utf-8-sig", help="Source CSV encoding, e.g. utf-8-sig or cp874")
    parser.add_argument("--dry-run", action="store_true", help="Normalize and report column mapping without importing")
    args = parser.parse_args()

    source_path = Path(args.file).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"CSV not found: {source_path}")

    batch_id = args.batch_id or f"{args.source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    explicit_mapping = parse_mapping(args.map)

    with tempfile.TemporaryDirectory(prefix="drug_identity_import_") as temp_dir:
        normalized_csv_path = Path(temp_dir) / "staging_drug_identity_imports.csv"
        row_count, mapping = normalize_csv(
            source_path,
            normalized_csv_path,
            args.source_name,
            args.source_version,
            args.source_url,
            batch_id,
            explicit_mapping,
            args.encoding,
        )

        print(f"Prepared {row_count} rows for batch {batch_id}")
        print("Column mapping:")
        for target in sorted(mapping):
            print(f"  {target} <= {mapping[target]}")

        if args.dry_run:
            print("Dry run only. No database changes were made.")
            return 0

        import_to_postgres(args.compose_file, normalized_csv_path, batch_id)

    print("External drug identity import completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
