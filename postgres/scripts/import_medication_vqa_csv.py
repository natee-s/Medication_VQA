import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_COLUMNS = [
    "source_row_number",
    "source_item_id",
    "label_name",
    "initial",
    "trade_name",
    "generic_name",
    "indication",
    "dosage_frequency",
    "instruction_time",
    "precaution",
    "rag_text",
    "embedding",
]


def normalize_embedding(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError("embedding must be a bracketed vector")

    items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
    if len(items) != 768:
        raise ValueError(f"embedding has {len(items)} dimensions, expected 768")

    # pgvector accepts the same bracketed representation.
    return "[" + ",".join(items) + "]"


def build_normalized_csv(source_path: Path, output_path: Path) -> tuple[int, int]:
    total_rows = 0
    embedding_rows = 0

    with source_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(missing)}")

        with output_path.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=REQUIRED_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                total_rows += 1
                normalized = {column: row.get(column, "") for column in REQUIRED_COLUMNS}
                embedding = normalize_embedding(normalized.get("embedding", ""))
                if embedding:
                    embedding_rows += 1
                normalized["embedding"] = embedding
                writer.writerow(normalized)

    return total_rows, embedding_rows


def run_psql(sql: str) -> None:
    command = ["psql", "-v", "ON_ERROR_STOP=1", "-d", os.environ.get("POSTGRES_DB", "medication_vqa"), "-c", sql]
    subprocess.run(command, check=True)


def copy_into_postgres(normalized_csv_path: Path) -> None:
    table = 'public."Medication_VQA"'
    run_psql(f"truncate table {table};")

    copy_sql = (
        f"\\copy {table} "
        "(source_row_number, source_item_id, label_name, initial, trade_name, generic_name, "
        "indication, dosage_frequency, instruction_time, precaution, rag_text, embedding) "
        f"from '{normalized_csv_path.as_posix()}' with (format csv, header true, null '')"
    )
    subprocess.run(
        ["psql", "-v", "ON_ERROR_STOP=1", "-d", os.environ.get("POSTGRES_DB", "medication_vqa"), "-c", copy_sql],
        check=True,
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python postgres/scripts/import_medication_vqa_csv.py /path/to/Medication_VQA_rows.csv", file=sys.stderr)
        return 2

    source_path = Path(sys.argv[1]).expanduser().resolve()
    if not source_path.exists():
        print(f"CSV not found: {source_path}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="medication_vqa_import_") as temp_dir:
        normalized_csv_path = Path(temp_dir) / "Medication_VQA_rows.normalized.csv"
        total_rows, embedding_rows = build_normalized_csv(source_path, normalized_csv_path)
        print(f"Prepared {total_rows} rows ({embedding_rows} rows with embeddings)")
        copy_into_postgres(normalized_csv_path)

    run_psql(
        """
        select
          count(*) as medication_rows,
          count(embedding) as embedding_rows
        from public."Medication_VQA";
        """
    )
    print("Medication_VQA import completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
