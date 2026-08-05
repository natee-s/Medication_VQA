#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.postgres.yml}"
BACKUP_DIR="${BACKUP_DIR:-${PROJECT_DIR}/postgres/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"

cd "${PROJECT_DIR}"
mkdir -p "${BACKUP_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_file="${BACKUP_DIR}/medication_vqa_${timestamp}.dump"
checksum_file="${backup_file}.sha256"

echo "Checking PostgreSQL container..."
docker compose -f "${COMPOSE_FILE}" exec -T db sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'

echo "Creating backup: ${backup_file}"
docker compose -f "${COMPOSE_FILE}" exec -T db sh -c \
  'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom --no-owner --no-privileges' \
  > "${backup_file}"

if [ ! -s "${backup_file}" ]; then
  echo "ERROR: Backup file is empty: ${backup_file}" >&2
  exit 1
fi

sha256sum "${backup_file}" > "${checksum_file}"

echo "Removing backups older than ${RETENTION_DAYS} days..."
find "${BACKUP_DIR}" -type f \( -name "*.dump" -o -name "*.dump.sha256" \) -mtime +"${RETENTION_DAYS}" -delete

echo "Backup complete."
echo "File: ${backup_file}"
echo "Checksum: ${checksum_file}"
