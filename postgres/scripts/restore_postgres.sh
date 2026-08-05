#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.postgres.yml}"
BACKUP_FILE="${1:-}"

if [ -z "${BACKUP_FILE}" ]; then
  echo "Usage: CONFIRM_RESTORE=YES bash postgres/scripts/restore_postgres.sh <backup-file.dump>" >&2
  exit 1
fi

if [ "${CONFIRM_RESTORE:-}" != "YES" ]; then
  echo "ERROR: Restore is destructive. Re-run with CONFIRM_RESTORE=YES." >&2
  echo "Example: CONFIRM_RESTORE=YES bash postgres/scripts/restore_postgres.sh ${BACKUP_FILE}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if [ ! -f "${BACKUP_FILE}" ]; then
  echo "ERROR: Backup file not found: ${BACKUP_FILE}" >&2
  exit 1
fi

echo "Checking PostgreSQL container..."
docker compose -f "${COMPOSE_FILE}" exec -T db sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'

echo "Restoring backup: ${BACKUP_FILE}"
cat "${BACKUP_FILE}" | docker compose -f "${COMPOSE_FILE}" exec -T db sh -c \
  'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner --no-privileges'

echo "Restore complete."
