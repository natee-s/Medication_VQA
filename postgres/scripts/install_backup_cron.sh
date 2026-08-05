#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CRON_SCHEDULE="${CRON_SCHEDULE:-30 2 * * *}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/postgres/backups/backup.log}"

MARKER_BEGIN="# BEGIN Medication_VQA PostgreSQL backup"
MARKER_END="# END Medication_VQA PostgreSQL backup"
BACKUP_SCRIPT="${PROJECT_DIR}/postgres/scripts/backup_postgres.sh"

if ! command -v crontab >/dev/null 2>&1; then
  echo "ERROR: crontab command not found. Please ask the server admin to install cron." >&2
  exit 1
fi

if [ ! -f "${BACKUP_SCRIPT}" ]; then
  echo "ERROR: backup script not found: ${BACKUP_SCRIPT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"
chmod +x "${BACKUP_SCRIPT}"

cron_command="cd ${PROJECT_DIR} && RETENTION_DAYS=${RETENTION_DAYS} bash postgres/scripts/backup_postgres.sh >> ${LOG_FILE} 2>&1"

tmp_current="$(mktemp)"
tmp_next="$(mktemp)"
trap 'rm -f "${tmp_current}" "${tmp_next}"' EXIT

crontab -l > "${tmp_current}" 2>/dev/null || true

awk -v begin="${MARKER_BEGIN}" -v end="${MARKER_END}" '
  $0 == begin {skip=1; next}
  $0 == end {skip=0; next}
  skip != 1 {print}
' "${tmp_current}" > "${tmp_next}"

{
  cat "${tmp_next}"
  echo "${MARKER_BEGIN}"
  echo "${CRON_SCHEDULE} ${cron_command}"
  echo "${MARKER_END}"
} | crontab -

echo "Installed PostgreSQL backup cron."
echo "Schedule: ${CRON_SCHEDULE}"
echo "Retention days: ${RETENTION_DAYS}"
echo "Log file: ${LOG_FILE}"
echo "Server time now: $(date)"
echo
echo "Current managed cron block:"
crontab -l | sed -n "/${MARKER_BEGIN}/,/${MARKER_END}/p"
