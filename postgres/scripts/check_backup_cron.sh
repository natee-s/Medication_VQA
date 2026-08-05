#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-${PROJECT_DIR}/postgres/backups}"
LOG_FILE="${LOG_FILE:-${BACKUP_DIR}/backup.log}"

MARKER_BEGIN="# BEGIN Medication_VQA PostgreSQL backup"
MARKER_END="# END Medication_VQA PostgreSQL backup"

if ! command -v crontab >/dev/null 2>&1; then
  echo "ERROR: crontab command not found. Please ask the server admin to install cron." >&2
  exit 1
fi

echo "Server time now:"
date
echo

echo "Managed cron block:"
if ! crontab -l 2>/dev/null | sed -n "/${MARKER_BEGIN}/,/${MARKER_END}/p" | grep -q .; then
  echo "No Medication_VQA backup cron block found."
else
  crontab -l | sed -n "/${MARKER_BEGIN}/,/${MARKER_END}/p"
fi

echo
echo "Recent backup files:"
if [ -d "${BACKUP_DIR}" ]; then
  ls -lt "${BACKUP_DIR}" | head -20
else
  echo "Backup directory not found: ${BACKUP_DIR}"
fi

echo
echo "Recent backup log:"
if [ -f "${LOG_FILE}" ]; then
  tail -80 "${LOG_FILE}"
else
  echo "Backup log not found yet: ${LOG_FILE}"
fi
