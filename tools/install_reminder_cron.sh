#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CRON_SCHEDULE="${CRON_SCHEDULE:-* * * * *}"
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/reminder_cron.log}"
REMINDER_CRON_URL="${REMINDER_CRON_URL:-http://127.0.0.1:17080/cron/check-reminder}"

MARKER_BEGIN="# BEGIN Medication_VQA reminder cron"
MARKER_END="# END Medication_VQA reminder cron"
RUN_SCRIPT="${PROJECT_DIR}/tools/run_reminder_cron.sh"

if ! command -v crontab >/dev/null 2>&1; then
  echo "ERROR: crontab command not found. Please ask the server admin to install cron." >&2
  exit 1
fi

if [ ! -f "${RUN_SCRIPT}" ]; then
  echo "ERROR: reminder cron runner not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"
chmod +x "${RUN_SCRIPT}"

cron_command="cd ${PROJECT_DIR} && REMINDER_CRON_URL=${REMINDER_CRON_URL} bash tools/run_reminder_cron.sh >> ${LOG_FILE} 2>&1"

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

echo "Installed Medication_VQA reminder cron."
echo "Schedule: ${CRON_SCHEDULE}"
echo "Reminder URL: ${REMINDER_CRON_URL}"
echo "Log file: ${LOG_FILE}"
echo "Server time now: $(date)"
echo
echo "Current managed cron block:"
crontab -l | sed -n "/${MARKER_BEGIN}/,/${MARKER_END}/p"
