#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/reminder_cron.log}"
REMINDER_CRON_URL="${REMINDER_CRON_URL:-http://127.0.0.1:17080/cron/check-reminder}"
CURL_TIMEOUT_SECONDS="${CURL_TIMEOUT_SECONDS:-20}"

MARKER_BEGIN="# BEGIN Medication_VQA reminder cron"
MARKER_END="# END Medication_VQA reminder cron"

if ! command -v crontab >/dev/null 2>&1; then
  echo "ERROR: crontab command not found. Please ask the server admin to install cron." >&2
  exit 1
fi

echo "Server time now:"
date
echo

echo "Managed reminder cron block:"
if ! crontab -l 2>/dev/null | sed -n "/${MARKER_BEGIN}/,/${MARKER_END}/p" | grep -q .; then
  echo "No Medication_VQA reminder cron block found."
else
  crontab -l | sed -n "/${MARKER_BEGIN}/,/${MARKER_END}/p"
fi

echo
echo "Manual reminder endpoint check:"
curl -fsS --max-time "${CURL_TIMEOUT_SECONDS}" "${REMINDER_CRON_URL}"
echo

echo
echo "Recent reminder cron log:"
if [ -f "${LOG_FILE}" ]; then
  tail -80 "${LOG_FILE}"
else
  echo "Reminder cron log not found yet: ${LOG_FILE}"
fi
