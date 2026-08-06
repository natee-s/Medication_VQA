#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMINDER_CRON_URL="${REMINDER_CRON_URL:-http://127.0.0.1:17080/cron/check-reminder}"
CURL_TIMEOUT_SECONDS="${CURL_TIMEOUT_SECONDS:-50}"
LOCK_FILE="${LOCK_FILE:-/tmp/medication_vqa_reminder_cron.lock}"

run_check() {
  echo "BEGIN Medication_VQA reminder check: $(date)"
  echo "URL: ${REMINDER_CRON_URL}"
  curl -fsS --max-time "${CURL_TIMEOUT_SECONDS}" "${REMINDER_CRON_URL}"
  echo
  echo "END Medication_VQA reminder check: $(date)"
}

cd "${PROJECT_DIR}"

if command -v flock >/dev/null 2>&1; then
  (
    if ! flock -n 9; then
      echo "SKIP Medication_VQA reminder check: previous run still active at $(date)"
      exit 0
    fi
    run_check
  ) 9>"${LOCK_FILE}"
else
  run_check
fi
