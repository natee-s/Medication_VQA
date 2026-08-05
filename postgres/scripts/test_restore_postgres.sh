#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKUP_FILE="${1:-}"
IMAGE="${POSTGRES_TEST_IMAGE:-pgvector/pgvector:pg16}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CONTAINER_NAME="${TEST_RESTORE_CONTAINER:-medication-vqa-postgres-restore-test-${TIMESTAMP}}"
VOLUME_NAME="${TEST_RESTORE_VOLUME:-medication_vqa_restore_test_${TIMESTAMP}}"

if [ -z "${BACKUP_FILE}" ]; then
  echo "Usage: bash postgres/scripts/test_restore_postgres.sh <backup-file.dump>" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if [ ! -f "${BACKUP_FILE}" ]; then
  echo "ERROR: Backup file not found: ${BACKUP_FILE}" >&2
  exit 1
fi

if [ ! -f ".env" ]; then
  echo "ERROR: .env file not found in ${PROJECT_DIR}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
. ./.env
set +a

POSTGRES_DB="${POSTGRES_DB:-medication_vqa}"
POSTGRES_USER="${POSTGRES_USER:-medication_vqa}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set in .env}"

cleanup() {
  if [ "${KEEP_TEST_RESTORE:-}" = "YES" ]; then
    echo "Keeping test restore container because KEEP_TEST_RESTORE=YES"
    echo "Container: ${CONTAINER_NAME}"
    echo "Volume: ${VOLUME_NAME}"
    return
  fi

  echo "Cleaning up test restore container and volume..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker volume rm "${VOLUME_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Creating temporary restore volume: ${VOLUME_NAME}"
docker volume create "${VOLUME_NAME}" >/dev/null

echo "Starting temporary restore container: ${CONTAINER_NAME}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  -e POSTGRES_DB="${POSTGRES_DB}" \
  -e POSTGRES_USER="${POSTGRES_USER}" \
  -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  -v "${VOLUME_NAME}:/var/lib/postgresql/data" \
  "${IMAGE}" >/dev/null

echo "Waiting for temporary PostgreSQL to be ready..."
ready="false"
for _ in $(seq 1 40); do
  if docker exec "${CONTAINER_NAME}" pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" >/dev/null 2>&1; then
    ready="true"
    break
  fi
  sleep 2
done

if [ "${ready}" != "true" ]; then
  echo "ERROR: Temporary PostgreSQL did not become ready in time." >&2
  docker logs "${CONTAINER_NAME}" >&2 || true
  exit 1
fi

echo "Restoring backup into temporary database: ${BACKUP_FILE}"
cat "${BACKUP_FILE}" | docker exec -i "${CONTAINER_NAME}" pg_restore \
  -U "${POSTGRES_USER}" \
  -d "${POSTGRES_DB}" \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges

echo "Checking restored data..."
docker exec "${CONTAINER_NAME}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -v ON_ERROR_STOP=1 <<'SQL'
select 'Medication_VQA rows' as check_name, count(*)::text as result
from public."Medication_VQA";

select 'Medication_VQA embeddings' as check_name, count(*)::text as result
from public."Medication_VQA"
where embedding is not null;

select 'user_profiles rows' as check_name, count(*)::text as result
from public.user_profiles;

select 'reminder_schedules rows' as check_name, count(*)::text as result
from public.reminder_schedules;

select 'match_symptoms rows' as check_name, count(*)::text as result
from public.match_symptoms(
  (select embedding from public."Medication_VQA" where embedding is not null limit 1),
  0.1,
  3
);
SQL

echo "Safe restore test completed successfully."
