#!/usr/bin/env bash
set -u

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-https://ginya.v89tech.com}"
TEST_DRUG="${TEST_DRUG:-AMITRIPTYLINE}"
CURL_TIMEOUT_SECONDS="${CURL_TIMEOUT_SECONDS:-20}"

MAIN_LOCAL_URL="${MAIN_LOCAL_URL:-http://127.0.0.1:17080/}"
PDPA_HEALTH_URL="${PDPA_HEALTH_URL:-http://127.0.0.1:17081/health}"

failed=0

section() {
  printf "\n== %s ==\n" "$1"
}

pass() {
  printf "PASS: %s\n" "$1"
}

fail() {
  printf "FAIL: %s\n" "$1" >&2
  failed=$((failed + 1))
}

check_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    pass "command exists: $1"
  else
    fail "missing command: $1"
  fi
}

check_url() {
  local label="$1"
  local url="$2"
  local output

  if output="$(curl -fsS --max-time "${CURL_TIMEOUT_SECONDS}" "$url" 2>&1)"; then
    pass "$label"
    printf "%s\n" "$output" | head -c 500
    printf "\n"
  else
    fail "$label"
    printf "%s\n" "$output" | head -c 1000 >&2
    printf "\n" >&2
  fi
}

check_docker_health() {
  local container="$1"
  local status

  if ! docker inspect "$container" >/dev/null 2>&1; then
    fail "container not found: $container"
    return
  fi

  status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container" 2>/dev/null || true)"
  if [ "$status" = "healthy" ] || [ "$status" = "running" ]; then
    pass "$container status: $status"
  else
    fail "$container status: ${status:-unknown}"
  fi
}

section "Project"
cd "$PROJECT_DIR" || {
  fail "cannot cd to PROJECT_DIR: $PROJECT_DIR"
  exit 1
}
printf "PROJECT_DIR=%s\n" "$PROJECT_DIR"
printf "PUBLIC_BASE_URL=%s\n" "$PUBLIC_BASE_URL"

section "Required Commands"
check_cmd docker
check_cmd curl

section "Docker Containers"
docker compose -f docker-compose.postgres.yml ps || fail "docker compose postgres ps failed"
docker compose -f docker-compose.ubuntu.yml ps || fail "docker compose ubuntu ps failed"
check_docker_health medication-vqa-postgres
check_docker_health medication-vqa-main
check_docker_health medication-vqa-pdpa-masker

section "HTTP Health"
check_url "main local endpoint" "$MAIN_LOCAL_URL"
check_url "main public domain" "${PUBLIC_BASE_URL}/"
check_url "LIFF camera page" "${PUBLIC_BASE_URL}/liff/camera"
check_url "PDPA masker health" "$PDPA_HEALTH_URL"
check_url "database lookup through main app" "${PUBLIC_BASE_URL}/test-db/${TEST_DRUG}"

section "PostgreSQL"
if docker compose -f docker-compose.postgres.yml exec -T db sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"' >/dev/null 2>&1; then
  pass "PostgreSQL is ready"
else
  fail "PostgreSQL is not ready"
fi

if medication_count="$(docker compose -f docker-compose.postgres.yml exec -T db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "select count(*) from public.\"Medication_VQA\";"' 2>&1)"; then
  pass "Medication_VQA row count"
  printf "Medication_VQA rows: %s\n" "$medication_count"
else
  fail "Medication_VQA row count failed"
  printf "%s\n" "$medication_count" >&2
fi

if embedding_count="$(docker compose -f docker-compose.postgres.yml exec -T db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "select count(*) from public.\"Medication_VQA\" where embedding is not null;"' 2>&1)"; then
  pass "Medication_VQA embedding count"
  printf "Medication_VQA rows with embedding: %s\n" "$embedding_count"
else
  fail "Medication_VQA embedding count failed"
  printf "%s\n" "$embedding_count" >&2
fi

if vector_result="$(docker compose -f docker-compose.postgres.yml exec -T db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "select count(*) from public.match_symptoms((select embedding from public.\"Medication_VQA\" where embedding is not null limit 1), 0.1, 3);"' 2>&1)"; then
  pass "vector search function"
  printf "match_symptoms result count: %s\n" "$vector_result"
else
  fail "vector search function failed"
  printf "%s\n" "$vector_result" >&2
fi

section "Disk"
df -h "$PROJECT_DIR" || fail "df failed"
if [ -d "$PROJECT_DIR/postgres/backups" ]; then
  du -sh "$PROJECT_DIR/postgres/backups" || true
fi
if [ -d "$PROJECT_DIR/test/local_pdpa_debug" ]; then
  du -sh "$PROJECT_DIR/test/local_pdpa_debug" || true
fi
docker system df || true

section "Recent Logs"
printf "Main app errors/warnings:\n"
docker compose -f docker-compose.ubuntu.yml logs --tail=80 main 2>/dev/null | grep -Ei "error|exception|failed|unauthorized|timeout|traceback|warning|ขัดข้อง" || true

printf "\nPDPA masker errors/warnings:\n"
docker compose -f docker-compose.ubuntu.yml logs --tail=80 pdpa-masker 2>/dev/null | grep -Ei "error|exception|failed|unauthorized|timeout|traceback|warning" || true

printf "\nPostgreSQL errors/warnings:\n"
docker compose -f docker-compose.postgres.yml logs --tail=80 db 2>/dev/null | grep -Ei "error|fatal|panic|warning" || true

section "Summary"
if [ "$failed" -eq 0 ]; then
  printf "All health checks passed.\n"
else
  printf "%s health check(s) failed. Check the section above, then see docs/monitoring_logs_health.md.\n" "$failed" >&2
fi

exit "$failed"
