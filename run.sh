#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
else
  echo "[Warn] .env 不存在，将使用程序默认值。建议先执行: cp .env.example .env"
fi

mkdir -p "$ROOT_DIR/build"
cd "$ROOT_DIR/build"

cmake ..
make -j"$(nproc)"

exec ./server
