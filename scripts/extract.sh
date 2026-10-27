#!/usr/bin/env bash
set -e
if [ "$#" -ne 1 ]; then
  echo "사용법: bash scripts/extract.sh configs/config.yaml" >&2
  exit 2
fi
root="$(cd "$(dirname "$0")/.." && pwd)"
exec /usr/bin/python3 "$root/src/bootstrap.py" extract "$1"
