#!/usr/bin/env bash
# Verify the standalone lab and write evidence outside the source checkout.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 verify.py "$@"
