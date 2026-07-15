#!/usr/bin/env bash
# Create project venv with a working Python (avoid broken Homebrew 3.14 pyexpat).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pick_python() {
  local candidates=(
    "/usr/bin/python3"
    "/opt/homebrew/bin/python3.12"
    "/opt/homebrew/bin/python3.11"
    "/opt/homebrew/bin/python3.10"
    "/opt/homebrew/bin/python3.9"
    "python3"
  )
  local c
  for c in "${candidates[@]}"; do
    if ! command -v "$c" >/dev/null 2>&1 && [[ ! -x "$c" ]]; then
      continue
    fi
    # Skip known-broken Homebrew 3.14 on some macOS versions (pyexpat/libexpat mismatch).
    if "$c" -c "import sys, xml.parsers.expat; assert sys.version_info[:2] >= (3, 9)" 2>/dev/null; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(pick_python)" || {
  echo "No working Python 3.9+ found."
  echo "Homebrew python3.14 often fails with:"
  echo "  ImportError: Symbol not found: _XML_SetAllocTrackerActivationThreshold"
  echo "Fix: use Apple/Xcode Python:"
  echo "  /usr/bin/python3 -m venv .venv"
  echo "Or reinstall a stable Homebrew Python (3.11/3.12) and re-run this script."
  exit 1
}

echo "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])'))"

# Recreate venv if missing or if it points at a broken interpreter.
NEED_NEW=0
if [ ! -x .venv/bin/python3 ]; then
  NEED_NEW=1
elif ! .venv/bin/python3 -c "import xml.parsers.expat" 2>/dev/null; then
  echo "Existing .venv is broken (expat). Recreating..."
  NEED_NEW=1
fi

if [ "$NEED_NEW" -eq 1 ]; then
  rm -rf .venv
  "$PYTHON_BIN" -m venv .venv
fi

PY="$ROOT/.venv/bin/python3"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt

echo
echo "Setup OK."
echo "  source .venv/bin/activate"
echo "  # always prefer the venv binary:"
echo "  .venv/bin/python3 data-generator/generate_state_budget_gst.py"
