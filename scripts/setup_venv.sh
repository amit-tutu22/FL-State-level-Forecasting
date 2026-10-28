#!/usr/bin/env bash
# Create .venv on Python 3.11+ and install the real-data pipeline dependencies.
# Pass --fl to also install the Flower/PyTorch experiment dependencies.
# Some Homebrew Python builds ship a broken pyexpat that crashes pip, so every
# fallback interpreter is checked before use.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
REQUIREMENTS=(-r "$ROOT/requirements-data.txt")
if [ "${1:-}" = "--fl" ]; then
  REQUIREMENTS+=(-r "$ROOT/requirements-fl.txt")
elif [ -n "${1:-}" ]; then
  echo "Usage: $0 [--fl]" >&2
  exit 2
fi

works() {
  "$1" -c 'import sys, xml.parsers.expat; sys.exit(sys.version_info < (3, 11))' \
    >/dev/null 2>&1
}

if [ -d "$VENV" ] && ! works "$VENV/bin/python"; then
  echo "Removing unusable $VENV"
  rm -rf "$VENV"
fi

if [ ! -d "$VENV" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv -p 3.11 "$VENV"
  else
    PYTHON=""
    for candidate in python3.11 python3.12 python3.13; do
      if command -v "$candidate" >/dev/null 2>&1 && works "$candidate"; then
        PYTHON="$candidate"
        break
      fi
    done
    if [ -z "$PYTHON" ]; then
      echo "No working Python 3.11+ found. Install uv (brew install uv) and rerun." >&2
      exit 1
    fi
    "$PYTHON" -m venv "$VENV"
  fi
fi

if ! works "$VENV/bin/python"; then
  echo "$VENV/bin/python is not a working Python 3.11+." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install -p "$VENV/bin/python" "${REQUIREMENTS[@]}"
else
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install "${REQUIREMENTS[@]}"
fi

echo
echo "Environment ready: $("$VENV/bin/python" --version)"
echo "Activate:   . .venv/bin/activate"
echo "Run tests:  python -m unittest discover -s tests -v"
