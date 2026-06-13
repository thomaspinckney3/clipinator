#!/bin/bash
# Double-click this file to open the Clipinator Review Editor.
#
# It needs only Python 3.11+ (no pip packages, no Homebrew). ffmpeg is optional
# and used only when you click "Render this clip". The editor runs locally in
# your browser; close this window to stop it.

cd "$(dirname "$0")" || exit 1

# Find a usable Python 3.11+ (prefer the newest).
PYTHON=""
for cand in python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      PYTHON="$cand"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Could not find Python 3.11 or newer on this Mac."
  echo "Install it from https://www.python.org/downloads/macos/ and try again."
  echo
  read -r -p "Press return to close."
  exit 1
fi

echo "Starting the Clipinator Review Editor with $PYTHON ..."
echo "Your browser will open automatically. Keep this window open while editing."
echo "Tip: drag a project folder onto this icon to open that project directly;"
echo "otherwise it reopens the last folder you used."
echo
# A folder dragged onto this .command arrives as "$1"; forward it (and any flags).
PYTHONPATH="src" "$PYTHON" -m extractor.review_server "$@"
