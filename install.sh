#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV="$ROOT/.venv"
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip
"$VENV/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/requirements.txt"
(cd "$ROOT" && "$VENV/bin/python" stratabyte_7m.py --self-test)
echo "StrataByte-7M installation verified"

# Desktop control-panel dependency. Kept in the project venv.
GUI_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
GUI_VENV="$GUI_ROOT/.venv"
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }
[ -x "$GUI_VENV/bin/python" ] || python3 -m venv "$GUI_VENV"
"$GUI_VENV/bin/python" -m pip install --disable-pip-version-check --upgrade PySide6
touch "$GUI_VENV/.repo-gui-ready"
