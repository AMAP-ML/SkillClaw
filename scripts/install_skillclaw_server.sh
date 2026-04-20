#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${SKILLCLAW_SERVER_VENV_DIR:-$ROOT_DIR/.venv-server}"
RUN_HELP=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--venv-dir PATH] [--python BIN] [--run-help]

Installs the Python-side dependencies for SkillClaw's evolve servers from the
current repository checkout.

- `skillclaw-evolve-server` works after this install.
- To use the agent engine, also install the external OpenClaw binary:
    npm install -g openclaw

Default install command:
  python -m pip install -e ".[server]"

After install you can run:
  skillclaw-evolve-server --help
  skillclaw-evolve-server --port 8787 --interval 300
  skillclaw-evolve-server --engine agent --help   # requires openclaw installed separately
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --run-help)
      RUN_HELP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "[install_skillclaw_server] repo root: $ROOT_DIR"
echo "[install_skillclaw_server] python: $PYTHON_BIN"
echo "[install_skillclaw_server] venv: $VENV_DIR"

cd "$ROOT_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install -U pip
python -m pip install -e ".[server]"

echo
echo "[install_skillclaw_server] install complete"
echo "[install_skillclaw_server] activate with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "[install_skillclaw_server] next steps:"
echo "  skillclaw-evolve-server --help"
echo "  skillclaw-evolve-server --port 8787 --interval 300"
echo "  skillclaw-evolve-server --engine agent --help    # requires openclaw installed separately"

if [[ "$RUN_HELP" -eq 1 ]]; then
  echo
  echo "[install_skillclaw_server] running: skillclaw-evolve-server --help"
  skillclaw-evolve-server --help
  if command -v openclaw >/dev/null 2>&1; then
    echo
    echo "[install_skillclaw_server] running: skillclaw-evolve-server --engine agent --help"
    skillclaw-evolve-server --engine agent --help
  else
    echo
    echo "[install_skillclaw_server] skipping agent help: openclaw is not installed"
  fi
fi
