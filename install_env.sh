#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sam3model}"
ENV_FILE="${ENV_FILE:-sam3model.yml}"
PIP_REQUIREMENTS="${PIP_REQUIREMENTS:-requirements.txt}"
INSTALL_PIP_FREEZE=0

usage() {
  cat <<'EOF'
Usage: ./install_env.sh [options]

Create or update the conda environment for this SAM3 tracker repository.

Options:
  --name NAME          Conda environment name. Default: sam3model
  --file FILE          Conda environment YAML. Default: sam3model.yml
  --requirements FILE  Pip requirements file. Default: requirements.txt
  --pip                Also install the exact pip freeze after conda update
  -h, --help           Show this help

Environment variables:
  ENV_NAME             Same as --name
  ENV_FILE             Same as --file
  PIP_REQUIREMENTS     Same as --requirements
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      ENV_NAME="$2"
      shift 2
      ;;
    --file)
      ENV_FILE="$2"
      shift 2
      ;;
    --requirements)
      PIP_REQUIREMENTS="$2"
      shift 2
      ;;
    --pip)
      INSTALL_PIP_FREEZE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Install Miniconda or Anaconda first." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file not found: $ENV_FILE" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating conda environment: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "Creating conda environment: $ENV_NAME"
  conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

if [[ "$INSTALL_PIP_FREEZE" -eq 1 ]]; then
  if [[ ! -f "$PIP_REQUIREMENTS" ]]; then
    echo "Requirements file not found: $PIP_REQUIREMENTS" >&2
    exit 1
  fi
  echo "Installing exact pip freeze from: $PIP_REQUIREMENTS"
  conda run -n "$ENV_NAME" python -m pip install -r "$PIP_REQUIREMENTS"
fi

echo
echo "Environment ready."
echo "Activate it with:"
echo "  conda activate $ENV_NAME"
