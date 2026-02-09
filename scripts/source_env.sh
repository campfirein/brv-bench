#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

VENV_NAME=".venv"
PYTHON_VERSION="3.12"
REQUIREMENT_FILE="requirements.txt"

# =============================================================================
# Functions
# =============================================================================

function log() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

function error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1" >&2
    return 1
}

function ensure_uv_installed() {
    if ! command -v uv &> /dev/null; then
        log "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh || {
            error "Failed to install uv."
            return 1
        }
        export PATH="$HOME/.cargo/bin:$PATH"  # In case it's installed via cargo
    fi
}

function create_venv() {
    log "Creating virtual environment: ${VENV_NAME}"
    uv venv "${VENV_NAME}" --python "python${PYTHON_VERSION}" || {
        error "uv venv failed."
        return 1
    }
    log "Virtual environment created."
}

function activate_venv() {
    # shellcheck source=/dev/null
    source "${VENV_NAME}/bin/activate"
    log "Virtual environment activated."
}

function install_requirements() {
    if [ -f "${REQUIREMENT_FILE}" ]; then
        uv pip install -r "${REQUIREMENT_FILE}"
        uv pip install -e .
        log "Requirements installed from ${REQUIREMENT_FILE}"
    else
        error "Requirements file '${REQUIREMENT_FILE}' not found."
        return 1
    fi
}

# =============================================================================
# Main
# =============================================================================

ensure_uv_installed

if [ ! -d "${VENV_NAME}" ]; then
    create_venv
    activate_venv
    install_requirements
else
    activate_venv
fi
