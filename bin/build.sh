#!/usr/bin/env bash

set -eu

# Resolve the project root directory.
ROOT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Initialize the python environment and load configuration variables.
source ".venv/bin/activate"
source ".env"

# Build the records database, downloading source files when necessary.
python "build/extract/terms.py"
python "build/extract/annotations.py"
python "build/extract/documents.py"
python "build/frequencies.py"
