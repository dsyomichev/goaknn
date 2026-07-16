# Load python environment and configurations.
source .venv/bin/activate
source .env

# Compile records from source files, downloading them when necessary.
python src/build/records/terms.py
python src/build/records/annotations.py
python src/build/records/documents.py
