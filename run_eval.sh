#!/bin/bash

# Add environment variables - HF_TOKEN must be set
export HF_TOKEN="YOUR_HF_TOKEN"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run evalulation
echo "Running evaluation..."
python -m src.eval.eval --log-level DEBUG

# Run
echo "Running translation..."
python -m src.cap_translator.translate