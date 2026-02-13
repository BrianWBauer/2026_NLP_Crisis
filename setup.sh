#!/bin/bash
# ============================================
# Project Setup (Mac/Linux)
# 2026_NLP_Crisis
# ============================================
# Run this once on a new machine to create the
# conda environment for this project.
#
# Usage: bash setup.sh
# ============================================

ENV_NAME="nlp_ema"

# Check if environment already exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
    echo "To rebuild: conda env remove -n $ENV_NAME, then rerun this script."
    exit 0
fi

conda env create -f environment.yml

echo ""
echo "========================================="
echo "Setup complete."
echo "Run: conda activate $ENV_NAME"
echo "========================================="
