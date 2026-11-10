#!/bin/bash

# Download PP-Matting model weights (PyTorch + ONNX) from GitHub releases
# Usage: ./download.sh

BASE_URL="https://github.com/yakhyo/ppmatting/releases/download/weights"

WEIGHTS_DIR="weights"
mkdir -p ${WEIGHTS_DIR}

FILES=(
    "ppmatting_v2.pt"
    "ppmatting_v2.onnx"
    "ppmatting_v1.pt"
    "ppmatting_v1.onnx"
)

echo "Downloading model weights..."
for f in "${FILES[@]}"; do
    echo "  -> ${f}"
    curl -L --fail "${BASE_URL}/${f}" -o "${WEIGHTS_DIR}/${f}"
done

echo "Done! Weights saved to ${WEIGHTS_DIR}/"
ls -lh ${WEIGHTS_DIR}/
