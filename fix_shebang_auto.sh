#!/bin/bash

# Automatic shebang fix script for uv/conda virtual environments
# Run this after 'colcon build'

echo "🔧 Fixing shebang lines for virtual environment..."

# Get current Python executable path
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    # If using conda
    PYTHON_PATH=$(which python3)
    ENV_TYPE="conda environment: $CONDA_DEFAULT_ENV"
elif [[ -n "$VIRTUAL_ENV" ]]; then
    # If using standard venv
    PYTHON_PATH="$VIRTUAL_ENV/bin/python3"
    ENV_TYPE="virtual environment: $VIRTUAL_ENV"
elif [[ -f ".venv/bin/python3" ]]; then
    # If using uv venv (local .venv)
    PYTHON_PATH="$(pwd)/.venv/bin/python3"
    ENV_TYPE="uv environment: $(pwd)/.venv"
else
    echo "❌ No virtual environment detected!"
    echo "Please activate your virtual environment first."
    exit 1
fi

echo "📍 Detected $ENV_TYPE"
echo "📍 Python path: $PYTHON_PATH"

# Find and fix shebang in digit_recognizer
DIGIT_RECOGNIZER_FILE="install/fsr_array_publisher/lib/fsr_array_publisher/digit_recognizer"

if [[ -f "$DIGIT_RECOGNIZER_FILE" ]]; then
    echo "🔍 Found: $DIGIT_RECOGNIZER_FILE"
    
    # Backup original
    cp "$DIGIT_RECOGNIZER_FILE" "${DIGIT_RECOGNIZER_FILE}.backup"
    
    # Get current shebang
    CURRENT_SHEBANG=$(head -n 1 "$DIGIT_RECOGNIZER_FILE")
    echo "📝 Current shebang: $CURRENT_SHEBANG"
    
    # Replace shebang
    sed -i "1s|.*|#!$PYTHON_PATH|" "$DIGIT_RECOGNIZER_FILE"
    
    # Verify change
    NEW_SHEBANG=$(head -n 1 "$DIGIT_RECOGNIZER_FILE")
    echo "✅ New shebang: $NEW_SHEBANG"
    
    echo "🎉 Shebang fixed successfully!"
else
    echo "❌ File not found: $DIGIT_RECOGNIZER_FILE"
    echo "💡 Make sure you have run 'colcon build --packages-select fsr_array_publisher' first"
fi
