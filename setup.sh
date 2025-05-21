#!/bin/bash

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
uv pip install --upgrade pip

# Check for rdoc_package
if [ ! -d "rdoc_package" ]; then
    echo "Warning: rdoc_package directory not found in workspace."
    echo "Please ensure rdoc_package is available in one of these ways:"
    echo "1. Clone the rdoc_package repository into this workspace"
    echo "2. Install it from PyPI if available"
    echo "3. Provide the correct path to the package"
    exit 1
fi

# Install requirements
echo "Installing requirements..."
uv pip install -r requirements.txt

# Install rdoc-package in development mode
echo "Installing rdoc-package in development mode..."
uv pip install -e .

echo "Setup complete! Your environment is ready to use."
echo "To activate the environment in the future, run: source .venv/bin/activate"

# Run the preprocessing and analysis scripts
echo "Running preprocessing script..."
python preprocess.py

echo "Running analysis script..."
python analyze_behavioral_data.py

# Verify the activation
ls .venv/bin/activate

chmod +x setup.sh 