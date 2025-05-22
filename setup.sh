#!/bin/bash

# Exit on error
set -e

echo "Setting up Python environment with uv..."

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv

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

# Install project in development mode
echo "Installing project in development mode..."
uv pip install -e .

echo "Setup complete! Your environment is ready to use."
echo "To activate the environment in the future, run: source .venv/bin/activate"
echo ""
echo "To run preprocessing for a subject, use:"
echo "python preprocess.py <subject_folder>"
echo "Example: python preprocess.py sub-sK"
echo ""
echo "To run analysis for a subject, use:"
echo "python analyze_behavioral_data.py <subject_folder>"
echo "Example: python analyze_behavioral_data.py sub-sK"

chmod +x setup.sh 