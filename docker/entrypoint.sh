#!/bin/bash
# Run this script by sourcing it so the virtual environment activates in your current shell:

# sed -i 's/\r$//' entrypoint.sh
# source entrypoint.sh


# Navigate to the workspace directory
cd /app

# Check if uv is already installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Ensure uv is in the PATH for the remainder of this script
    export PATH="/root/.local/bin:$PATH"
else
    echo "uv is already installed. Skipping installation."
fi

# Create a Python 3.11 virtual environment if one does not exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.11..."
    uv venv --python 3.11
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
source .venv/bin/activate

# Install from requirements.txt
echo "Installing Requirements..."
uv pip install -r docker/requirements.txt


echo "Environment setup complete!"