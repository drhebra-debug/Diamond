#!/bin/bash
# ================================================
# Diamond v4.1 - FULL INSTALLATION SCRIPT
# Runs inside your project folder
# ================================================

set -e  # Exit immediately if any command fails

echo "🚀 Starting Diamond v4.1 Complete Installation..."

# 1. CONDA ENVIRONMENT
echo "🔧 Checking Conda environment 'llmprod'..."
if ! conda env list | grep -q "llmprod"; then
    echo "Creating new conda environment: llmprod"
    conda create -n llmprod python=3.11 -y
fi

echo "Activating conda environment llmprod..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmprod

# 2. PROJECT VIRTUAL ENVIRONMENT (inside folder)
VENV_DIR="./venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating isolated Python virtual environment in $VENV_DIR..."
    python -m venv "$VENV_DIR"
else
    echo "Using existing virtual environment in $VENV_DIR"
fi

echo "Activating project virtual environment..."
source "$VENV_DIR/bin/activate"

# 3. UPGRADE PIP & TOOLS
echo "Upgrading pip, setuptools and wheel..."
pip install --upgrade pip setuptools wheel

# 4. GPU-ACCELERATED LLAMA-CPP-PYTHON (MOST IMPORTANT)
echo "Building llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DLLAMA_CUDA=on" \
pip install llama-cpp-python --force-reinstall --no-cache-dir -vvv

# 5. ALL OTHER DEPENDENCIES
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# 6. FINAL CHECKS
echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "📁 Project structure:"
echo "   ├── docker-compose.yml"
echo "   ├── .env"
echo "   ├── setup_db.py"
echo "   ├── diamond.py"
echo "   ├── requirements.txt"
echo "   ├── install.sh"
echo "   └── venv/          ← isolated environment"
echo ""
echo "🚀 Next steps:"
echo "1. Start databases:      docker compose up -d"
echo "2. Setup database:       python setup_db.py"
echo "3. Run Diamond:          python diamond.py"
echo ""
echo "Test command:"
echo "   export ANTHROPIC_BASE_URL=http://localhost:11434"
echo "   claude \"completely survey all files and folders...\""
echo ""
echo "Diamond v4.1 is now fully installed and ready! 🔥"
