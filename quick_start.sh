#!/bin/bash

echo "🚀 Quick RAG Setup for WSL"
echo "========================="

# Check if we can use venv
if ! python3 -m venv --help > /dev/null 2>&1; then
    echo "❌ Python venv not available"
    echo ""
    echo "Please run this first:"
    echo "  sudo apt update"
    echo "  sudo apt install python3-venv python3-pip"
    echo ""
    echo "Then run this script again"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv neo_rag_env

# Activate it
source neo_rag_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip --quiet

# Install minimal requirements for UI
echo "📚 Installing core packages..."
pip install --quiet \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    gradio \
    langchain \
    langchain-community \
    sentence-transformers \
    faiss-cpu \
    pypdf

echo ""
echo "✅ Setup complete!"
echo ""
echo "Now run:"
echo "  source neo_rag_env/bin/activate"
echo "  python rag_ui_gradio.py --model 125M"
echo ""
echo "This will:"
echo "  1. Start with smallest model (125M)"
echo "  2. Open web UI at http://localhost:7860"
echo "  3. Let you upload documents and chat"