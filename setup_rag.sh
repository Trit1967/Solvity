#!/bin/bash

# Setup script for RAG Chatbot
echo "🚀 Setting up RAG Chatbot..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv rag_env
source rag_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python packages..."
pip install -r requirements.txt

# Install Ollama (for local LLM)
echo "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "Ollama already installed"
fi

# Pull a model (choose one)
echo "Pulling LLM model..."
echo "Choose a model:"
echo "1) llama3.2 (3B parameters, 2GB)"
echo "2) phi3 (3.8B parameters, 2.3GB)"
echo "3) mistral (7B parameters, 4.1GB)"
echo "4) llama2 (7B parameters, 3.8GB)"
read -p "Enter choice (1-4): " choice

case $choice in
    1) ollama pull llama3.2:3b ;;
    2) ollama pull phi3 ;;
    3) ollama pull mistral ;;
    4) ollama pull llama2 ;;
    *) ollama pull llama3.2:3b ;;
esac

# Create directories
mkdir -p documents chroma_db

echo "✅ Setup complete!"
echo ""
echo "To use the chatbot:"
echo "1. Activate environment: source rag_env/bin/activate"
echo "2. Add documents to ./documents/"
echo "3. Run: python3 rag_chatbot.py"