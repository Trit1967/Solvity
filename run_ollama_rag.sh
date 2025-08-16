#!/bin/bash

echo "🦙 Complete Ollama RAG Setup"
echo "============================"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not installed. Please run:"
    echo ""
    echo "   sudo ./install_ollama.sh"
    echo ""
    echo "Or manually:"
    echo "   sudo curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    exit 1
fi

echo "✅ Ollama is installed"

# Check if Ollama service is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "🚀 Starting Ollama service..."
    ollama serve &
    sleep 5
fi

echo "✅ Ollama service is running"

# Check if llama3.2 is available
echo "📥 Checking for LLaMA 3.2 model..."
if ! ollama list | grep -q "llama3.2"; then
    echo "Downloading LLaMA 3.2 (3GB)..."
    ollama pull llama3.2
else
    echo "✅ LLaMA 3.2 is already downloaded"
fi

# Start the RAG interface
echo ""
echo "🎯 Starting RAG Interface..."
echo "============================"
echo ""

# Activate virtual environment if it exists
if [ -d "neo_rag_env" ]; then
    source neo_rag_env/bin/activate
fi

# Install required packages
pip install gradio requests -q

# Run the RAG
python ollama_rag.py