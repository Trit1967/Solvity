#!/bin/bash

echo "🦙 Ollama + LLaMA 3.2 Setup"
echo "============================"
echo ""
echo "Step 1: Install Ollama (needs sudo password)"
echo "---------------------------------------------"
curl -fsSL https://ollama.ai/install.sh | sh

echo ""
echo "Step 2: Pull LLaMA 3.2 model (3GB download)"
echo "--------------------------------------------"
ollama pull llama3.2

echo ""
echo "Step 3: Start Ollama service"
echo "-----------------------------"
ollama serve &

echo ""
echo "✅ Setup complete!"
echo ""
echo "Test it: ollama run llama3.2"
echo ""