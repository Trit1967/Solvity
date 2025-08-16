#!/bin/bash

echo "🚀 GPT-Neo RAG Setup"
echo "==================="

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv neo_rag_env
source neo_rag_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Detect system
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🐧 Linux/Mac detected"
    HAS_GPU=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
else
    echo "🪟 Windows detected"
    HAS_GPU="False"
fi

# Install PyTorch
echo "🔥 Installing PyTorch..."
if [ "$HAS_GPU" = "True" ]; then
    echo "  GPU detected! Installing CUDA version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "  Installing CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "📚 Installing dependencies..."
pip install -r requirements_neo.txt

# Create directories
mkdir -p documents faiss_index

echo ""
echo "✅ Setup Complete!"
echo ""
echo "📊 Model Size Recommendations:"
echo "  • 125M  - Fast, needs 2GB RAM (good for testing)"
echo "  • 1.3B  - Balanced, needs 6GB RAM (recommended)"
echo "  • 2.7B  - Better quality, needs 12GB RAM"
echo "  • 6B    - Best quality, needs 16GB+ RAM (GPU recommended)"
echo ""
echo "🎯 Quick Start:"
echo "  1. Activate: source neo_rag_env/bin/activate"
echo "  2. Add documents to ./documents/"
echo "  3. Run: python rag_gpt_neo.py --model 1.3B"
echo ""
echo "📝 First Run Commands:"
echo "  python rag_gpt_neo.py --model 125M --ingest ./documents"
echo ""