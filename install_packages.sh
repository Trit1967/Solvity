#!/bin/bash

echo "📦 Installing RAG packages step by step..."

# Activate virtual environment
source neo_rag_env/bin/activate

# Update pip and install wheel
echo "1️⃣ Updating pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch CPU version
echo "2️⃣ Installing PyTorch (CPU)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core packages one by one
echo "3️⃣ Installing transformers..."
pip install transformers

echo "4️⃣ Installing Gradio..."
pip install gradio

echo "5️⃣ Installing LangChain..."
pip install langchain langchain-community

echo "6️⃣ Installing embeddings..."
pip install sentence-transformers

echo "7️⃣ Installing FAISS..."
pip install faiss-cpu

echo "8️⃣ Installing document loaders..."
pip install pypdf python-docx unstructured

echo "9️⃣ Installing remaining dependencies..."
pip install numpy accelerate tqdm

echo ""
echo "✅ All packages installed!"
echo ""
echo "To start the UI, run:"
echo "  source neo_rag_env/bin/activate"
echo "  python rag_ui_gradio.py --model 125M"