#!/bin/bash
# Quick setup with minimal dependencies

echo "🚀 Quick RAGbot Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate it
source venv/bin/activate

# 3. Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 4. Install core dependencies one by one
echo "📦 Installing core dependencies..."

# Install in order of importance
echo "Installing FastAPI..."
pip install fastapi uvicorn python-multipart

echo "Installing authentication..."
pip install python-jose passlib bcrypt python-dotenv

echo "Installing security..."
pip install cryptography

echo "Installing document processing..."
pip install PyPDF2 python-docx pandas openpyxl

echo "Installing ChromaDB for vector storage..."
pip install chromadb

echo "Installing rate limiting..."
pip install slowapi

# 5. Create directories
echo "📁 Creating directories..."
mkdir -p data tenant_data uploads cache logs secure_data

# 6. Create basic .env if needed
if [ ! -f .env ]; then
    echo "🔑 Creating .env file..."
    cat > .env << 'EOF'
# RAGbot Configuration
SECRET_KEY=change-this-to-a-secure-key-123456789012345678901234567890
MASTER_KEY=change-this-master-key-123456789012345678901234567890
OLLAMA_HOST=http://localhost:11434
ENVIRONMENT=development
EOF
    echo "✅ Created .env (remember to update the keys!)"
fi

echo "
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Basic setup complete!

Next steps:
1. Activate virtual environment:
   source venv/bin/activate

2. Install Ollama (if not installed):
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve &
   ollama pull llama3.2

3. Test with a simple version:
   python test_minimal.py

Or use Docker to avoid all dependency issues:
   ./deploy_ragbot.sh docker
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"