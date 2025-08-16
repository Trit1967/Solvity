#!/bin/bash
# Special setup for WSL2 environments

echo "🚀 WSL2 RAGbot Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Ensure Python and venv are installed
echo "📦 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Installing Python..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv python3-dev
fi

# 2. Clean up any existing virtual environment
if [ -d "venv" ]; then
    echo "🧹 Removing old virtual environment..."
    rm -rf venv
fi

# 3. Create fresh virtual environment
echo "📦 Creating fresh virtual environment..."
python3 -m venv venv

# 4. Activate and upgrade pip
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# 5. Install minimal dependencies first
echo "📦 Installing core dependencies..."

# Install one by one to avoid conflicts
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install python-multipart==0.0.6
pip install python-dotenv==1.0.0

# Basic auth
pip install python-jose==3.3.0
pip install passlib==1.7.4
pip install bcrypt==4.1.1

# Rate limiting
pip install slowapi==0.1.9

# Security
pip install cryptography==41.0.7

# Document processing (minimal)
pip install PyPDF2==3.0.1

# ChromaDB for your existing code
echo "📦 Installing ChromaDB..."
pip install chromadb==0.4.22

echo "📦 Installing basic sentence transformers..."
# Install sentence transformers without heavy torch
pip install --no-deps sentence-transformers
pip install transformers
pip install numpy

# 6. Create necessary directories
echo "📁 Creating directories..."
mkdir -p data tenant_data uploads cache logs secure_data
chmod 700 secure_data

# 7. Create .env file
if [ ! -f .env ]; then
    echo "🔑 Creating .env file..."
    cat > .env << 'EOF'
SECRET_KEY=your-secret-key-change-in-production-123456789012345678901234567890
MASTER_KEY=your-master-key-change-in-production-123456789012345678901234567890
OLLAMA_HOST=http://localhost:11434
ENVIRONMENT=development
EOF
fi

# 8. Check Ollama
echo "🔍 Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
    if pgrep -x "ollama" > /dev/null; then
        echo "✅ Ollama is running"
    else
        echo "⚠️  Ollama not running. Start with: ollama serve"
    fi
else
    echo "⚠️  Ollama not installed"
    echo "   Install with: curl -fsSL https://ollama.com/install.sh | sh"
fi

echo "
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ WSL2 Setup Complete!

To start RAGbot:

1. Activate virtual environment:
   source venv/bin/activate

2. If Ollama not installed:
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve &
   ollama pull llama3.2

3. Run the test:
   python test_minimal.py

4. Start the API:
   python ragbot_secure_app.py

Alternative: Install Docker Desktop on Windows
for easier deployment!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"