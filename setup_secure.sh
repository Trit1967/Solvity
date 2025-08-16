#!/bin/bash
# Quick setup script for secure RAGbot with virtual environment

set -e

echo "🔐 Setting up Secure RAGbot..."

# 1. Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

# 2. Create .env if doesn't exist
if [ ! -f .env ]; then
    cp .env.ragbot .env
    
    # Generate secure keys
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    MASTER_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    
    # Update .env with secure keys
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/change-this-to-a-secure-random-string-minimum-32-characters/$SECRET_KEY/" .env
        echo "MASTER_KEY=$MASTER_KEY" >> .env
    else
        # Linux
        sed -i "s/change-this-to-a-secure-random-string-minimum-32-characters/$SECRET_KEY/" .env
        echo "MASTER_KEY=$MASTER_KEY" >> .env
    fi
    
    echo "✅ Generated secure keys"
fi

# 3. Create necessary directories with proper permissions
mkdir -p data tenant_data uploads cache logs secure_data
chmod 700 secure_data  # Restricted access

# 4. Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip

# 5. Install dependencies in virtual environment
echo "📦 Installing dependencies..."
pip install -r requirements_ragbot.txt
pip install cryptography  # For encryption

# 6. Initialize databases
echo "🗄️ Initializing databases..."
python3 -c "
try:
    from security_enhanced_rag import SecureMultiTenantRAG
    secure_rag = SecureMultiTenantRAG()
    print('✅ Security database initialized')
except ImportError as e:
    print('⚠️  Some dependencies may be missing. Installing core components...')
    import subprocess
    subprocess.run(['pip', 'install', 'pydantic', 'fastapi', 'cryptography'])
"

# 7. Test the setup
echo "🧪 Testing setup..."
python3 -c "
import os
from pathlib import Path

# Check critical components
checks = {
    'Environment': os.path.exists('.env'),
    'Virtual Environment': os.path.exists('venv'),
    'Data directories': Path('data').exists() and Path('secure_data').exists(),
}

# Check if Ollama is accessible
import subprocess
try:
    result = subprocess.run(['curl', '-s', 'http://localhost:11434/api/tags'], 
                          capture_output=True, timeout=2)
    checks['Ollama'] = result.returncode == 0
except:
    checks['Ollama'] = False

print('\n📋 System Check:')
for component, status in checks.items():
    symbol = '✅' if status else '❌'
    print(f'{symbol} {component}')

if not checks['Ollama']:
    print('\n⚠️  Ollama not running. Start with:')
    print('   curl -fsSL https://ollama.com/install.sh | sh')
    print('   ollama serve &')
    print('   ollama pull llama3.2')
"

echo "
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Setup complete! 

🚀 To start RAGbot:

1. Activate virtual environment:
   source venv/bin/activate

2. Start Ollama (if not running):
   ollama serve &
   ollama pull llama3.2

3. Run the secure API:
   python ragbot_secure_app.py

4. Or deploy with Docker:
   ./deploy_ragbot.sh docker

📝 Note: Always activate the virtual environment before running:
   source venv/bin/activate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"