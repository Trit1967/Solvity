#!/bin/bash
# Easy run script for RAGbot with virtual environment

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting RAGbot Secure API${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Running setup...${NC}"
    ./setup_secure.sh
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Ollama is not running${NC}"
    echo "Would you like to start Ollama? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        echo "Starting Ollama..."
        ollama serve &
        sleep 5
        echo "Pulling llama3.2 model..."
        ollama pull llama3.2
    fi
else
    echo -e "${GREEN}✅ Ollama is running${NC}"
fi

# Start the application
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Starting RAGbot Secure API...${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "🌐 API URL: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo "🔐 Security: Encryption ✅ Audit Logging ✅ Rate Limiting ✅"
echo ""
echo "Press Ctrl+C to stop"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Run the application
python ragbot_secure_app.py