#!/bin/bash
# Test RAGbot anywhere - minimal dependencies

echo "🧪 RAGbot Universal Test Script"
echo "================================"

# Check Python
if command -v python3 &> /dev/null; then
    echo "✅ Python3 found"
else
    echo "❌ Python3 not found"
    exit 1
fi

# Install minimal deps to user directory
echo "Installing minimal dependencies..."
pip install --user fastapi uvicorn

# Create test data
mkdir -p test_data

# Run the free test version
echo "
🚀 Starting test server...

Visit these URLs:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

Press Ctrl+C to stop
"

python3 free_local_test.py