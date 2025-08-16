#!/bin/bash

echo "🚀 RAG Chatbot UI Launcher"
echo "========================"

# Check if virtual environment exists
if [ ! -d "neo_rag_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run ./setup_gpt_neo.sh first"
    exit 1
fi

# Activate virtual environment
source neo_rag_env/bin/activate

# Check for UI libraries
echo "📦 Checking UI libraries..."
pip install -q gradio streamlit

echo ""
echo "Choose UI Framework:"
echo "1) Gradio (Recommended - cleaner, tabs)"
echo "2) Streamlit (More features, analytics)"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo "🎨 Starting Gradio UI..."
        echo "Opening in browser: http://localhost:7860"
        python rag_ui_gradio.py --model 1.3B
        ;;
    2)
        echo "🎨 Starting Streamlit UI..."
        echo "Opening in browser: http://localhost:8501"
        streamlit run rag_ui_streamlit.py
        ;;
    *)
        echo "Invalid choice. Starting Gradio..."
        python rag_ui_gradio.py --model 1.3B
        ;;
esac