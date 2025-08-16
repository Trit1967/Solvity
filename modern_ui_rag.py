#!/usr/bin/env python3
"""
Modern, Professional RAG UI
Inspired by ChatGPT, Claude, and Perplexity interfaces
"""

import gradio as gr
import requests
import json
import time
from pathlib import Path
from typing import List, Tuple
import random

class ModernRAGInterface:
    def __init__(self):
        self.model = "llama3.2"  # or mistral
        self.base_url = "http://localhost:11434"
        self.documents = {}
        self.chat_history = []
        
    def check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def load_documents(self, files):
        """Load documents with progress"""
        if not files:
            return gr.update(value="📎 No files selected"), gr.update(visible=False)
        
        self.documents = {}
        loaded = []
        
        for file in files:
            try:
                with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                filename = Path(file.name).name
                self.documents[filename] = content
                loaded.append(filename)
            except Exception as e:
                print(f"Error: {e}")
        
        if loaded:
            return (
                gr.update(value=f"✅ {len(loaded)} files ready"),
                gr.update(visible=True, value=f"📚 Loaded: {', '.join(loaded)}")
            )
        return gr.update(value="❌ Failed to load files"), gr.update(visible=False)
    
    def stream_response(self, message: str, history: List[Tuple[str, str]]):
        """Stream response with typing effect"""
        if not message.strip():
            return history
        
        # Add user message
        history.append([message, None])
        
        # Prepare context
        context = ""
        if self.documents:
            for filename, content in self.documents.items():
                context += f"\n{content[:2000]}\n"
            prompt = f"Based on these documents:\n{context[:4000]}\n\nQuestion: {message}\n\nAnswer:"
        else:
            prompt = message
        
        # Simulate streaming (replace with actual Ollama streaming)
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False}
            )
            
            if response.status_code == 200:
                full_response = response.json().get('response', 'No response')
                
                # Simulate typing effect
                history[-1][1] = ""
                for i in range(0, len(full_response), 5):
                    history[-1][1] = full_response[:i+5]
                    yield history
                    time.sleep(0.01)
            else:
                history[-1][1] = "❌ Error connecting to AI model"
                yield history
                
        except Exception as e:
            history[-1][1] = f"❌ Error: {str(e)}"
            yield history

def create_modern_ui():
    """Create a modern, professional UI"""
    
    rag = ModernRAGInterface()
    
    # Custom CSS for modern look
    custom_css = """
    /* Modern Dark Theme */
    :root {
        --primary: #10B981;
        --primary-hover: #059669;
        --bg-primary: #0F172A;
        --bg-secondary: #1E293B;
        --bg-tertiary: #334155;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --border: #334155;
        --accent: #6366F1;
    }
    
    /* Global Styles */
    .gradio-container {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        color: var(--text-primary) !important;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Chat Container */
    .chat-container {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 1rem !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
    }
    
    /* Message Bubbles */
    .message {
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 1rem !important;
        padding: 1rem !important;
        margin: 0.5rem !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    
    .bot-message {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border-radius: 1rem !important;
        padding: 1rem !important;
        margin: 0.5rem !important;
        border-left: 3px solid var(--primary) !important;
    }
    
    /* Input Field */
    .input-box textarea {
        background: var(--bg-tertiary) !important;
        border: 2px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 0.75rem !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .input-box textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
        outline: none !important;
    }
    
    /* Buttons */
    .primary-btn {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 0.75rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2) !important;
    }
    
    .primary-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 12px rgba(16, 185, 129, 0.3) !important;
    }
    
    .secondary-btn {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border) !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 0.75rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .secondary-btn:hover {
        background: var(--bg-secondary) !important;
        border-color: var(--primary) !important;
    }
    
    /* File Upload Area */
    .file-upload {
        background: var(--bg-tertiary) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 1rem !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }
    
    .file-upload:hover {
        border-color: var(--primary) !important;
        background: rgba(16, 185, 129, 0.05) !important;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-online {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid #10B981;
    }
    
    .status-offline {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
    }
    
    /* Tabs */
    .tab-nav button {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        position: relative !important;
        transition: all 0.3s ease !important;
    }
    
    .tab-nav button.selected {
        color: var(--primary) !important;
    }
    
    .tab-nav button.selected::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--primary);
        animation: slideIn 0.3s ease-out;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--bg-tertiary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--border);
    }
    
    /* Loading Animation */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 4px;
        padding: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--primary);
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.7;
        }
        30% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }
    
    /* Glassmorphism Effect */
    .glass {
        background: rgba(30, 41, 59, 0.5) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Base(),
        css=custom_css,
        title="AI Document Assistant"
    ) as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">
                ✨ AI Document Assistant
            </h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.1rem;">
                Chat with your documents using advanced AI
            </p>
            <div style="margin-top: 1rem;">
                <span class="status-badge status-online">● AI Online</span>
            </div>
        </div>
        """)
        
        with gr.Row():
            # Main Chat Area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=600,
                    elem_classes="chat-container",
                    show_label=False,
                    avatar_images=["🧑‍💼", "🤖"],
                    bubble_full_width=False,
                    render_markdown=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask me anything about your documents... (Press Enter to send)",
                        show_label=False,
                        elem_classes="input-box",
                        scale=4,
                        lines=1,
                        max_lines=3,
                        autofocus=True
                    )
                    
                    send_btn = gr.Button(
                        "Send →",
                        elem_classes="primary-btn",
                        scale=1
                    )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", elem_classes="secondary-btn")
                    stop_btn = gr.Button("⏹️ Stop", elem_classes="secondary-btn")
                    
                    # Example prompts
                    with gr.Row():
                        gr.Examples(
                            examples=[
                                "Summarize the key points",
                                "What are the main findings?",
                                "Explain this in simple terms",
                                "What questions does this raise?",
                                "Create action items from this"
                            ],
                            inputs=msg,
                            label="Quick Prompts"
                        )
            
            # Right Sidebar
            with gr.Column(scale=1):
                # Document Upload Card
                with gr.Group(elem_classes="glass"):
                    gr.Markdown("### 📚 Documents")
                    
                    file_upload = gr.File(
                        label="Drag & Drop Files",
                        file_count="multiple",
                        file_types=[".txt", ".pdf", ".md", ".docx"],
                        elem_classes="file-upload",
                        height=150
                    )
                    
                    upload_btn = gr.Button(
                        "📤 Upload Documents",
                        elem_classes="primary-btn",
                        size="sm"
                    )
                    
                    upload_status = gr.Textbox(
                        value="📎 No documents loaded",
                        show_label=False,
                        interactive=False,
                        elem_classes="status-text"
                    )
                    
                    file_list = gr.Markdown(
                        visible=False,
                        elem_classes="file-list"
                    )
                
                # Settings Card
                with gr.Group(elem_classes="glass"):
                    gr.Markdown("### ⚙️ Settings")
                    
                    model_select = gr.Dropdown(
                        choices=["llama3.2", "mistral", "phi3", "deepseek-coder"],
                        value="llama3.2",
                        label="AI Model",
                        interactive=True
                    )
                    
                    temperature = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.7,
                        step=0.1,
                        label="Creativity",
                        info="Higher = more creative"
                    )
                    
                    max_length = gr.Slider(
                        minimum=100,
                        maximum=2000,
                        value=500,
                        step=100,
                        label="Response Length",
                        info="Maximum response length"
                    )
                
                # Info Card
                with gr.Group(elem_classes="glass"):
                    gr.Markdown("""
                    ### 💡 Tips
                    
                    - Upload documents first
                    - Ask specific questions
                    - Use examples for inspiration
                    - Try different models
                    
                    ### 🔒 Privacy
                    
                    All processing happens locally.
                    Your data never leaves your computer.
                    """)
        
        # Event handlers
        def process_message(msg, history):
            if not msg.strip():
                return "", history
            
            # Clear input immediately
            yield "", history
            
            # Stream the response
            for updated_history in rag.stream_response(msg, history):
                yield "", updated_history
        
        # Connect events
        msg.submit(
            process_message,
            [msg, chatbot],
            [msg, chatbot]
        )
        
        send_btn.click(
            process_message,
            [msg, chatbot],
            [msg, chatbot]
        )
        
        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, msg]
        )
        
        upload_btn.click(
            rag.load_documents,
            inputs=[file_upload],
            outputs=[upload_status, file_list]
        )
    
    return app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("✨ Modern AI Document Assistant")
    print("="*60)
    print("Features:")
    print("  • Beautiful dark theme")
    print("  • Smooth animations")
    print("  • Typing indicators")
    print("  • Glassmorphism effects")
    print("  • Professional design")
    print("="*60 + "\n")
    
    app = create_modern_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )