#!/usr/bin/env python3
"""
Ollama RAG - Clean, Simple, Powerful
Using LLaMA 3.2 for local AI
"""

import gradio as gr
import requests
import json
from pathlib import Path
from typing import List, Tuple
import subprocess
import time

class OllamaRAG:
    def __init__(self, model="deepseek-coder:6.7b"):  # Or deepseek-llm:7b
        """Initialize Ollama RAG"""
        self.model = model
        self.base_url = "http://localhost:11434"
        self.documents = {}
        
        # Check if Ollama is running
        if not self.check_ollama():
            print("❌ Ollama not running!")
            print("\nPlease run these commands:")
            print("1. sudo curl -fsSL https://ollama.ai/install.sh | sh")
            print("2. ollama serve")
            print("3. ollama pull llama3.2")
        else:
            print(f"✅ Ollama connected! Using {model}")
    
    def check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def pull_model(self) -> str:
        """Pull the model if not available"""
        try:
            # Check if model exists
            response = requests.get(f"{self.base_url}/api/tags")
            models = response.json()
            
            model_names = [m['name'] for m in models.get('models', [])]
            if self.model in model_names:
                return f"✅ Model {self.model} already available"
            
            # Pull model
            print(f"📥 Pulling {self.model}... This may take a few minutes...")
            subprocess.run(["ollama", "pull", self.model], check=True)
            return f"✅ Model {self.model} downloaded successfully!"
            
        except Exception as e:
            return f"❌ Error: {e}"
    
    def load_documents(self, files):
        """Load documents"""
        if not files:
            return "No files uploaded", []
        
        self.documents = {}
        loaded_info = []
        
        for file in files:
            try:
                with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                filename = Path(file.name).name
                self.documents[filename] = content
                loaded_info.append([filename, f"{len(content):,} chars"])
                
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        
        if loaded_info:
            return f"✅ Loaded {len(loaded_info)} documents", loaded_info
        return "❌ No files loaded", []
    
    def chat(self, message: str, history: List[Tuple[str, str]]):
        """Chat with Ollama"""
        
        if not self.check_ollama():
            return history + [(message, "❌ Ollama not running. Please start it first.")]
        
        if not message.strip():
            return history
        
        # Prepare context from documents
        context = ""
        if self.documents:
            # Use all documents as context
            for filename, content in self.documents.items():
                context += f"\n=== {filename} ===\n{content[:2000]}\n"
            
            prompt = f"""Based on these documents:

{context[:4000]}

Question: {message}

Answer:"""
        else:
            prompt = message
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', 'No response')
                return history + [(message, answer)]
            else:
                return history + [(message, f"❌ Error: {response.status_code}")]
                
        except Exception as e:
            return history + [(message, f"❌ Error: {str(e)}")]

def create_ui():
    """Create clean UI"""
    
    rag = OllamaRAG()
    
    with gr.Blocks(title="Ollama RAG", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🦙 Ollama RAG with LLaMA 3.2
        ### Powerful local AI - No API keys needed!
        """)
        
        # Status
        with gr.Row():
            status = gr.Textbox(
                value="✅ Ready" if rag.check_ollama() else "❌ Ollama not running",
                label="Status",
                interactive=False
            )
            model_btn = gr.Button("📥 Download Model")
            model_status = gr.Textbox(label="Model Status", interactive=False)
        
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(height=450)
                
                msg = gr.Textbox(
                    placeholder="Ask anything...",
                    label="Message",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                gr.Examples(
                    examples=[
                        "Summarize this document",
                        "What are the key points?",
                        "Explain the main ideas",
                        "What questions does this raise?"
                    ],
                    inputs=msg
                )
            
            # Documents Tab
            with gr.Tab("📁 Documents"):
                file_upload = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[".txt", ".md", ".pdf", ".docx"]
                )
                
                upload_btn = gr.Button("📥 Load Documents", variant="primary")
                
                upload_status = gr.Textbox(label="Upload Status")
                file_list = gr.Dataframe(
                    headers=["File", "Size"],
                    label="Loaded Documents"
                )
            
            # Setup Tab
            with gr.Tab("🛠️ Setup"):
                gr.Markdown("""
                ## Quick Setup Guide
                
                ### 1. Install Ollama (one time)
                ```bash
                sudo curl -fsSL https://ollama.ai/install.sh | sh
                ```
                
                ### 2. Start Ollama Service
                ```bash
                ollama serve
                ```
                
                ### 3. Pull LLaMA 3.2 (3GB)
                ```bash
                ollama pull llama3.2
                ```
                
                ### 4. Test it works
                ```bash
                ollama run llama3.2 "Hello world"
                ```
                
                ### Available Models
                - **llama3.2** - 3GB, fast, good quality
                - **mistral** - 4GB, very good
                - **phi3** - 2GB, fastest
                - **llama2:13b** - 13GB, excellent
                
                ### Why Ollama?
                - ✅ Easy to install and use
                - ✅ Optimized for CPU/GPU
                - ✅ Many models available
                - ✅ Fast inference
                - ✅ No Python dependencies issues
                """)
        
        # Event handlers
        def send_message(msg, history):
            return "", rag.chat(msg, history)
        
        msg.submit(send_message, [msg, chatbot], [msg, chatbot])
        send_btn.click(send_message, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: [], outputs=[chatbot])
        
        upload_btn.click(
            rag.load_documents,
            inputs=[file_upload],
            outputs=[upload_status, file_list]
        )
        
        model_btn.click(
            rag.pull_model,
            outputs=[model_status]
        )
    
    return app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🦙 Ollama RAG with LLaMA 3.2")
    print("="*60)
    print("\nSetup Instructions:")
    print("1. Install: sudo curl -fsSL https://ollama.ai/install.sh | sh")
    print("2. Start: ollama serve (in another terminal)")
    print("3. Pull model: ollama pull llama3.2")
    print("="*60 + "\n")
    
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)