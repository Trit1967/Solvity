#!/usr/bin/env python3
"""
Web UI for GPT-Neo RAG Chatbot using Gradio
Clean, modern interface that runs in your browser
"""

import gradio as gr
import os
import torch
from pathlib import Path
from typing import List, Tuple
import shutil
from datetime import datetime

# Import our RAG system
from rag_gpt_neo import GPTNeoRAG

class RAGWebUI:
    def __init__(self, model_size="1.3B"):
        """Initialize the RAG system and UI components"""
        self.rag = None
        self.model_size = model_size
        self.chat_history = []
        
    def initialize_model(self, model_size, progress=gr.Progress()):
        """Initialize or reinitialize the model"""
        progress(0.3, desc="Loading model...")
        try:
            self.rag = GPTNeoRAG(model_size=model_size)
            self.model_size = model_size
            progress(1.0, desc="Model loaded!")
            return f"✅ Model {model_size} loaded successfully!"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def ingest_files(self, files, progress=gr.Progress()):
        """Ingest uploaded documents"""
        if not files:
            return "❌ No files uploaded", ""
        
        if self.rag is None:
            return "❌ Please load a model first", ""
        
        # Create temp directory for uploaded files
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        progress(0.2, desc="Processing files...")
        
        # Copy uploaded files to temp directory
        file_list = []
        for file in files:
            dest_path = os.path.join(temp_dir, os.path.basename(file.name))
            shutil.copy(file.name, dest_path)
            file_list.append(os.path.basename(file.name))
        
        progress(0.5, desc="Creating embeddings...")
        
        # Ingest documents
        try:
            self.rag.ingest_documents(temp_dir)
            
            # Clean up temp files
            shutil.rmtree(temp_dir)
            
            progress(1.0, desc="Complete!")
            
            return (
                f"✅ Successfully ingested {len(files)} files",
                "\n".join(f"• {f}" for f in file_list)
            )
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return f"❌ Error: {str(e)}", ""
    
    def chat(self, message, history):
        """Process chat messages"""
        if self.rag is None:
            return history + [[message, "❌ Please load a model first"]]
        
        if self.rag.vectorstore is None:
            return history + [[message, "❌ Please ingest some documents first"]]
        
        # Get response from RAG
        try:
            response = self.rag.query(message)
            return history + [[message, response]]
        except Exception as e:
            return history + [[message, f"❌ Error: {str(e)}"]]
    
    def clear_chat(self):
        """Clear chat history"""
        return []
    
    def create_ui(self):
        """Create the Gradio interface"""
        
        # Custom CSS for better styling
        custom_css = """
        .container {
            max-width: 1200px;
            margin: auto;
        }
        .chat-container {
            height: 500px;
        }
        footer {
            display: none !important;
        }
        """
        
        with gr.Blocks(title="RAG Chatbot", theme=gr.themes.Soft(), css=custom_css) as app:
            
            # Header
            gr.Markdown(
                """
                # 🤖 GPT-Neo RAG Chatbot
                ### Local, Private, and Free Document Q&A System
                """
            )
            
            with gr.Tabs():
                # Chat Tab
                with gr.TabItem("💬 Chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                height=500,
                                show_label=False,
                                elem_classes="chat-container"
                            )
                            
                            msg = gr.Textbox(
                                label="Ask a question",
                                placeholder="Type your question here...",
                                lines=2
                            )
                            
                            with gr.Row():
                                submit_btn = gr.Button("Send", variant="primary")
                                clear_btn = gr.Button("Clear Chat")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 📊 System Info")
                            model_status = gr.Textbox(
                                label="Current Model",
                                value=f"Model: {self.model_size}",
                                interactive=False
                            )
                            
                            device_info = gr.Textbox(
                                label="Device",
                                value=f"Running on: {'GPU' if torch.cuda.is_available() else 'CPU'}",
                                interactive=False
                            )
                            
                            gr.Markdown("### 💡 Tips")
                            gr.Markdown(
                                """
                                - Upload documents first
                                - Ask specific questions
                                - Reference document names
                                - Be patient with large docs
                                """
                            )
                
                # Document Management Tab
                with gr.TabItem("📄 Documents"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Upload Documents")
                            file_upload = gr.File(
                                label="Select files to upload",
                                file_count="multiple",
                                file_types=[".pdf", ".txt", ".md", ".docx"]
                            )
                            
                            ingest_btn = gr.Button("📥 Ingest Documents", variant="primary")
                            
                            ingest_status = gr.Textbox(
                                label="Status",
                                interactive=False
                            )
                            
                            file_list = gr.Textbox(
                                label="Ingested Files",
                                lines=10,
                                interactive=False
                            )
                
                # Settings Tab
                with gr.TabItem("⚙️ Settings"):
                    gr.Markdown("### Model Configuration")
                    
                    model_selector = gr.Dropdown(
                        choices=["125M", "1.3B", "2.7B", "6B"],
                        value=self.model_size,
                        label="Model Size",
                        info="Larger models are more accurate but slower"
                    )
                    
                    load_model_btn = gr.Button("🔄 Load Model", variant="primary")
                    
                    model_load_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                    
                    gr.Markdown(
                        """
                        ### Model Requirements
                        | Model | RAM | Quality | Speed |
                        |-------|-----|---------|-------|
                        | 125M | 2GB | Basic | Very Fast |
                        | 1.3B | 6GB | Good | Fast |
                        | 2.7B | 12GB | Better | Medium |
                        | 6B | 16GB+ | Best | Slow |
                        """
                    )
                
                # About Tab
                with gr.TabItem("ℹ️ About"):
                    gr.Markdown(
                        """
                        ## About This System
                        
                        This is a **100% local** RAG (Retrieval-Augmented Generation) system that:
                        
                        - ✅ Runs entirely on your machine
                        - ✅ Never sends data to the cloud
                        - ✅ Works offline after initial setup
                        - ✅ Completely free to use
                        
                        ### How It Works
                        
                        1. **Upload Documents**: PDF, TXT, MD, or DOCX files
                        2. **Ask Questions**: The AI searches your documents
                        3. **Get Answers**: With source references
                        
                        ### Technology Stack
                        
                        - **LLM**: GPT-Neo/GPT-J (EleutherAI)
                        - **Embeddings**: Sentence-Transformers
                        - **Vector DB**: FAISS
                        - **Framework**: LangChain
                        - **UI**: Gradio
                        
                        ### Privacy First
                        
                        All processing happens locally. Your documents and conversations never leave your computer.
                        """
                    )
            
            # Event handlers
            submit_btn.click(
                fn=self.chat,
                inputs=[msg, chatbot],
                outputs=chatbot
            )
            
            msg.submit(
                fn=self.chat,
                inputs=[msg, chatbot],
                outputs=chatbot
            )
            
            clear_btn.click(
                fn=self.clear_chat,
                outputs=chatbot
            )
            
            ingest_btn.click(
                fn=self.ingest_files,
                inputs=file_upload,
                outputs=[ingest_status, file_list]
            )
            
            load_model_btn.click(
                fn=self.initialize_model,
                inputs=model_selector,
                outputs=model_load_status
            )
        
        return app

def main():
    """Launch the web UI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Chatbot Web UI")
    parser.add_argument(
        "--model",
        choices=["125M", "1.3B", "2.7B", "6B"],
        default="1.3B",
        help="Initial model size"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link (for sharing)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on"
    )
    
    args = parser.parse_args()
    
    # Create UI
    print("🚀 Starting RAG Chatbot Web UI...")
    ui = RAGWebUI(model_size=args.model)
    
    # Initialize model
    print(f"📥 Loading {args.model} model...")
    ui.initialize_model(args.model)
    
    # Launch app
    app = ui.create_ui()
    app.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )

if __name__ == "__main__":
    main()