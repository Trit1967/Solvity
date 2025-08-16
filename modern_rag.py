#!/usr/bin/env python3
"""
Modern RAG Chatbot with LLM - Clean, Professional UI
Using DistilBERT for Q&A (lightweight, runs on CPU)
"""

import gradio as gr
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch
import os
from pathlib import Path
import json
from datetime import datetime
from typing import List, Tuple

class ModernRAG:
    def __init__(self):
        """Initialize with a lightweight Q&A model"""
        print("🚀 Loading AI model... (one-time download)")
        
        # Use DistilBERT - small, fast, good for Q&A
        model_name = "distilbert-base-cased-distilled-squad"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1  # CPU
            )
            print("✅ AI Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.qa_pipeline = None
        
        self.documents = {}
        self.chat_history = []
        
    def load_documents(self, files):
        """Load and process documents"""
        if not files:
            return "No files uploaded", [], gr.update(value="")
        
        self.documents = {}
        loaded = []
        total_chars = 0
        
        for file in files:
            try:
                # Read file content
                if file.name.endswith('.pdf'):
                    content = f"[PDF file - install pypdf for full support]"
                else:
                    with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                
                filename = Path(file.name).name
                self.documents[filename] = content
                loaded.append(filename)
                total_chars += len(content)
                
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        
        if loaded:
            status = f"✅ Loaded {len(loaded)} files ({total_chars:,} characters)"
            # Create file cards
            file_info = [[f"📄 {f}", f"{len(self.documents[f]):,} chars"] for f in loaded]
            return status, file_info, gr.update(value=f"{len(loaded)} documents ready")
        
        return "❌ No files loaded", [], gr.update(value="")
    
    def answer_question(self, question: str, history: List[Tuple[str, str]]):
        """Answer questions using the loaded documents"""
        
        if not self.documents:
            response = "📚 Please upload some documents first using the Documents tab."
            return history + [(question, response)]
        
        if not question.strip():
            return history
        
        if self.qa_pipeline is None:
            response = "❌ AI model not loaded. Using keyword search instead..."
            # Fallback to simple search
            results = self.simple_search(question)
            return history + [(question, results)]
        
        # Combine all documents into context
        context = "\n\n".join([
            f"=== {filename} ===\n{content[:2000]}"  # Limit per document
            for filename, content in self.documents.items()
        ])
        
        # Truncate context to model limits (512 tokens for DistilBERT)
        max_context = 3000  # Characters, not tokens
        if len(context) > max_context:
            context = context[:max_context]
        
        try:
            # Get answer from AI
            result = self.qa_pipeline(
                question=question,
                context=context
            )
            
            answer = result['answer']
            confidence = result['score']
            
            # Format response
            if confidence > 0.5:
                response = f"💡 {answer}"
            elif confidence > 0.2:
                response = f"🤔 {answer}\n\n*Confidence: {confidence:.1%}*"
            else:
                response = f"❓ Low confidence answer: {answer}\n\nTry rephrasing your question or adding more specific documents."
            
            # Add source info if we can identify it
            for filename, content in self.documents.items():
                if answer.lower() in content.lower():
                    response += f"\n\n📄 *Source: {filename}*"
                    break
                    
        except Exception as e:
            print(f"Error: {e}")
            response = "❌ Error processing question. Try a simpler query."
        
        return history + [(question, response)]
    
    def simple_search(self, query: str) -> str:
        """Fallback keyword search"""
        query_lower = query.lower()
        results = []
        
        for filename, content in self.documents.items():
            if query_lower in content.lower():
                # Find the sentence containing the query
                sentences = content.split('.')
                for sent in sentences:
                    if query_lower in sent.lower():
                        results.append(f"📄 **{filename}**: ...{sent.strip()}...")
                        break
        
        if results:
            return "\n\n".join(results[:3])  # Top 3 results
        return "No matches found. Try different keywords."
    
    def clear_all(self):
        """Clear documents and history"""
        self.documents = {}
        self.chat_history = []
        return [], "All data cleared", [], gr.update(value="")

def create_modern_ui():
    """Create a professional, modern UI"""
    
    rag = ModernRAG()
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }
    .gr-button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    #chat-box {
        height: 500px !important;
    }
    .user-message {
        background: #e3f2fd !important;
        border-radius: 18px !important;
        padding: 10px 15px !important;
    }
    .bot-message {
        background: #f5f5f5 !important;
        border-radius: 18px !important;
        padding: 10px 15px !important;
    }
    """
    
    with gr.Blocks(
        title="AI Document Assistant",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate"
        ),
        css=custom_css
    ) as app:
        
        # Header
        with gr.Row():
            gr.Markdown("""
            # 🤖 AI Document Assistant
            #### Chat with your documents using advanced AI - 100% local and private
            """)
        
        # Status bar
        status_bar = gr.Textbox(
            value="Ready to load documents",
            label="",
            interactive=False,
            elem_id="status-bar"
        )
        
        # Main tabs
        with gr.Tabs() as tabs:
            
            # Chat Tab
            with gr.Tab("💬 Chat", id=1):
                chatbot = gr.Chatbot(
                    height=500,
                    elem_id="chat-box",
                    bubble_full_width=False,
                    avatar_images=["🧑", "🤖"]
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask anything about your documents...",
                        label="",
                        lines=2,
                        scale=6
                    )
                    send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat")
                    example_btn = gr.Button("💡 Show Examples")
                
                # Example questions
                examples = gr.Examples(
                    examples=[
                        "What is the main topic of this document?",
                        "Summarize the key points",
                        "What are the important dates mentioned?",
                        "Who are the main people discussed?",
                        "What conclusions are drawn?"
                    ],
                    inputs=msg,
                    label="Example Questions"
                )
            
            # Documents Tab
            with gr.Tab("📁 Documents", id=2):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_upload = gr.File(
                            label="Upload Documents",
                            file_count="multiple",
                            file_types=[".txt", ".md", ".log", ".csv", ".json"],
                            height=200
                        )
                        
                        upload_btn = gr.Button("📥 Process Documents", variant="primary", size="lg")
                        clear_docs_btn = gr.Button("🗑️ Clear All Documents", variant="stop")
                    
                    with gr.Column(scale=2):
                        upload_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=1
                        )
                        
                        file_list = gr.Dataframe(
                            headers=["File", "Size"],
                            label="Loaded Documents",
                            interactive=False,
                            row_count=10
                        )
            
            # Settings Tab
            with gr.Tab("⚙️ Settings", id=3):
                gr.Markdown("""
                ### About This System
                
                This AI-powered document assistant uses:
                - **Model**: DistilBERT (66M parameters)
                - **Task**: Question-Answering
                - **Privacy**: 100% local processing
                - **Speed**: Real-time responses on CPU
                
                ### Tips for Best Results
                
                1. **Upload relevant documents** - The AI searches within your uploaded files
                2. **Ask specific questions** - "What date..." works better than "Tell me about..."
                3. **Keep documents focused** - Better results with related documents
                4. **Use keywords** - Include important terms from your documents
                
                ### System Requirements
                
                - **RAM**: 2-4 GB
                - **Disk**: 500 MB for model
                - **CPU**: Any modern processor
                - **GPU**: Not required
                """)
                
                with gr.Row():
                    gr.Markdown(f"""
                    **Model Status**: {'✅ Loaded' if rag.qa_pipeline else '❌ Not loaded'}  
                    **Documents**: {len(rag.documents)} loaded  
                    **Device**: CPU
                    """)
        
        # Event handlers
        def send_message(msg, history):
            return "", rag.answer_question(msg, history)
        
        msg.submit(send_message, [msg, chatbot], [msg, chatbot])
        send_btn.click(send_message, [msg, chatbot], [msg, chatbot])
        
        upload_btn.click(
            rag.load_documents,
            inputs=[file_upload],
            outputs=[upload_status, file_list, status_bar]
        )
        
        clear_btn.click(
            lambda: [],
            outputs=[chatbot]
        )
        
        clear_docs_btn.click(
            rag.clear_all,
            outputs=[chatbot, upload_status, file_list, status_bar]
        )
    
    return app

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting AI Document Assistant")
    print("="*50)
    print("📥 First run will download AI model (~250MB)")
    print("🌐 Opening at http://localhost:7860")
    print("="*50 + "\n")
    
    app = create_modern_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        show_error=True
    )