#!/usr/bin/env python3
"""
OpenAI-Powered RAG (Using API)
Much better quality but costs money
"""

import gradio as gr
import os
from pathlib import Path
from typing import List, Tuple
import openai
from openai import OpenAI

class OpenAIRAG:
    def __init__(self, api_key: str = None):
        """Initialize with OpenAI API"""
        
        # Check for API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("❌ No OpenAI API key found!")
            print("\nTo use OpenAI:")
            print("1. Get API key from https://platform.openai.com/api-keys")
            print("2. Set environment variable: export OPENAI_API_KEY='your-key'")
            print("3. Or enter it in the Settings tab\n")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)
                print("✅ OpenAI API connected!")
                # Test the connection
                self.test_connection()
            except Exception as e:
                print(f"❌ OpenAI connection failed: {e}")
                self.client = None
        
        self.documents = {}
        self.embeddings_cache = {}
        
    def test_connection(self):
        """Test OpenAI connection"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            print(f"✅ API test successful! Using GPT-3.5-turbo")
            return True
        except Exception as e:
            print(f"❌ API test failed: {e}")
            return False
    
    def update_api_key(self, api_key: str):
        """Update API key"""
        if api_key:
            self.api_key = api_key
            try:
                self.client = OpenAI(api_key=api_key)
                if self.test_connection():
                    return "✅ API key updated and tested successfully!"
            except Exception as e:
                return f"❌ Invalid API key: {e}"
        return "❌ Please enter an API key"
    
    def load_documents(self, files):
        """Load documents"""
        if not files:
            return "No files uploaded", []
        
        self.documents = {}
        loaded = []
        
        for file in files:
            try:
                with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                filename = Path(file.name).name
                self.documents[filename] = content
                loaded.append([filename, f"{len(content):,} chars"])
                
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        
        if loaded:
            return f"✅ Loaded {len(loaded)} documents", loaded
        return "❌ No files loaded", []
    
    def chat_with_gpt(self, question: str, history: List[Tuple[str, str]]):
        """Chat using OpenAI GPT"""
        
        if not self.client:
            response = "❌ Please add your OpenAI API key in the Settings tab"
            return history + [(question, response)]
        
        if not self.documents:
            response = "📚 Please upload documents first"
            return history + [(question, response)]
        
        if not question.strip():
            return history
        
        try:
            # Prepare context from documents
            context = ""
            for filename, content in self.documents.items():
                # Use first 2000 chars from each doc
                context += f"\n--- {filename} ---\n{content[:2000]}\n"
            
            # Create the prompt
            system_prompt = f"""You are a helpful assistant analyzing these documents:
            
{context[:4000]}

Answer questions based on the document content above."""
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Cheaper than GPT-4
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Add cost estimate
            tokens_used = response.usage.total_tokens
            cost = (tokens_used / 1000) * 0.002  # GPT-3.5 pricing
            
            answer += f"\n\n*[Tokens: {tokens_used}, Cost: ${cost:.4f}]*"
            
        except Exception as e:
            answer = f"❌ Error: {str(e)}"
        
        return history + [(question, answer)]
    
    def estimate_cost(self) -> str:
        """Estimate costs"""
        if not self.documents:
            return "No documents loaded"
        
        total_chars = sum(len(doc) for doc in self.documents.values())
        estimated_tokens = total_chars / 4  # Rough estimate
        
        costs = f"""
### Cost Estimates (per question):
        
**GPT-3.5-turbo**: ~${(estimated_tokens/1000 * 0.002):.4f}
**GPT-4**: ~${(estimated_tokens/1000 * 0.06):.4f}
**Embeddings**: ~${(estimated_tokens/1000 * 0.0001):.4f}

### Monthly estimates:
- 100 questions/month: ~${(estimated_tokens/1000 * 0.002 * 100):.2f}
- 1000 questions/month: ~${(estimated_tokens/1000 * 0.002 * 1000):.2f}
"""
        return costs

def create_openai_ui():
    """Create UI for OpenAI RAG"""
    
    rag = OpenAIRAG()
    
    with gr.Blocks(title="OpenAI RAG", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🤖 OpenAI-Powered RAG
        ### Better quality using GPT-3.5/GPT-4 (requires API key)
        """)
        
        # API Status
        api_status = gr.Textbox(
            value="❌ No API key" if not rag.client else "✅ API connected",
            label="API Status",
            interactive=False
        )
        
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(height=400)
                
                msg = gr.Textbox(
                    placeholder="Ask anything about your documents...",
                    label="Question",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("Send 🚀", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                gr.Examples(
                    examples=[
                        "Summarize these documents",
                        "What are the key points?",
                        "Extract all dates mentioned",
                        "What is the main conclusion?"
                    ],
                    inputs=msg
                )
            
            # Documents Tab
            with gr.Tab("📁 Documents"):
                file_upload = gr.File(
                    label="Upload Documents",
                    file_count="multiple"
                )
                
                upload_btn = gr.Button("📥 Load Documents", variant="primary")
                
                upload_status = gr.Textbox(label="Status")
                file_list = gr.Dataframe(
                    headers=["File", "Size"],
                    label="Loaded Documents"
                )
            
            # Settings Tab
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("""
                ### OpenAI API Setup
                
                1. Get API key from: https://platform.openai.com/api-keys
                2. Enter it below (it's stored only in this session)
                """)
                
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-..."
                )
                
                update_key_btn = gr.Button("Update API Key", variant="primary")
                key_status = gr.Textbox(label="Key Status")
                
                gr.Markdown("### Cost Information")
                cost_info = gr.Markdown(rag.estimate_cost())
                
                gr.Markdown("""
                ### Comparison: OpenAI vs Open-Source
                
                | Feature | OpenAI (This) | Open-Source (GPT-Neo, etc) |
                |---------|--------------|---------------------------|
                | **Quality** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
                | **Speed** | ⚡ Very Fast | 🐢 Slower |
                | **Cost** | 💰 ~$0.002/query | ✅ Free |
                | **Privacy** | ☁️ Cloud | 🔒 Local |
                | **Setup** | Easy (API key) | Complex |
                | **Internet** | Required | Not needed |
                """)
        
        # Event handlers
        def send_message(msg, history):
            return "", rag.chat_with_gpt(msg, history)
        
        msg.submit(send_message, [msg, chatbot], [msg, chatbot])
        send_btn.click(send_message, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: [], outputs=[chatbot])
        
        upload_btn.click(
            rag.load_documents,
            inputs=[file_upload],
            outputs=[upload_status, file_list]
        )
        
        def update_key(key):
            status = rag.update_api_key(key)
            api_status_text = "✅ API connected" if "successful" in status else "❌ No API key"
            cost_text = rag.estimate_cost()
            return status, api_status_text, cost_text
        
        update_key_btn.click(
            update_key,
            inputs=[api_key_input],
            outputs=[key_status, api_status, cost_info]
        )
    
    return app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("💰 OpenAI RAG (Requires API Key)")
    print("="*60)
    print("\nTwo options:")
    print("\n1. FREE OPEN-SOURCE (what we've been using):")
    print("   - GPT-Neo, BART, T5")
    print("   - Runs locally, no API needed")
    print("   - Lower quality")
    print("\n2. OPENAI API (this file):")
    print("   - GPT-3.5-turbo or GPT-4")
    print("   - Much better quality")
    print("   - Costs ~$0.002 per question")
    print("   - Needs API key from https://platform.openai.com")
    print("="*60 + "\n")
    
    app = create_openai_ui()
    app.launch(server_port=7860, inbrowser=True)