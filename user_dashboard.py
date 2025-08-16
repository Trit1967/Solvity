#!/usr/bin/env python3
"""
USER DASHBOARD - Simple, Clean Interface for End Users
All the power, none of the complexity
"""

import gradio as gr
import requests
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import hashlib
import PyPDF2
import pdfplumber
from docx import Document
import pandas as pd
import pickle
from datetime import datetime

class SimpleRAG:
    def __init__(self):
        # Optimized default settings
        self.config = {
            'model': 'llama3.2',
            'base_url': 'http://localhost:11434',
            'similarity_threshold': 0.12,  # Optimized for best results
            'chunk_size': 1500,
            'chunk_overlap': 200,
            'top_k_results': 5
        }
        
        # Initialize
        print("🚀 Starting RAG system...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Ready!")
        
        # Storage
        self.documents = {}
        self.chunks = []
        self.chunk_embeddings = None
        self.chunk_metadata = []
        
        # Simple caching
        self.query_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    def extract_text(self, file_path: str) -> str:
        """Extract text from any file type"""
        filename = Path(file_path).name.lower()
        
        try:
            if filename.endswith('.pdf'):
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                return text.strip()
            
            elif filename.endswith('.docx'):
                doc = Document(file_path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                return "\n\n".join(paragraphs)
            
            elif filename.endswith(('.csv', '.xlsx', '.xls')):
                # Handle spreadsheets
                df = pd.read_excel(file_path) if filename.endswith(('.xlsx', '.xls')) else pd.read_csv(file_path)
                text_parts = [f"Spreadsheet: {filename}"]
                text_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                text_parts.append(f"Columns: {', '.join(df.columns)}")
                text_parts.append("\nData Preview:")
                text_parts.append(df.head(50).to_string())
                return "\n".join(text_parts)
            
            else:
                # Plain text files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
                    
        except Exception as e:
            print(f"Error extracting {filename}: {e}")
            return f"[Could not extract from {filename}]"
    
    def smart_chunk(self, text: str, filename: str) -> List[Dict]:
        """Intelligent chunking"""
        chunks = []
        
        # For resumes/CVs, keep overview
        if any(keyword in filename.lower() for keyword in ['resume', 'cv', 'bio']):
            overview = text[:3000] if len(text) > 3000 else text
            chunks.append({
                'text': overview,
                'source': filename,
                'type': 'overview'
            })
        
        # Regular chunking
        words = text.split()
        chunk_words = self.config['chunk_size'] // 5
        overlap_words = self.config['chunk_overlap'] // 5
        
        for i in range(0, len(words), chunk_words - overlap_words):
            chunk_text = ' '.join(words[i:i + chunk_words])
            if len(chunk_text) > 50:
                chunks.append({
                    'text': chunk_text,
                    'source': filename,
                    'type': 'content'
                })
        
        return chunks if chunks else [{'text': text[:2000], 'source': filename, 'type': 'fallback'}]
    
    def load_documents(self, files):
        """Load and process documents"""
        if not files:
            return "Please select files to upload", "", "No documents loaded"
        
        self.documents = {}
        self.chunks = []
        self.chunk_metadata = []
        loaded_files = []
        total_size = 0
        
        for file in files:
            try:
                filename = Path(file.name).name
                
                # Extract text
                content = self.extract_text(file.name)
                
                if content and len(content) > 10:
                    self.documents[filename] = content
                    total_size += len(content)
                    
                    # Create chunks
                    file_chunks = self.smart_chunk(content, filename)
                    
                    for chunk in file_chunks:
                        self.chunks.append(chunk['text'])
                        self.chunk_metadata.append({
                            'source': chunk['source'],
                            'type': chunk['type']
                        })
                    
                    loaded_files.append(f"✅ {filename} ({len(file_chunks)} sections)")
                else:
                    loaded_files.append(f"⚠️ {filename} (empty)")
                    
            except Exception as e:
                loaded_files.append(f"❌ {filename} (error)")
                print(f"Error loading {filename}: {e}")
        
        if self.chunks:
            # Create embeddings
            print(f"Processing {len(self.chunks)} text sections...")
            self.chunk_embeddings = self.embedder.encode(self.chunks)
            
            # Clear cache when new docs loaded
            self.query_cache.clear()
            
            status = f"✅ Loaded {len([f for f in loaded_files if '✅' in f])} files successfully"
            details = "\n".join(loaded_files)
            info = f"📚 {len(self.documents)} documents | 📄 {len(self.chunks)} searchable sections | 💾 {total_size:,} characters"
            
            return status, details, info
        
        return "❌ No valid documents loaded", "\n".join(loaded_files), "No content to search"
    
    def search(self, query: str, use_cache: bool = True) -> List[Dict]:
        """Search with caching"""
        if not self.chunks:
            return []
        
        # Check cache
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if use_cache and cache_key in self.query_cache:
            cached = self.query_cache[cache_key]
            if time.time() - cached['time'] < self.cache_ttl:
                return cached['results']
        
        # Perform search
        query_embedding = self.embedder.encode([query])
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[-self.config['top_k_results']:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > self.config['similarity_threshold']:
                results.append({
                    'text': self.chunks[idx],
                    'score': float(similarities[idx]),
                    'source': self.chunk_metadata[idx]['source']
                })
        
        # Always return at least one result
        if not results and len(top_indices) > 0:
            idx = top_indices[0]
            results.append({
                'text': self.chunks[idx],
                'score': float(similarities[idx]),
                'source': self.chunk_metadata[idx]['source']
            })
        
        # Cache results
        if use_cache:
            self.query_cache[cache_key] = {
                'results': results,
                'time': time.time()
            }
        
        return results
    
    def chat(self, message: str, history: List[Tuple[str, str]]):
        """Process chat messages"""
        if not message.strip():
            return history
        
        history.append([message, None])
        
        # Search for relevant content
        if self.chunks:
            results = self.search(message)
            
            if results:
                # Build context
                context = "\n\n".join([r['text'] for r in results[:3]])
                sources = list(set([r['source'] for r in results]))
                
                prompt = f"""Based on the following information, please answer the question.

Information:
{context[:4000]}

Question: {message}

Please provide a clear and helpful answer:"""
                
                # Get response from Ollama
                try:
                    response = requests.post(
                        f"{self.config['base_url']}/api/generate",
                        json={
                            "model": self.config['model'],
                            "prompt": prompt,
                            "stream": False,
                            "temperature": 0.7
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        answer = response.json().get('response', 'No response generated')
                        answer += f"\n\n📄 Sources: {', '.join(sources[:3])}"
                    else:
                        answer = "⚠️ AI model is not responding. Please check if Ollama is running."
                        
                except requests.exceptions.ConnectionError:
                    answer = "❌ Cannot connect to AI model. Please ensure Ollama is running:\n```bash\nollama serve\nollama run llama3.2\n```"
                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
            else:
                answer = "I don't have any relevant information to answer that question. Please upload documents first."
        else:
            # No documents loaded
            answer = "Please upload documents first. I need information to answer your questions."
        
        history[-1][1] = answer
        return history

def create_user_interface():
    """Create the clean user interface"""
    
    rag = SimpleRAG()
    
    # Professional styling
    custom_css = """
    :root {
        --primary: #4F46E5;
        --primary-dark: #4338CA;
        --success: #10B981;
        --bg: #F9FAFB;
        --surface: #FFFFFF;
        --text: #111827;
        --text-secondary: #6B7280;
        --border: #E5E7EB;
    }
    
    .gradio-container {
        background: var(--bg) !important;
        font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .header-banner {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.2);
    }
    
    .chat-container {
        background: var(--surface);
        border-radius: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        padding: 1rem;
    }
    
    .upload-area {
        border: 2px dashed var(--border);
        border-radius: 0.75rem;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s;
        background: var(--surface);
    }
    
    .upload-area:hover {
        border-color: var(--primary);
        background: rgba(79, 70, 229, 0.02);
    }
    
    .status-success {
        color: var(--success);
        font-weight: 500;
    }
    
    .btn-primary {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    
    .btn-primary:hover {
        background: var(--primary-dark) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    """
    
    with gr.Blocks(css=custom_css, title="Document Assistant") as app:
        
        # Header
        gr.HTML("""
        <div class="header-banner">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">
                📚 Document Assistant
            </h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.95;">
                Upload documents and ask questions - it's that simple!
            </p>
        </div>
        """)
        
        with gr.Row():
            # Left: Document Upload
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Your Documents")
                
                file_upload = gr.File(
                    label="Drag & drop files here",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx"],
                    elem_classes="upload-area"
                )
                
                upload_btn = gr.Button(
                    "📤 Process Documents",
                    variant="primary",
                    elem_classes="btn-primary"
                )
                
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=1
                )
                
                file_list = gr.Textbox(
                    label="Loaded Files",
                    interactive=False,
                    lines=5
                )
                
                doc_info = gr.Textbox(
                    label="Statistics",
                    interactive=False,
                    max_lines=1
                )
                
                # Help section
                with gr.Accordion("💡 Tips", open=False):
                    gr.Markdown("""
                    **Supported Files:**
                    - PDF documents
                    - Word documents (.docx)
                    - Text files (.txt, .md)
                    - Spreadsheets (.csv, .xlsx)
                    
                    **Best Practices:**
                    - Upload all related documents at once
                    - For resumes: name file with "resume" or "cv"
                    - Larger documents take longer to process
                    
                    **Example Questions:**
                    - "Summarize this document"
                    - "What are the key points?"
                    - "Create a bio from this resume"
                    - "What experience is mentioned?"
                    """)
            
            # Right: Chat Interface
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask Questions")
                
                chatbot = gr.Chatbot(
                    height=500,
                    elem_classes="chat-container",
                    show_label=False,
                    avatar_images=["👤", "🤖"]
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your question here...",
                        show_label=False,
                        scale=4,
                        lines=1
                    )
                    
                    send_btn = gr.Button(
                        "Send →",
                        scale=1,
                        variant="primary"
                    )
                
                clear_btn = gr.Button("🗑️ Clear Conversation")
                
                # Quick examples
                gr.Examples(
                    examples=[
                        "Summarize the main points",
                        "What is this document about?",
                        "Create a professional bio",
                        "What are the key findings?",
                        "List the important dates mentioned"
                    ],
                    inputs=msg,
                    label="Quick Questions"
                )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #6B7280;">
            <p>💡 Tip: Upload your documents first, then ask any questions about them!</p>
        </div>
        """)
        
        # Event handlers
        upload_btn.click(
            rag.load_documents,
            inputs=[file_upload],
            outputs=[upload_status, file_list, doc_info]
        )
        
        msg.submit(
            lambda m, h: ("", rag.chat(m, h)),
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        send_btn.click(
            lambda m, h: ("", rag.chat(m, h)),
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(
            lambda: [],
            outputs=[chatbot]
        )
    
    return app

if __name__ == "__main__":
    print("\n" + "="*50)
    print("📚 USER DASHBOARD - Document Assistant")
    print("="*50)
    print("Simple, clean interface for end users")
    print("Optimized settings for best performance")
    print("="*50 + "\n")
    
    app = create_user_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )