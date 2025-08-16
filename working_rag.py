#!/usr/bin/env python3
"""
Working RAG - Free, Local, and Actually Works
Uses sentence embeddings for semantic search + better models
"""

import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from pathlib import Path
from typing import List, Tuple, Dict
import re

class WorkingRAG:
    def __init__(self):
        """Initialize with working components"""
        print("\n🚀 Loading Working RAG System...")
        
        # Load embedding model for semantic search (not keyword matching!)
        print("Loading sentence embeddings for semantic search...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load a better model for generation
        print("Loading language model...")
        try:
            # Try FLAN-T5 first (better for Q&A)
            self.generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=-1,
                max_length=512
            )
            self.model_name = "FLAN-T5"
        except:
            # Fallback to smaller model
            self.generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device=-1
            )
            self.model_name = "FLAN-T5-Small"
        
        print(f"✅ Models loaded! Using {self.model_name}")
        
        self.documents = {}
        self.chunks = []
        self.chunk_embeddings = None
        self.chunk_metadata = []
    
    def smart_chunk_document(self, text: str, filename: str) -> List[Dict]:
        """Intelligently chunk documents"""
        chunks = []
        
        # Clean text
        text = text.strip()
        
        if filename.endswith('.md'):
            # Markdown - split by headers
            sections = re.split(r'\n(?=#{1,6}\s+)', text)
            
            for section in sections:
                if len(section.strip()) > 50:  # Minimum chunk size
                    # Keep sections under 1000 chars
                    if len(section) > 1000:
                        # Split by paragraphs
                        paragraphs = section.split('\n\n')
                        current = ""
                        for para in paragraphs:
                            if len(current) + len(para) < 1000:
                                current += para + "\n\n"
                            else:
                                if current:
                                    chunks.append({
                                        'text': current.strip(),
                                        'source': filename,
                                        'type': 'markdown_section'
                                    })
                                current = para
                        if current:
                            chunks.append({
                                'text': current.strip(),
                                'source': filename,
                                'type': 'markdown_section'
                            })
                    else:
                        chunks.append({
                            'text': section.strip(),
                            'source': filename,
                            'type': 'markdown_section'
                        })
        
        elif filename.endswith('.txt') or filename.endswith('.text'):
            # Plain text - split by paragraphs
            paragraphs = text.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < 800:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'source': filename,
                            'type': 'text_paragraph'
                        })
                    current_chunk = para
            
            if current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'source': filename,
                    'type': 'text_paragraph'
                })
        
        else:
            # Generic - split by size
            words = text.split()
            current = []
            
            for word in words:
                current.append(word)
                if len(' '.join(current)) > 500:
                    chunks.append({
                        'text': ' '.join(current),
                        'source': filename,
                        'type': 'generic'
                    })
                    current = []
            
            if current:
                chunks.append({
                    'text': ' '.join(current),
                    'source': filename,
                    'type': 'generic'
                })
        
        # If no chunks created, use whole document
        if not chunks:
            chunks = [{
                'text': text[:2000],
                'source': filename,
                'type': 'whole_doc'
            }]
        
        return chunks
    
    def load_documents(self, files):
        """Load and process documents with embeddings"""
        if not files:
            return "No files", [], "Not ready"
        
        self.documents = {}
        self.chunks = []
        self.chunk_metadata = []
        
        loaded_info = []
        
        for file in files:
            try:
                # Read file
                with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                filename = Path(file.name).name
                self.documents[filename] = content
                
                # Smart chunking
                file_chunks = self.smart_chunk_document(content, filename)
                
                # Add to chunks
                for chunk in file_chunks:
                    self.chunks.append(chunk['text'])
                    self.chunk_metadata.append({
                        'source': chunk['source'],
                        'type': chunk['type'],
                        'length': len(chunk['text'])
                    })
                
                loaded_info.append([
                    f"📄 {filename}",
                    f"{len(content):,} chars",
                    f"{len(file_chunks)} chunks"
                ])
                
            except Exception as e:
                print(f"Error: {e}")
        
        if self.chunks:
            # Create embeddings for semantic search
            print(f"Creating embeddings for {len(self.chunks)} chunks...")
            self.chunk_embeddings = self.embedder.encode(self.chunks)
            print("✅ Embeddings created!")
            
            status = f"✅ Loaded {len(files)} files, {len(self.chunks)} chunks"
            ready = f"Ready! {len(self.chunks)} searchable chunks"
        else:
            status = "❌ No chunks created"
            ready = "Not ready"
        
        return status, loaded_info, ready
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[int]:
        """Find relevant chunks using semantic similarity"""
        if self.chunk_embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.embedder.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return top_indices
    
    def answer_question(self, question: str, history: List[Tuple[str, str]]):
        """Answer using semantic search + generation"""
        
        if not self.chunks:
            return history + [(question, "📚 Please upload documents first")]
        
        if not question.strip():
            return history
        
        try:
            # SEMANTIC SEARCH (not keyword matching!)
            relevant_indices = self.semantic_search(question, top_k=3)
            
            if len(relevant_indices) == 0:
                return history + [(question, "❌ No relevant content found")]
            
            # Get relevant chunks
            context_parts = []
            sources = set()
            
            for idx in relevant_indices:
                context_parts.append(self.chunks[idx])
                sources.add(self.chunk_metadata[idx]['source'])
            
            context = "\n\n".join(context_parts)
            
            # Generate answer
            if "summar" in question.lower():
                # Summarization prompt
                prompt = f"Summarize this text:\n\n{context[:2000]}"
            else:
                # Q&A prompt
                prompt = f"""Answer the question based on the context.

Context: {context[:1500]}

Question: {question}

Answer:"""
            
            # Generate response
            result = self.generator(
                prompt,
                max_length=200,
                temperature=0.7,
                do_sample=True
            )
            
            if isinstance(result, list):
                answer = result[0]['generated_text']
            else:
                answer = result['generated_text']
            
            # Format response
            response = f"💡 {answer}"
            
            if sources:
                response += f"\n\n📄 Sources: {', '.join(sources)}"
            
        except Exception as e:
            print(f"Error: {e}")
            response = f"❌ Error: {str(e)}"
        
        return history + [(question, response)]
    
    def test_search(self, query: str) -> str:
        """Test semantic search to see what's retrieved"""
        if not self.chunks:
            return "No documents loaded"
        
        indices = self.semantic_search(query, top_k=5)
        
        output = [f"🔍 Semantic Search Results for: '{query}'\n"]
        output.append("="*50 + "\n")
        
        for i, idx in enumerate(indices, 1):
            chunk = self.chunks[idx]
            meta = self.chunk_metadata[idx]
            
            output.append(f"\n#{i} Match (from {meta['source']}):")
            output.append(f"Type: {meta['type']}")
            output.append(f"Preview: {chunk[:200]}...")
            output.append("-"*30)
        
        return "\n".join(output)

def create_ui():
    """Create the UI"""
    
    rag = WorkingRAG()
    
    with gr.Blocks(title="Working RAG", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🎯 Working RAG System
        ### Free, Local, with Semantic Search (not keyword matching!)
        """)
        
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(height=400)
                
                msg = gr.Textbox(
                    placeholder="Ask questions or 'summarize'...",
                    label="Question"
                )
                
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                gr.Examples(
                    examples=[
                        "Summarize my experience",
                        "What are my key skills?",
                        "What is my education?",
                        "Describe my background"
                    ],
                    inputs=msg
                )
            
            # Documents Tab
            with gr.Tab("📁 Documents"):
                file_upload = gr.File(
                    label="Upload Documents",
                    file_count="multiple"
                )
                
                upload_btn = gr.Button("📥 Process Documents", variant="primary")
                
                status = gr.Textbox(label="Status")
                file_info = gr.Dataframe(
                    headers=["File", "Size", "Chunks"],
                    label="Loaded Files"
                )
                ready = gr.Textbox(label="System Status")
            
            # Debug Tab
            with gr.Tab("🔍 Test Search"):
                test_query = gr.Textbox(
                    label="Test Query",
                    placeholder="Enter text to test semantic search..."
                )
                test_btn = gr.Button("Test Search")
                test_output = gr.Textbox(
                    label="Search Results",
                    lines=15
                )
            
            # Info Tab
            with gr.Tab("ℹ️ Info"):
                gr.Markdown(f"""
                ## How This Works
                
                ### 🧠 Semantic Search (The Key!)
                - Uses **sentence embeddings** to understand meaning
                - Finds relevant content by **meaning**, not just keywords
                - "Software engineer" matches "programmer", "developer", "coder"
                
                ### 📝 Smart Chunking
                - Markdown aware (preserves structure)
                - Paragraph-based for text files
                - Optimal chunk sizes for context
                
                ### 🤖 Generation Model
                - Using: **{rag.model_name}**
                - Designed for Q&A and summarization
                - Runs locally, no API needed
                
                ### Why This Works Better
                1. **Semantic search** > keyword matching
                2. **Smart chunking** preserves document structure
                3. **Specialized model** for Q&A tasks
                4. **Vector embeddings** understand meaning
                
                ### Tips
                - Upload your resume/documents
                - Ask natural questions
                - Test the search to see what's being retrieved
                """)
        
        # Event handlers
        def send_message(msg, history):
            return "", rag.answer_question(msg, history)
        
        msg.submit(send_message, [msg, chatbot], [msg, chatbot])
        send_btn.click(send_message, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: [], outputs=[chatbot])
        
        upload_btn.click(
            rag.load_documents,
            inputs=[file_upload],
            outputs=[status, file_info, ready]
        )
        
        test_btn.click(
            rag.test_search,
            inputs=[test_query],
            outputs=[test_output]
        )
    
    return app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 WORKING RAG SYSTEM")
    print("="*60)
    print("Features:")
    print("  ✅ Semantic search (understands meaning)")
    print("  ✅ Smart document chunking")
    print("  ✅ Better generation model")
    print("  ✅ Actually works with resumes!")
    print("="*60 + "\n")
    
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)