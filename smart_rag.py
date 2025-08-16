#!/usr/bin/env python3
"""
Smart RAG with Better Summarization
Uses BART for summarization and T5 for Q&A - much better results!
"""

import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from pathlib import Path
from typing import List, Tuple
import re

class SmartRAG:
    def __init__(self):
        """Initialize with better models for specific tasks"""
        print("\n🚀 Loading Smart AI Models...")
        print("📥 First run will download models (one-time)\n")
        
        try:
            # Use BART for summarization - MUCH better than GPT-Neo for this
            print("Loading BART summarization model...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",  # Excellent for summarization
                device=-1  # CPU
            )
            
            # Use T5 for Q&A - designed for question answering
            print("Loading T5 Q&A model...")
            self.qa_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",  # Great for Q&A, smaller than BART
                device=-1
            )
            
            print("✅ All models loaded successfully!\n")
            self.models_loaded = True
            
        except Exception as e:
            print(f"⚠️ Falling back to lighter models: {e}")
            # Fallback to smaller models
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",  # Smaller BART
                    device=-1
                )
                self.qa_pipeline = pipeline(
                    "text2text-generation", 
                    model="google/flan-t5-small",  # Smaller T5
                    device=-1
                )
                self.models_loaded = True
            except:
                print("❌ Could not load models")
                self.models_loaded = False
        
        self.documents = {}
        self.document_chunks = {}
        
    def process_document(self, content: str, filename: str) -> List[str]:
        """Smart document chunking for better processing"""
        # Clean the content
        content = content.strip()
        
        # Handle Markdown specially
        if filename.endswith('.md'):
            # Split by headers for better structure
            sections = re.split(r'\n#{1,6}\s+', content)
            chunks = []
            for section in sections:
                if section.strip():
                    # Keep chunks reasonable size
                    if len(section) > 1000:
                        # Split long sections into paragraphs
                        paragraphs = section.split('\n\n')
                        for para in paragraphs:
                            if para.strip():
                                chunks.append(para[:1000])
                    else:
                        chunks.append(section)
            return chunks if chunks else [content[:3000]]
        
        # For other files, split by paragraphs
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < 1000:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [content[:3000]]
    
    def load_documents(self, files):
        """Load and intelligently process documents"""
        if not files:
            return "No files uploaded", [], ""
        
        self.documents = {}
        self.document_chunks = {}
        loaded = []
        total_chunks = 0
        
        for file in files:
            try:
                with open(file.name, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                filename = Path(file.name).name
                self.documents[filename] = content
                
                # Process into smart chunks
                chunks = self.process_document(content, filename)
                self.document_chunks[filename] = chunks
                total_chunks += len(chunks)
                
                loaded.append(filename)
                print(f"✅ Loaded {filename}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        
        if loaded:
            status = f"✅ Loaded {len(loaded)} files ({total_chunks} chunks ready)"
            file_info = [[f"📄 {f}", f"{len(self.documents[f]):,} chars", f"{len(self.document_chunks[f])} chunks"] 
                        for f in loaded]
            return status, file_info, "Documents processed and ready!"
        
        return "❌ No files loaded", [], ""
    
    def summarize_document(self, filename: str = None) -> str:
        """Summarize a document or all documents"""
        if not self.documents:
            return "Please upload documents first"
        
        if not self.models_loaded:
            return "Models are still loading..."
        
        # Get content to summarize
        if filename and filename in self.documents:
            content = self.documents[filename]
            chunks = self.document_chunks[filename]
        else:
            # Summarize all documents
            content = "\n\n".join(self.documents.values())
            chunks = []
            for doc_chunks in self.document_chunks.values():
                chunks.extend(doc_chunks[:3])  # First 3 chunks from each doc
        
        try:
            # For long documents, summarize in chunks
            if len(chunks) > 1:
                summaries = []
                for i, chunk in enumerate(chunks[:5]):  # Max 5 chunks
                    if len(chunk) > 100:
                        summary = self.summarizer(
                            chunk,
                            max_length=150,
                            min_length=30,
                            do_sample=False
                        )[0]['summary_text']
                        summaries.append(summary)
                
                # Combine summaries
                combined = " ".join(summaries)
                
                # If still too long, summarize the summaries
                if len(combined) > 500:
                    final_summary = self.summarizer(
                        combined,
                        max_length=200,
                        min_length=50,
                        do_sample=False
                    )[0]['summary_text']
                    return final_summary
                return combined
            else:
                # Single chunk - direct summarization
                result = self.summarizer(
                    content[:1024],  # BART has token limits
                    max_length=200,
                    min_length=50,
                    do_sample=False
                )
                return result[0]['summary_text']
                
        except Exception as e:
            print(f"Summarization error: {e}")
            # Fallback to simple extraction
            lines = content.split('\n')
            important_lines = [l for l in lines if len(l) > 20][:5]
            return "Summary: " + " ".join(important_lines)
    
    def answer_question(self, question: str, history: List[Tuple[str, str]]):
        """Answer questions using T5"""
        if not self.documents:
            return history + [(question, "📚 Please upload documents first")]
        
        if not question.strip():
            return history
        
        if not self.models_loaded:
            return history + [(question, "⏳ Models loading...")]
        
        # Special handling for summarization requests
        if any(word in question.lower() for word in ['summarize', 'summary', 'tldr', 'overview']):
            summary = self.summarize_document()
            return history + [(question, f"📝 Summary:\n\n{summary}")]
        
        # Find relevant context
        relevant_text = ""
        question_words = set(question.lower().split())
        
        # Score each chunk by relevance
        scored_chunks = []
        for filename, chunks in self.document_chunks.items():
            for chunk in chunks:
                chunk_words = set(chunk.lower().split())
                # Simple relevance scoring
                score = len(question_words.intersection(chunk_words))
                if score > 0:
                    scored_chunks.append((score, chunk, filename))
        
        # Get top relevant chunks
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        relevant_chunks = scored_chunks[:3]
        
        if relevant_chunks:
            context = "\n\n".join([chunk for _, chunk, _ in relevant_chunks])
            source_files = list(set([filename for _, _, filename in relevant_chunks]))
        else:
            # No relevant chunks found, use beginning of first document
            first_doc = list(self.documents.values())[0]
            context = first_doc[:1500]
            source_files = [list(self.documents.keys())[0]]
        
        try:
            # Create prompt for T5
            prompt = f"Answer this question based on the context. Context: {context[:1000]} Question: {question}"
            
            # Generate answer
            result = self.qa_pipeline(
                prompt,
                max_length=150,
                temperature=0.7,
                do_sample=True
            )
            
            answer = result[0]['generated_text']
            
            # Format response
            response = f"💡 {answer}"
            if source_files:
                response += f"\n\n📄 *Sources: {', '.join(source_files)}*"
            
        except Exception as e:
            print(f"Q&A error: {e}")
            # Fallback to keyword search
            response = self.simple_search(question)
        
        return history + [(question, response)]
    
    def simple_search(self, query: str) -> str:
        """Fallback keyword search"""
        query_lower = query.lower()
        results = []
        
        for filename, content in self.documents.items():
            lines = content.split('\n')
            for line in lines:
                if query_lower in line.lower() and len(line) > 20:
                    results.append(f"📄 {filename}: {line[:200]}")
                    if len(results) >= 3:
                        break
        
        return "\n\n".join(results) if results else "No matches found"

def create_ui():
    """Create a beautiful, functional UI"""
    
    rag = SmartRAG()
    
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
        background-attachment: fixed;
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
    }
    """
    
    with gr.Blocks(title="Smart RAG Assistant", theme=gr.themes.Soft()) as app:
        
        gr.HTML("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; margin-bottom: 1rem;">
            <h1 style="color: white; margin: 0;">🧠 Smart Document Assistant</h1>
            <p style="color: white; opacity: 0.9;">Powered by BART & T5 - Specialized AI Models</p>
        </div>
        """)
        
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(height=400)
                
                msg = gr.Textbox(
                    placeholder="Ask questions or type 'summarize' for summary...",
                    label="Ask anything",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("Send 🚀", variant="primary")
                    clear_btn = gr.Button("Clear")
                    summarize_btn = gr.Button("📝 Summarize All", variant="secondary")
                
                gr.Examples(
                    examples=[
                        "Summarize this document",
                        "What are the main points?",
                        "What experience do I have?",
                        "What are the key skills mentioned?",
                        "Give me a brief overview"
                    ],
                    inputs=msg
                )
            
            # Documents Tab  
            with gr.Tab("📁 Documents"):
                file_upload = gr.File(
                    label="Upload Files (MD, TXT, etc.)",
                    file_count="multiple",
                    file_types=[".txt", ".md", ".csv", ".json"]
                )
                
                upload_btn = gr.Button("📥 Process Documents", variant="primary", size="lg")
                
                upload_status = gr.Textbox(label="Status")
                file_table = gr.Dataframe(
                    headers=["File", "Size", "Chunks"],
                    label="Processed Documents"
                )
                doc_status = gr.Textbox(label="Ready Status")
            
            # Info Tab
            with gr.Tab("ℹ️ Info"):
                gr.Markdown("""
                ## Smart Models Used:
                
                ### 📝 BART (Summarization)
                - **Model**: facebook/bart-large-cnn
                - **Task**: Document summarization
                - **Quality**: State-of-the-art summaries
                
                ### 🤔 T5 (Q&A)
                - **Model**: google/flan-t5-base  
                - **Task**: Question answering
                - **Quality**: Excellent comprehension
                
                ### Why It's Better:
                - **BART** is specifically trained for summarization (unlike GPT-Neo)
                - **T5** is designed for Q&A tasks
                - **Smart chunking** preserves document structure
                - **Markdown aware** - handles MD files properly
                
                ### Tips:
                - Type "summarize" to get document summary
                - Ask specific questions about content
                - Upload MD/TXT files for best results
                """)
        
        # Event handlers
        def send_message(msg, history):
            return "", rag.answer_question(msg, history)
        
        def summarize_all(history):
            summary = rag.summarize_document()
            return history + [("Please summarize all documents", f"📝 Summary:\n\n{summary}")]
        
        msg.submit(send_message, [msg, chatbot], [msg, chatbot])
        send_btn.click(send_message, [msg, chatbot], [msg, chatbot])
        summarize_btn.click(summarize_all, [chatbot], [chatbot])
        clear_btn.click(lambda: [], outputs=[chatbot])
        
        upload_btn.click(
            rag.load_documents,
            inputs=[file_upload],
            outputs=[upload_status, file_table, doc_status]
        )
    
    return app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 Smart RAG with BART & T5")
    print("="*60)
    print("Models:")
    print("  • BART for summarization")
    print("  • T5 for Q&A")
    print("="*60 + "\n")
    
    app = create_ui()
    app.launch(server_port=7860, inbrowser=True)