#!/usr/bin/env python3
"""
Simplified RAG Chatbot - Minimal dependencies version
Works with just transformers and gradio
"""

import gradio as gr
from transformers import pipeline
import os

# Simple RAG class without heavy dependencies
class SimpleRAG:
    def __init__(self):
        print("Loading model... (this takes a minute first time)")
        # Use a small model that works well
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=-1  # CPU
        )
        self.context = ""
        
    def load_documents(self, files):
        """Load text from uploaded files"""
        if not files:
            return "No files uploaded"
        
        self.context = ""
        loaded_files = []
        
        for file in files:
            try:
                with open(file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.context += f"\n\n--- {os.path.basename(file.name)} ---\n{content}"
                    loaded_files.append(os.path.basename(file.name))
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        
        if loaded_files:
            return f"✅ Loaded {len(loaded_files)} files: {', '.join(loaded_files)}"
        return "❌ No files could be loaded"
    
    def answer(self, question):
        """Answer question based on loaded context"""
        if not self.context:
            return "Please upload and load documents first"
        
        try:
            # Simple approach: use first 2000 chars of context
            # (DistilBERT has token limits)
            truncated_context = self.context[:2000]
            
            result = self.qa_pipeline(
                question=question,
                context=truncated_context
            )
            
            confidence = f" (confidence: {result['score']:.2%})" if result['score'] > 0.1 else ""
            return result['answer'] + confidence
            
        except Exception as e:
            return f"Error: {str(e)}"

# Create the UI
def create_ui():
    rag = SimpleRAG()
    
    with gr.Blocks(title="Simple RAG") as app:
        gr.Markdown("# 🤖 Simple Local RAG Chatbot")
        gr.Markdown("Upload text files and ask questions about them")
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Upload text files",
                    file_count="multiple",
                    file_types=[".txt", ".md"]
                )
                load_btn = gr.Button("Load Documents", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
                
            with gr.Column():
                question = gr.Textbox(
                    label="Ask a question",
                    placeholder="What is this document about?"
                )
                answer = gr.Textbox(label="Answer", lines=3)
                ask_btn = gr.Button("Get Answer", variant="primary")
        
        # Connect functions
        load_btn.click(
            fn=rag.load_documents,
            inputs=file_input,
            outputs=status
        )
        
        ask_btn.click(
            fn=rag.answer,
            inputs=question,
            outputs=answer
        )
        
        question.submit(
            fn=rag.answer,
            inputs=question,
            outputs=answer
        )
    
    return app

if __name__ == "__main__":
    print("Starting Simple RAG UI...")
    app = create_ui()
    app.launch(inbrowser=True)