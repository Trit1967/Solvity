#!/usr/bin/env python3
"""
Ultra Simple RAG - Works with just Gradio
No ML models, just text search
"""

import gradio as gr
import os
from typing import List

class UltraSimpleRAG:
    def __init__(self):
        self.documents = {}
        self.loaded_files = []
        
    def load_files(self, files):
        """Load text files into memory"""
        if not files:
            return "No files uploaded", ""
        
        self.documents = {}
        self.loaded_files = []
        
        for file in files:
            try:
                with open(file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                    filename = os.path.basename(file.name)
                    self.documents[filename] = content
                    self.loaded_files.append(filename)
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        
        if self.loaded_files:
            status = f"✅ Loaded {len(self.loaded_files)} files successfully"
            file_list = "\n".join(f"• {f}" for f in self.loaded_files)
            return status, file_list
        return "❌ No files could be loaded", ""
    
    def search(self, query: str) -> str:
        """Simple keyword search in documents"""
        if not self.documents:
            return "❌ Please upload some documents first"
        
        if not query:
            return "❌ Please enter a search query"
        
        query_lower = query.lower()
        results = []
        
        # Search each document
        for filename, content in self.documents.items():
            lines = content.split('\n')
            matches = []
            
            # Find lines containing the query
            for i, line in enumerate(lines, 1):
                if query_lower in line.lower():
                    # Get context (line before and after)
                    context_start = max(0, i-2)
                    context_end = min(len(lines), i+1)
                    context = lines[context_start:context_end]
                    matches.append({
                        'line': i,
                        'text': '\n'.join(context)
                    })
            
            if matches:
                results.append(f"📄 **{filename}** ({len(matches)} matches):")
                for match in matches[:3]:  # Show first 3 matches
                    results.append(f"  Line {match['line']}:")
                    results.append(f"  {match['text'][:200]}...")
                    results.append("")
        
        if results:
            return "\n".join(results)
        else:
            return f"No matches found for '{query}'"
    
    def get_summary(self) -> str:
        """Get summary of loaded documents"""
        if not self.documents:
            return "No documents loaded"
        
        summary = []
        for filename, content in self.documents.items():
            words = len(content.split())
            lines = len(content.split('\n'))
            summary.append(f"• {filename}: {words} words, {lines} lines")
        
        return "\n".join(summary)

def create_ui():
    rag = UltraSimpleRAG()
    
    with gr.Blocks(title="Ultra Simple RAG", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🔍 Ultra Simple Document Search
        ### Upload text files and search through them instantly
        """)
        
        with gr.Tab("📤 Upload"):
            file_input = gr.File(
                label="Upload text files",
                file_count="multiple",
                file_types=[".txt", ".md", ".log", ".csv"]
            )
            load_btn = gr.Button("Load Files", variant="primary")
            
            with gr.Row():
                status = gr.Textbox(label="Status", interactive=False)
                file_list = gr.Textbox(label="Loaded Files", lines=5, interactive=False)
        
        with gr.Tab("🔍 Search"):
            search_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter keywords to search...",
                lines=1
            )
            search_btn = gr.Button("Search", variant="primary")
            search_results = gr.Markdown(label="Results")
            
        with gr.Tab("📊 Summary"):
            summary_btn = gr.Button("Get Summary", variant="primary")
            summary_text = gr.Textbox(label="Document Summary", lines=10, interactive=False)
        
        # Event handlers
        load_btn.click(
            fn=rag.load_files,
            inputs=file_input,
            outputs=[status, file_list]
        )
        
        search_btn.click(
            fn=rag.search,
            inputs=search_input,
            outputs=search_results
        )
        
        search_input.submit(
            fn=rag.search,
            inputs=search_input,
            outputs=search_results
        )
        
        summary_btn.click(
            fn=rag.get_summary,
            outputs=summary_text
        )
        
        gr.Markdown("""
        ---
        ### How to use:
        1. Upload text files in the Upload tab
        2. Search for keywords in the Search tab
        3. View document statistics in Summary tab
        
        This is a simple keyword search - no AI needed!
        """)
    
    return app

if __name__ == "__main__":
    print("🚀 Starting Ultra Simple RAG...")
    print("📝 This version uses simple text search - no AI models needed")
    print("🌐 Opening browser at http://localhost:7860")
    app = create_ui()
    app.launch(inbrowser=True)