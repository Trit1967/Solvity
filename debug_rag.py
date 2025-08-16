#!/usr/bin/env python3
"""
Debug RAG - See EXACTLY what the AI sees
Shows document extraction, chunking, and what gets sent to the model
"""

import gradio as gr
from transformers import pipeline
import torch
import os
from pathlib import Path
from typing import List, Dict
import json
import re

class DebugRAG:
    def __init__(self):
        """Initialize with simple model for testing"""
        print("\n🔍 Debug RAG - See what the AI actually receives\n")
        
        try:
            # Use a simple model for testing
            print("Loading simple Q&A model for debugging...")
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=-1
            )
            self.model_loaded = True
        except:
            self.model_loaded = False
            print("❌ Model loading failed - will show extraction only")
        
        self.raw_documents = {}
        self.processed_chunks = {}
        self.debug_info = {}
    
    def extract_and_chunk(self, content: str, filename: str) -> Dict:
        """Extract and chunk with full debugging info"""
        debug_data = {
            "filename": filename,
            "original_length": len(content),
            "first_500_chars": content[:500],
            "extraction_method": "",
            "chunks": [],
            "issues": []
        }
        
        # Check file type and content
        if filename.endswith('.md'):
            debug_data["extraction_method"] = "Markdown Parser"
            
            # Show raw markdown structure
            debug_data["markdown_structure"] = {
                "headers": len(re.findall(r'^#{1,6}\s+.*$', content, re.MULTILINE)),
                "lists": len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE)),
                "code_blocks": len(re.findall(r'```[\s\S]*?```', content)),
                "links": len(re.findall(r'\[.*?\]\(.*?\)', content))
            }
            
            # Extract sections by headers
            sections = re.split(r'\n(?=#{1,6}\s+)', content)
            
            for i, section in enumerate(sections[:10]):  # First 10 sections
                chunk_info = {
                    "chunk_id": i,
                    "length": len(section),
                    "preview": section[:200] + "..." if len(section) > 200 else section,
                    "has_content": bool(section.strip())
                }
                
                # Check for common issues
                if not section.strip():
                    chunk_info["issue"] = "Empty section"
                elif len(section) < 20:
                    chunk_info["issue"] = "Very short section"
                elif "�" in section:
                    chunk_info["issue"] = "Encoding issues detected"
                
                debug_data["chunks"].append(chunk_info)
        
        elif filename.endswith(('.txt', '.text')):
            debug_data["extraction_method"] = "Plain Text Parser"
            
            # Split by paragraphs
            paragraphs = content.split('\n\n')
            
            for i, para in enumerate(paragraphs[:10]):
                if para.strip():
                    chunk_info = {
                        "chunk_id": i,
                        "length": len(para),
                        "preview": para[:200] + "..." if len(para) > 200 else para,
                        "lines": len(para.split('\n'))
                    }
                    debug_data["chunks"].append(chunk_info)
        
        else:
            debug_data["extraction_method"] = "Generic Parser"
            debug_data["issues"].append(f"Unknown file type: {filename}")
            
            # Simple line-based extraction
            lines = content.split('\n')
            current_chunk = ""
            chunk_id = 0
            
            for line in lines:
                if len(current_chunk) + len(line) < 500:
                    current_chunk += line + "\n"
                else:
                    if current_chunk.strip():
                        debug_data["chunks"].append({
                            "chunk_id": chunk_id,
                            "length": len(current_chunk),
                            "preview": current_chunk[:200] + "..."
                        })
                        chunk_id += 1
                    current_chunk = line + "\n"
        
        # Check for common problems
        if not debug_data["chunks"]:
            debug_data["issues"].append("No chunks extracted!")
        
        if len(content) > 0 and len(debug_data["chunks"]) == 0:
            debug_data["issues"].append("Content exists but no chunks created")
        
        # Check encoding
        if content.count('�') > 0:
            debug_data["issues"].append(f"Found {content.count('�')} encoding errors")
        
        return debug_data
    
    def load_and_debug(self, files):
        """Load documents with full debugging output"""
        if not files:
            return "No files", "", {}
        
        debug_output = []
        all_debug_info = {}
        
        for file in files:
            try:
                # Try different encodings
                content = None
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                
                for encoding in encodings:
                    try:
                        with open(file.name, 'r', encoding=encoding) as f:
                            content = f.read()
                            debug_output.append(f"✅ Read {file.name} with {encoding} encoding")
                            break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    # Try binary mode as last resort
                    with open(file.name, 'rb') as f:
                        raw_bytes = f.read()
                        content = raw_bytes.decode('utf-8', errors='ignore')
                        debug_output.append(f"⚠️ Read {file.name} in binary mode with errors ignored")
                
                filename = Path(file.name).name
                
                # Store raw content
                self.raw_documents[filename] = content
                
                # Extract and debug
                debug_info = self.extract_and_chunk(content, filename)
                self.debug_info[filename] = debug_info
                all_debug_info[filename] = debug_info
                
                # Create summary
                debug_output.append(f"\n📄 {filename}:")
                debug_output.append(f"  - Size: {len(content):,} chars")
                debug_output.append(f"  - Method: {debug_info['extraction_method']}")
                debug_output.append(f"  - Chunks: {len(debug_info['chunks'])}")
                
                if debug_info.get('issues'):
                    debug_output.append(f"  - ⚠️ Issues: {', '.join(debug_info['issues'])}")
                
                # Store processed chunks for AI
                self.processed_chunks[filename] = [
                    chunk['preview'] for chunk in debug_info['chunks']
                ]
                
            except Exception as e:
                debug_output.append(f"❌ Error loading {file.name}: {str(e)}")
                all_debug_info[file.name] = {"error": str(e)}
        
        debug_text = "\n".join(debug_output)
        
        # Create detailed JSON view
        json_view = json.dumps(all_debug_info, indent=2)
        
        return "Files loaded - see debug info", debug_text, json_view
    
    def test_extraction(self, question: str) -> str:
        """Show exactly what gets sent to the AI"""
        if not self.processed_chunks:
            return "No documents loaded"
        
        output = []
        output.append("🔍 DEBUG: What the AI receives:\n")
        output.append("="*50 + "\n")
        
        # Show the question
        output.append(f"QUESTION: {question}\n")
        output.append("-"*50 + "\n")
        
        # Find relevant chunks (simple keyword matching)
        question_words = set(question.lower().split())
        relevant_chunks = []
        
        for filename, chunks in self.processed_chunks.items():
            for i, chunk in enumerate(chunks):
                chunk_words = set(chunk.lower().split())
                overlap = len(question_words.intersection(chunk_words))
                if overlap > 0:
                    relevant_chunks.append({
                        "file": filename,
                        "chunk_id": i,
                        "relevance_score": overlap,
                        "content": chunk[:300]
                    })
        
        # Sort by relevance
        relevant_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Show what would be sent as context
        output.append("CONTEXT SENT TO AI (top 3 chunks):\n")
        output.append("-"*50 + "\n")
        
        if relevant_chunks:
            context_for_ai = ""
            for i, chunk in enumerate(relevant_chunks[:3], 1):
                output.append(f"Chunk {i} (from {chunk['file']}, relevance: {chunk['relevance_score']}):\n")
                output.append(f"{chunk['content']}\n")
                output.append("-"*30 + "\n")
                context_for_ai += chunk['content'] + " "
            
            # Try to get AI answer if model loaded
            if self.model_loaded and context_for_ai:
                output.append("\nAI RESPONSE:\n")
                output.append("-"*50 + "\n")
                try:
                    result = self.qa_pipeline(
                        question=question,
                        context=context_for_ai[:1000]
                    )
                    output.append(f"Answer: {result['answer']}")
                    output.append(f"Confidence: {result['score']:.2%}")
                except Exception as e:
                    output.append(f"Error: {e}")
        else:
            output.append("❌ NO RELEVANT CHUNKS FOUND!\n")
            output.append("This means the AI has no context to answer from.\n")
            output.append("\nAll available chunks:\n")
            for filename, chunks in self.processed_chunks.items():
                output.append(f"\n{filename}: {len(chunks)} chunks")
                if chunks:
                    output.append(f"  First chunk preview: {chunks[0][:100]}...")
        
        return "\n".join(output)
    
    def show_raw_content(self, filename: str) -> str:
        """Show raw document content"""
        if filename in self.raw_documents:
            content = self.raw_documents[filename]
            return f"📄 {filename} (Raw Content):\n\n{content[:5000]}\n\n... (showing first 5000 chars)"
        return "File not found"

def create_debug_ui():
    """Create debug interface"""
    
    rag = DebugRAG()
    
    with gr.Blocks(title="RAG Debugger", theme=gr.themes.Base()) as app:
        
        gr.Markdown("""
        # 🔍 RAG Document Extraction Debugger
        ### See EXACTLY what the AI receives from your documents
        """)
        
        with gr.Tabs():
            # Load & Debug Tab
            with gr.Tab("📥 Load & Debug"):
                file_upload = gr.File(
                    label="Upload documents to debug",
                    file_count="multiple"
                )
                
                load_btn = gr.Button("🔍 Load & Analyze", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        status = gr.Textbox(label="Status", lines=1)
                        debug_output = gr.Textbox(
                            label="Extraction Summary",
                            lines=15
                        )
                    
                    with gr.Column():
                        json_view = gr.Code(
                            label="Detailed Debug Info (JSON)",
                            language="json",
                            lines=15
                        )
            
            # Test Extraction Tab
            with gr.Tab("🧪 Test Extraction"):
                test_question = gr.Textbox(
                    label="Test Question",
                    placeholder="Enter a question to see what context the AI gets...",
                    lines=2
                )
                
                test_btn = gr.Button("🔍 Show AI Context", variant="primary")
                
                extraction_output = gr.Code(
                    label="Debug Output - What AI Sees",
                    language="text",
                    lines=20
                )
            
            # Raw Content Tab
            with gr.Tab("📄 Raw Content"):
                filename_select = gr.Dropdown(
                    label="Select file to view",
                    choices=[]
                )
                
                view_btn = gr.Button("View Raw Content")
                
                raw_content = gr.Textbox(
                    label="Raw Document Content",
                    lines=20
                )
            
            # Help Tab
            with gr.Tab("❓ Help"):
                gr.Markdown("""
                ## What This Shows You:
                
                ### 1. **Extraction Summary**
                - How your document was parsed
                - Number of chunks created
                - Any encoding issues
                - File reading problems
                
                ### 2. **Debug JSON**
                - Detailed chunk information
                - Character counts
                - Preview of each chunk
                - Extraction method used
                
                ### 3. **Test Extraction**
                - Enter a question
                - See which chunks are found as "relevant"
                - See the EXACT context sent to AI
                - Understand why AI might fail
                
                ### Common Issues:
                
                #### ❌ "No relevant chunks found"
                - Your question keywords don't match document content
                - Document wasn't chunked properly
                - Encoding issues corrupted the text
                
                #### ❌ "Empty chunks"
                - Document has unusual formatting
                - Wrong parser for file type
                - Encoding problems
                
                #### ❌ "Wrong context retrieved"
                - Simple keyword matching isn't finding right sections
                - Need better embedding-based retrieval (vector search)
                
                ### Solutions:
                1. Check if chunks contain your actual content
                2. Try different question wording
                3. Make sure document is plain text or proper markdown
                4. Check for encoding issues (� characters)
                """)
        
        # Event handlers
        def on_load(files):
            status, debug, json_data = rag.load_and_debug(files)
            # Update filename dropdown
            if rag.raw_documents:
                return status, debug, json_data, gr.update(choices=list(rag.raw_documents.keys()))
            return status, debug, json_data, gr.update()
        
        load_btn.click(
            on_load,
            inputs=[file_upload],
            outputs=[status, debug_output, json_view, filename_select]
        )
        
        test_btn.click(
            rag.test_extraction,
            inputs=[test_question],
            outputs=[extraction_output]
        )
        
        view_btn.click(
            rag.show_raw_content,
            inputs=[filename_select],
            outputs=[raw_content]
        )
    
    return app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔍 RAG DEBUGGER")
    print("="*60)
    print("This tool shows you:")
    print("  1. How documents are extracted")
    print("  2. How they're chunked")
    print("  3. What context the AI actually receives")
    print("  4. Why the AI might be failing")
    print("="*60 + "\n")
    
    app = create_debug_ui()
    app.launch(server_port=7860, inbrowser=True)