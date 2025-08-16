#!/usr/bin/env python3
"""
ADMIN DASHBOARD - Full Control & Monitoring
All configuration, debugging, and optimization tools
"""

import gradio as gr
import requests
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
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
import yaml
import os
from datetime import datetime
from collections import OrderedDict

class AdminRAG:
    def __init__(self):
        # Full configuration with all options
        self.config = {
            # Model settings
            'model': 'llama3.2',
            'base_url': 'http://localhost:11434',
            'temperature': 0.7,
            'max_tokens': 2000,
            
            # Search settings  
            'similarity_threshold': 0.12,
            'top_k_results': 5,
            'use_reranking': False,
            
            # Chunking settings
            'chunk_size': 1500,
            'chunk_overlap': 200,
            'min_chunk_size': 50,
            'smart_chunking': True,
            'preserve_sentences': True,
            
            # Context settings
            'max_context_length': 4000,
            'context_window_type': 'sliding',  # or 'fixed'
            
            # Features
            'enable_resume_mode': True,
            'enable_caching': True,
            'cache_ttl': 3600,
            'debug_mode': True,
            'log_queries': True,
            
            # Performance
            'batch_size': 32,
            'use_gpu': False,
            'num_threads': 4
        }
        
        # Initialize embedder
        print("🚀 Initializing Admin RAG System...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedder ready")
        
        # Storage
        self.documents = {}
        self.chunks = []
        self.chunk_embeddings = None
        self.chunk_metadata = []
        
        # Caching layers
        self.cache = {
            'queries': OrderedDict(),  # LRU cache
            'embeddings': {},
            'chunks': {},
            'extractions': {}
        }
        self.max_cache_size = 1000
        
        # Statistics & Monitoring
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0,
            'total_documents': 0,
            'total_chunks': 0,
            'errors': [],
            'query_log': []
        }
        
        # Load saved configuration if exists
        self.load_config()
        
    def save_config(self, filename='admin_config.yaml'):
        """Save configuration to file"""
        try:
            with open(filename, 'w') as f:
                yaml.dump(self.config, f)
            return f"✅ Configuration saved to {filename}"
        except Exception as e:
            return f"❌ Error saving config: {e}"
    
    def load_config(self, filename='admin_config.yaml'):
        """Load configuration from file"""
        try:
            if Path(filename).exists():
                with open(filename, 'r') as f:
                    loaded = yaml.load(f, Loader=yaml.FullLoader)
                    self.config.update(loaded)
                return f"✅ Configuration loaded from {filename}"
            return "No saved configuration found"
        except Exception as e:
            return f"❌ Error loading config: {e}"
    
    def export_config(self):
        """Export configuration as JSON"""
        return json.dumps(self.config, indent=2)
    
    def import_config(self, config_json: str):
        """Import configuration from JSON"""
        try:
            new_config = json.loads(config_json)
            self.config.update(new_config)
            return "✅ Configuration imported successfully"
        except Exception as e:
            return f"❌ Error importing config: {e}"
    
    def update_config(self, **kwargs):
        """Update configuration values"""
        for key, value in kwargs.items():
            if key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                
                # Log significant changes
                if old_value != value and self.config['log_queries']:
                    self.stats['query_log'].append({
                        'type': 'config_change',
                        'key': key,
                        'old': old_value,
                        'new': value,
                        'time': datetime.now().isoformat()
                    })
        
        return self.config
    
    def smart_chunk(self, text: str, filename: str) -> List[Dict]:
        """Advanced chunking with multiple strategies"""
        chunks = []
        
        # Check cache first
        doc_hash = hashlib.md5((text + filename).encode()).hexdigest()
        if self.config['enable_caching'] and doc_hash in self.cache['chunks']:
            return self.cache['chunks'][doc_hash]
        
        # Strategy based on file type and content
        if self.config['smart_chunking']:
            # Resume/CV special handling
            if self.config['enable_resume_mode'] and any(kw in filename.lower() for kw in ['resume', 'cv', 'bio']):
                # Keep full overview
                overview_size = min(3000, len(text))
                chunks.append({
                    'text': text[:overview_size],
                    'source': filename,
                    'type': 'resume_overview',
                    'size': overview_size,
                    'strategy': 'full_context'
                })
            
            # Markdown structure preservation
            if filename.endswith('.md'):
                sections = re.split(r'\n(?=#{1,6}\s+)', text)
                for section in sections:
                    if len(section.strip()) > self.config['min_chunk_size']:
                        chunks.append({
                            'text': section.strip(),
                            'source': filename,
                            'type': 'markdown_section',
                            'size': len(section.strip()),
                            'strategy': 'structure_based'
                        })
        
        # Regular chunking with overlap
        if not chunks or self.config['smart_chunking'] is False:
            if self.config['preserve_sentences']:
                # Sentence-aware chunking
                sentences = re.split(r'(?<=[.!?])\s+', text)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < self.config['chunk_size']:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append({
                                'text': current_chunk.strip(),
                                'source': filename,
                                'type': 'sentence_based',
                                'size': len(current_chunk.strip()),
                                'strategy': 'sentence_preserve'
                            })
                        current_chunk = sentence + " "
                
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'source': filename,
                        'type': 'sentence_based',
                        'size': len(current_chunk.strip()),
                        'strategy': 'sentence_preserve'
                    })
            else:
                # Word-based chunking
                words = text.split()
                chunk_words = self.config['chunk_size'] // 5
                overlap_words = self.config['chunk_overlap'] // 5
                
                for i in range(0, len(words), chunk_words - overlap_words):
                    chunk_text = ' '.join(words[i:i + chunk_words])
                    if len(chunk_text) > self.config['min_chunk_size']:
                        chunks.append({
                            'text': chunk_text,
                            'source': filename,
                            'type': 'word_based',
                            'size': len(chunk_text),
                            'strategy': 'fixed_size'
                        })
        
        # Cache chunks
        if self.config['enable_caching'] and chunks:
            self.cache['chunks'][doc_hash] = chunks
        
        return chunks if chunks else [{'text': text[:2000], 'source': filename, 'type': 'fallback', 'size': len(text[:2000]), 'strategy': 'fallback'}]
    
    def extract_text(self, file_path: str) -> str:
        """Extract text with caching"""
        # Check cache
        if self.config['enable_caching']:
            file_mtime = os.path.getmtime(file_path)
            cache_key = f"{file_path}_{file_mtime}"
            if cache_key in self.cache['extractions']:
                return self.cache['extractions'][cache_key]
        
        filename = Path(file_path).name.lower()
        text = ""
        
        try:
            if filename.endswith('.pdf'):
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
            
            elif filename.endswith('.docx'):
                doc = Document(file_path)
                text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            
            elif filename.endswith(('.csv', '.xlsx', '.xls')):
                df = pd.read_excel(file_path) if filename.endswith(('.xlsx', '.xls')) else pd.read_csv(file_path)
                text = f"Data: {df.shape[0]} rows, {df.shape[1]} columns\n"
                text += f"Columns: {', '.join(df.columns)}\n\n"
                text += df.to_string()
            
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            # Cache extraction
            if self.config['enable_caching'] and text:
                self.cache['extractions'][cache_key] = text
            
        except Exception as e:
            self.stats['errors'].append({
                'type': 'extraction_error',
                'file': filename,
                'error': str(e),
                'time': datetime.now().isoformat()
            })
            text = f"[Extraction failed: {e}]"
        
        return text.strip()
    
    def load_documents(self, files):
        """Load documents with detailed statistics"""
        if not files:
            return "No files selected", {}, ""
        
        start_time = time.time()
        self.documents = {}
        self.chunks = []
        self.chunk_metadata = []
        
        stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_chunks': 0,
            'total_chars': 0,
            'chunk_strategies': {},
            'processing_time': 0
        }
        
        details = []
        
        for file in files:
            try:
                filename = Path(file.name).name
                
                # Extract text
                content = self.extract_text(file.name)
                
                if content and len(content) > 10:
                    self.documents[filename] = content
                    stats['total_chars'] += len(content)
                    
                    # Create chunks
                    file_chunks = self.smart_chunk(content, filename)
                    stats['total_chunks'] += len(file_chunks)
                    
                    # Track strategies used
                    for chunk in file_chunks:
                        strategy = chunk.get('strategy', 'unknown')
                        stats['chunk_strategies'][strategy] = stats['chunk_strategies'].get(strategy, 0) + 1
                        
                        self.chunks.append(chunk['text'])
                        self.chunk_metadata.append({
                            'source': chunk['source'],
                            'type': chunk['type'],
                            'size': chunk['size'],
                            'strategy': chunk['strategy']
                        })
                    
                    stats['files_processed'] += 1
                    details.append(f"✅ {filename}: {len(file_chunks)} chunks, {len(content):,} chars")
                else:
                    stats['files_failed'] += 1
                    details.append(f"⚠️ {filename}: Empty or too small")
                    
            except Exception as e:
                stats['files_failed'] += 1
                details.append(f"❌ {filename}: {str(e)}")
                self.stats['errors'].append({
                    'type': 'load_error',
                    'file': filename,
                    'error': str(e),
                    'time': datetime.now().isoformat()
                })
        
        # Create embeddings
        if self.chunks:
            print(f"Creating embeddings for {len(self.chunks)} chunks...")
            
            # Check for cached embeddings
            if self.config['enable_caching']:
                doc_hash = hashlib.md5(str(self.chunks).encode()).hexdigest()
                if doc_hash in self.cache['embeddings']:
                    self.chunk_embeddings = self.cache['embeddings'][doc_hash]
                    print("Using cached embeddings")
                else:
                    self.chunk_embeddings = self.embedder.encode(
                        self.chunks,
                        batch_size=self.config['batch_size'],
                        show_progress_bar=True
                    )
                    self.cache['embeddings'][doc_hash] = self.chunk_embeddings
            else:
                self.chunk_embeddings = self.embedder.encode(
                    self.chunks,
                    batch_size=self.config['batch_size'],
                    show_progress_bar=True
                )
        
        stats['processing_time'] = time.time() - start_time
        
        # Update global stats
        self.stats['total_documents'] = len(self.documents)
        self.stats['total_chunks'] = len(self.chunks)
        
        status = f"Processed {stats['files_processed']} files ({stats['files_failed']} failed) in {stats['processing_time']:.2f}s"
        
        return status, stats, "\n".join(details)
    
    def search_with_debug(self, query: str) -> Tuple[List[Dict], Dict]:
        """Search with detailed debugging information"""
        if not self.chunks:
            return [], {'error': 'No documents loaded'}
        
        start_time = time.time()
        
        # Check cache
        cache_key = hashlib.md5(f"{query}_{self.config}".encode()).hexdigest()
        if self.config['enable_caching'] and cache_key in self.cache['queries']:
            self.stats['cache_hits'] += 1
            cached = self.cache['queries'][cache_key]
            if time.time() - cached['time'] < self.config['cache_ttl']:
                return cached['results'], cached['debug']
            else:
                del self.cache['queries'][cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Encode query
        query_embedding = self.embedder.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-self.config['top_k_results']:][::-1]
        
        # Build debug info
        debug = {
            'query': query,
            'query_length': len(query),
            'embedding_dim': query_embedding.shape[1],
            'total_chunks': len(self.chunks),
            'threshold': self.config['similarity_threshold'],
            'top_k': self.config['top_k_results'],
            'scores': [],
            'processing_time': 0
        }
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            metadata = self.chunk_metadata[idx]
            
            debug['scores'].append({
                'index': int(idx),
                'score': score,
                'above_threshold': score > self.config['similarity_threshold'],
                'source': metadata['source'],
                'type': metadata['type'],
                'size': metadata['size'],
                'strategy': metadata.get('strategy', 'unknown')
            })
            
            if score > self.config['similarity_threshold']:
                results.append({
                    'text': self.chunks[idx],
                    'score': score,
                    'metadata': metadata
                })
        
        # Fallback if no results
        if not results and len(top_indices) > 0:
            idx = top_indices[0]
            results.append({
                'text': self.chunks[idx],
                'score': float(similarities[idx]),
                'metadata': self.chunk_metadata[idx]
            })
            debug['fallback_used'] = True
        
        debug['processing_time'] = time.time() - start_time
        debug['results_returned'] = len(results)
        
        # Update stats
        self.stats['total_queries'] += 1
        response_times = [debug['processing_time']]
        if self.stats['avg_response_time'] > 0:
            response_times.append(self.stats['avg_response_time'])
        self.stats['avg_response_time'] = sum(response_times) / len(response_times)
        
        # Log query
        if self.config['log_queries']:
            self.stats['query_log'].append({
                'query': query,
                'results': len(results),
                'best_score': results[0]['score'] if results else 0,
                'time': datetime.now().isoformat(),
                'processing_ms': debug['processing_time'] * 1000
            })
        
        # Cache results
        if self.config['enable_caching']:
            self.cache['queries'][cache_key] = {
                'results': results,
                'debug': debug,
                'time': time.time()
            }
            
            # LRU eviction
            if len(self.cache['queries']) > self.max_cache_size:
                self.cache['queries'].popitem(last=False)
        
        return results, debug
    
    def test_query(self, query: str) -> str:
        """Test query with formatted output"""
        if not query:
            return "Enter a query to test"
        
        results, debug = self.search_with_debug(query)
        
        output = []
        output.append("🔍 QUERY TEST RESULTS")
        output.append("=" * 50)
        output.append(f"Query: '{query}'")
        output.append(f"Processing time: {debug.get('processing_time', 0)*1000:.2f}ms")
        output.append(f"Threshold: {self.config['similarity_threshold']}")
        output.append(f"Chunks searched: {debug.get('total_chunks', 0)}")
        
        if 'error' in debug:
            output.append(f"\n❌ Error: {debug['error']}")
        else:
            output.append(f"\n📊 Top {len(debug['scores'])} Matches:")
            output.append("-" * 40)
            
            for i, score_info in enumerate(debug['scores'], 1):
                status = "✅" if score_info['above_threshold'] else "❌"
                output.append(
                    f"{i}. {status} Score: {score_info['score']:.4f}\n"
                    f"   Source: {score_info['source']}\n"
                    f"   Type: {score_info['type']} | Size: {score_info['size']} | Strategy: {score_info['strategy']}"
                )
            
            if debug.get('fallback_used'):
                output.append("\n⚠️ No results above threshold - using best match")
            
            output.append(f"\n📋 Returned {debug['results_returned']} chunks for context")
            
            if results:
                output.append("\n🔍 Best Match Preview:")
                output.append(f"Score: {results[0]['score']:.4f}")
                output.append(f"Text: {results[0]['text'][:300]}...")
        
        return "\n".join(output)
    
    def get_system_status(self) -> str:
        """Get comprehensive system status"""
        status = []
        
        # Header
        status.append("🎛️ SYSTEM STATUS REPORT")
        status.append("=" * 50)
        status.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Ollama status
        status.append("\n📡 AI Model Status:")
        try:
            response = requests.get(f"{self.config['base_url']}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                status.append(f"✅ Ollama: Online")
                status.append(f"📦 Available models: {len(models)}")
                for model in models[:3]:
                    status.append(f"   - {model.get('name', 'unknown')}")
            else:
                status.append("⚠️ Ollama: Offline")
        except:
            status.append("❌ Ollama: Not reachable")
        
        # Document statistics
        status.append("\n📚 Document Statistics:")
        status.append(f"Documents loaded: {self.stats['total_documents']}")
        status.append(f"Total chunks: {self.stats['total_chunks']}")
        if self.chunk_metadata:
            avg_chunk_size = sum(m['size'] for m in self.chunk_metadata) / len(self.chunk_metadata)
            status.append(f"Average chunk size: {avg_chunk_size:.0f} chars")
            
            # Chunk strategies
            strategies = {}
            for m in self.chunk_metadata:
                s = m.get('strategy', 'unknown')
                strategies[s] = strategies.get(s, 0) + 1
            status.append("Chunking strategies used:")
            for strategy, count in strategies.items():
                status.append(f"   - {strategy}: {count} chunks")
        
        # Cache statistics
        status.append("\n💾 Cache Statistics:")
        status.append(f"Query cache: {len(self.cache['queries'])} entries")
        status.append(f"Embedding cache: {len(self.cache['embeddings'])} entries")
        status.append(f"Chunk cache: {len(self.cache['chunks'])} entries")
        status.append(f"Extraction cache: {len(self.cache['extractions'])} entries")
        
        if self.stats['total_queries'] > 0:
            hit_rate = (self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])) * 100
            status.append(f"Cache hit rate: {hit_rate:.1f}%")
        
        # Performance metrics
        status.append("\n⚡ Performance Metrics:")
        status.append(f"Total queries: {self.stats['total_queries']}")
        status.append(f"Avg response time: {self.stats['avg_response_time']*1000:.2f}ms")
        status.append(f"Cache hits: {self.stats['cache_hits']}")
        status.append(f"Cache misses: {self.stats['cache_misses']}")
        
        # Recent errors
        if self.stats['errors']:
            status.append("\n⚠️ Recent Errors:")
            for error in self.stats['errors'][-5:]:
                status.append(f"   {error['time']}: {error['type']} - {error.get('file', 'N/A')}")
        
        # Configuration summary
        status.append("\n⚙️ Current Configuration:")
        important_configs = [
            'model', 'similarity_threshold', 'chunk_size', 'top_k_results',
            'enable_caching', 'debug_mode', 'smart_chunking'
        ]
        for key in important_configs:
            status.append(f"{key}: {self.config[key]}")
        
        return "\n".join(status)
    
    def export_logs(self) -> str:
        """Export query logs"""
        if not self.stats['query_log']:
            return "No queries logged yet"
        
        filename = f"query_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.stats['query_log'], f, indent=2)
        
        return f"✅ Exported {len(self.stats['query_log'])} queries to {filename}"
    
    def clear_cache(self, cache_type: str = 'all') -> str:
        """Clear specified cache"""
        cleared = []
        
        if cache_type in ['all', 'queries']:
            self.cache['queries'].clear()
            cleared.append('queries')
        
        if cache_type in ['all', 'embeddings']:
            self.cache['embeddings'].clear()
            cleared.append('embeddings')
        
        if cache_type in ['all', 'chunks']:
            self.cache['chunks'].clear()
            cleared.append('chunks')
        
        if cache_type in ['all', 'extractions']:
            self.cache['extractions'].clear()
            cleared.append('extractions')
        
        return f"✅ Cleared cache: {', '.join(cleared)}"
    
    def chat(self, message: str, history: List[Tuple[str, str]]):
        """Process chat with admin features"""
        if not message.strip():
            return history
        
        history.append([message, None])
        
        # Search for relevant content
        if self.chunks:
            results, debug = self.search_with_debug(message)
            
            if results:
                # Build context
                context = "\n\n".join([r['text'] for r in results[:self.config['top_k_results']]])
                context = context[:self.config['max_context_length']]
                
                sources = list(set([r['metadata']['source'] for r in results]))
                
                prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {message}

Answer:"""
                
                # Query LLM
                try:
                    response = requests.post(
                        f"{self.config['base_url']}/api/generate",
                        json={
                            "model": self.config['model'],
                            "prompt": prompt,
                            "stream": False,
                            "temperature": self.config['temperature'],
                            "max_tokens": self.config['max_tokens']
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        answer = response.json().get('response', 'No response')
                        
                        # Add debug info if enabled
                        if self.config['debug_mode']:
                            answer += f"\n\n📊 Debug Info:"
                            answer += f"\n• Found {len(results)} chunks"
                            answer += f"\n• Best score: {results[0]['score']:.3f}"
                            answer += f"\n• Sources: {', '.join(sources[:3])}"
                            answer += f"\n• Processing: {debug['processing_time']*1000:.0f}ms"
                            
                            if debug.get('fallback_used'):
                                answer += f"\n• ⚠️ Using fallback (no results above threshold)"
                        else:
                            answer += f"\n\n📄 Sources: {', '.join(sources[:3])}"
                    else:
                        answer = f"⚠️ Model error: {response.status_code}"
                        
                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
                    self.stats['errors'].append({
                        'type': 'llm_error',
                        'error': str(e),
                        'time': datetime.now().isoformat()
                    })
            else:
                answer = "No relevant information found. Try adjusting the similarity threshold."
        else:
            answer = "No documents loaded. Please upload documents first."
        
        history[-1][1] = answer
        return history


def create_admin_interface():
    """Create the comprehensive admin interface"""
    
    rag = AdminRAG()
    
    # Dark theme for admin
    custom_css = """
    :root {
        --primary: #6366F1;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --dark: #0F172A;
        --dark-secondary: #1E293B;
        --dark-light: #334155;
        --text: #F1F5F9;
        --text-dim: #94A3B8;
    }
    
    .gradio-container {
        background: var(--dark) !important;
        color: var(--text) !important;
    }
    
    .admin-header {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    
    .tab-nav button {
        color: var(--text-dim) !important;
        background: transparent !important;
    }
    
    .tab-nav button.selected {
        color: var(--primary) !important;
        border-bottom: 2px solid var(--primary) !important;
    }
    
    .status-card {
        background: var(--dark-secondary);
        border: 1px solid var(--dark-light);
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    .debug-output {
        background: #000;
        color: #0F0;
        font-family: 'Courier New', monospace;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Admin Dashboard", theme=gr.themes.Base()) as app:
        
        # Header
        gr.HTML("""
        <div class="admin-header">
            <h1 style="margin: 0; font-size: 1.75rem; font-weight: 700;">
                🔐 RAG Admin Dashboard
            </h1>
            <p style="margin: 0.25rem 0 0 0; opacity: 0.9;">
                Full system control, monitoring, and optimization
            </p>
        </div>
        """)
        
        with gr.Tabs():
            # Chat Interface Tab
            with gr.TabItem("💬 Chat Interface"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=500)
                        msg = gr.Textbox(placeholder="Test queries here...", show_label=False)
                        
                        with gr.Row():
                            send_btn = gr.Button("Send", variant="primary")
                            clear_btn = gr.Button("Clear")
                    
                    with gr.Column(scale=1):
                        file_upload = gr.File(
                            label="Upload Documents",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx"]
                        )
                        upload_btn = gr.Button("Process Documents", variant="primary")
                        upload_status = gr.Textbox(label="Status", lines=1)
                        upload_details = gr.Textbox(label="Details", lines=5)
            
            # Configuration Tab
            with gr.TabItem("⚙️ Configuration"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Search Configuration")
                        threshold = gr.Slider(0, 1, value=rag.config['similarity_threshold'], 
                                             label="Similarity Threshold", step=0.01)
                        top_k = gr.Slider(1, 20, value=rag.config['top_k_results'], 
                                         label="Top K Results", step=1)
                        
                        gr.Markdown("### Chunking Configuration")
                        chunk_size = gr.Slider(100, 5000, value=rag.config['chunk_size'],
                                              label="Chunk Size", step=100)
                        overlap = gr.Slider(0, 1000, value=rag.config['chunk_overlap'],
                                          label="Chunk Overlap", step=50)
                        min_chunk = gr.Slider(10, 500, value=rag.config['min_chunk_size'],
                                            label="Min Chunk Size", step=10)
                    
                    with gr.Column():
                        gr.Markdown("### Features")
                        smart_chunk = gr.Checkbox(value=rag.config['smart_chunking'],
                                                label="Smart Chunking")
                        preserve_sent = gr.Checkbox(value=rag.config['preserve_sentences'],
                                                  label="Preserve Sentences")
                        resume_mode = gr.Checkbox(value=rag.config['enable_resume_mode'],
                                                label="Resume Mode")
                        enable_cache = gr.Checkbox(value=rag.config['enable_caching'],
                                                 label="Enable Caching")
                        debug_mode = gr.Checkbox(value=rag.config['debug_mode'],
                                               label="Debug Mode")
                        log_queries = gr.Checkbox(value=rag.config['log_queries'],
                                                label="Log Queries")
                        
                        gr.Markdown("### Model Settings")
                        model = gr.Dropdown(["llama3.2", "mistral", "phi3"], 
                                          value=rag.config['model'], label="Model")
                        temperature = gr.Slider(0, 1, value=rag.config['temperature'],
                                              label="Temperature", step=0.1)
                        max_context = gr.Slider(500, 8000, value=rag.config['max_context_length'],
                                              label="Max Context Length", step=500)
                
                with gr.Row():
                    save_config_btn = gr.Button("💾 Save Config", variant="primary")
                    load_config_btn = gr.Button("📂 Load Config")
                    export_config_btn = gr.Button("📤 Export Config")
                    config_status = gr.Textbox(label="Config Status", lines=1)
                
                gr.Markdown("### Import/Export Configuration")
                config_json = gr.Textbox(label="Configuration JSON", lines=10)
                import_config_btn = gr.Button("📥 Import Configuration")
            
            # Testing Tab
            with gr.TabItem("🧪 Query Testing"):
                test_query = gr.Textbox(label="Test Query", placeholder="Enter query to test...")
                test_btn = gr.Button("Run Test", variant="primary")
                test_output = gr.Textbox(label="Test Results", lines=20, elem_classes="debug-output")
                
                gr.Examples(
                    examples=[
                        "experience",
                        "skills and qualifications", 
                        "create a professional bio",
                        "summarize the document",
                        "what are the main points"
                    ],
                    inputs=test_query
                )
            
            # Monitoring Tab
            with gr.TabItem("📊 System Monitor"):
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Status")
                    export_logs_btn = gr.Button("📥 Export Logs")
                    clear_errors_btn = gr.Button("🗑️ Clear Errors")
                
                system_status = gr.Textbox(label="System Status", lines=25)
                
                with gr.Row():
                    gr.Markdown("### Cache Management")
                    clear_all_cache_btn = gr.Button("Clear All Caches", variant="stop")
                    clear_query_cache_btn = gr.Button("Clear Query Cache")
                    clear_embed_cache_btn = gr.Button("Clear Embedding Cache")
                    cache_status = gr.Textbox(label="Cache Status", lines=1)
            
            # Analytics Tab
            with gr.TabItem("📈 Analytics"):
                gr.Markdown("### Query Analytics")
                analytics_output = gr.Textbox(label="Query Log (Last 20)", lines=15)
                
                gr.Markdown("### Performance Metrics")
                metrics_output = gr.Textbox(label="Performance Stats", lines=10)
                
                refresh_analytics_btn = gr.Button("🔄 Refresh Analytics")
        
        # Event handlers
        def update_all_config(*args):
            config_updates = {
                'similarity_threshold': args[0],
                'top_k_results': args[1],
                'chunk_size': args[2],
                'chunk_overlap': args[3],
                'min_chunk_size': args[4],
                'smart_chunking': args[5],
                'preserve_sentences': args[6],
                'enable_resume_mode': args[7],
                'enable_caching': args[8],
                'debug_mode': args[9],
                'log_queries': args[10],
                'model': args[11],
                'temperature': args[12],
                'max_context_length': args[13]
            }
            rag.update_config(**config_updates)
            return "✅ Configuration updated"
        
        def show_analytics():
            # Query log
            log_output = []
            if rag.stats['query_log']:
                for entry in rag.stats['query_log'][-20:]:
                    log_output.append(
                        f"{entry['time']}: '{entry['query']}' "
                        f"[{entry['results']} results, "
                        f"best: {entry['best_score']:.3f}, "
                        f"{entry['processing_ms']:.0f}ms]"
                    )
            else:
                log_output.append("No queries logged yet")
            
            # Metrics
            metrics = []
            metrics.append(f"Total Queries: {rag.stats['total_queries']}")
            metrics.append(f"Cache Hits: {rag.stats['cache_hits']}")
            metrics.append(f"Cache Misses: {rag.stats['cache_misses']}")
            if rag.stats['total_queries'] > 0:
                hit_rate = (rag.stats['cache_hits'] / (rag.stats['cache_hits'] + rag.stats['cache_misses'])) * 100
                metrics.append(f"Cache Hit Rate: {hit_rate:.1f}%")
            metrics.append(f"Avg Response Time: {rag.stats['avg_response_time']*1000:.2f}ms")
            metrics.append(f"Total Documents: {rag.stats['total_documents']}")
            metrics.append(f"Total Chunks: {rag.stats['total_chunks']}")
            metrics.append(f"Errors Logged: {len(rag.stats['errors'])}")
            
            return "\n".join(log_output), "\n".join(metrics)
        
        # Connect all configuration controls
        config_controls = [
            threshold, top_k, chunk_size, overlap, min_chunk,
            smart_chunk, preserve_sent, resume_mode, enable_cache,
            debug_mode, log_queries, model, temperature, max_context
        ]
        
        for control in config_controls:
            control.change(update_all_config, inputs=config_controls, outputs=config_status)
        
        # File upload
        upload_btn.click(
            rag.load_documents,
            inputs=[file_upload],
            outputs=[upload_status, gr.State(), upload_details]
        )
        
        # Chat
        msg.submit(lambda m, h: ("", rag.chat(m, h)), [msg, chatbot], [msg, chatbot])
        send_btn.click(lambda m, h: ("", rag.chat(m, h)), [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: [], outputs=[chatbot])
        
        # Testing
        test_btn.click(rag.test_query, inputs=test_query, outputs=test_output)
        
        # Monitoring
        refresh_btn.click(rag.get_system_status, outputs=system_status)
        export_logs_btn.click(rag.export_logs, outputs=cache_status)
        
        # Cache management
        clear_all_cache_btn.click(lambda: rag.clear_cache('all'), outputs=cache_status)
        clear_query_cache_btn.click(lambda: rag.clear_cache('queries'), outputs=cache_status)
        clear_embed_cache_btn.click(lambda: rag.clear_cache('embeddings'), outputs=cache_status)
        
        # Config management
        save_config_btn.click(rag.save_config, outputs=config_status)
        load_config_btn.click(rag.load_config, outputs=config_status)
        export_config_btn.click(lambda: rag.export_config(), outputs=config_json)
        import_config_btn.click(rag.import_config, inputs=config_json, outputs=config_status)
        
        # Analytics
        refresh_analytics_btn.click(show_analytics, outputs=[analytics_output, metrics_output])
        
        # Auto-refresh system status on load
        app.load(rag.get_system_status, outputs=system_status)
    
    return app


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🔐 ADMIN DASHBOARD - Full Control Center")
    print("="*50)
    print("Features:")
    print("  • Complete configuration control")
    print("  • Query testing & debugging")
    print("  • Performance monitoring")
    print("  • Cache management")
    print("  • Analytics & logging")
    print("  • Import/Export settings")
    print("="*50 + "\n")
    
    app = create_admin_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )