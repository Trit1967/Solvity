#!/usr/bin/env python3
"""
Cached RAG System with Performance Optimization
Implements multiple caching layers for maximum efficiency
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
import pickle
import os
from datetime import datetime, timedelta
from functools import lru_cache
import threading
from collections import OrderedDict

class CacheManager:
    """Manages different types of caches with TTL and size limits"""
    
    def __init__(self, cache_dir=".rag_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory caches
        self.embedding_cache = {}  # Document -> Embeddings
        self.query_cache = OrderedDict()  # Query -> Results (LRU)
        self.chunk_cache = {}  # Document -> Chunks
        self.extraction_cache = {}  # File path -> Extracted text
        
        # Cache settings
        self.max_query_cache_size = 100
        self.query_cache_ttl = 3600  # 1 hour
        self.embedding_cache_ttl = 86400  # 24 hours
        
        # Cache statistics
        self.stats = {
            'query_hits': 0,
            'query_misses': 0,
            'embedding_hits': 0,
            'embedding_misses': 0,
            'chunk_hits': 0,
            'chunk_misses': 0,
            'extraction_hits': 0,
            'extraction_misses': 0,
            'total_cache_size_mb': 0
        }
        
        # Load persistent caches
        self.load_persistent_cache()
    
    def get_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        return hashlib.md5(str(args).encode()).hexdigest()
    
    def save_persistent_cache(self):
        """Save embeddings and chunks to disk"""
        try:
            # Save embedding cache
            embed_file = self.cache_dir / "embeddings.pkl"
            with open(embed_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            
            # Save chunk cache
            chunk_file = self.cache_dir / "chunks.pkl"
            with open(chunk_file, 'wb') as f:
                pickle.dump(self.chunk_cache, f)
            
            # Save extraction cache
            extract_file = self.cache_dir / "extractions.pkl"
            with open(extract_file, 'wb') as f:
                pickle.dump(self.extraction_cache, f)
                
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False
    
    def load_persistent_cache(self):
        """Load cached data from disk"""
        try:
            # Load embedding cache
            embed_file = self.cache_dir / "embeddings.pkl"
            if embed_file.exists():
                with open(embed_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"📦 Loaded {len(self.embedding_cache)} cached embeddings")
            
            # Load chunk cache
            chunk_file = self.cache_dir / "chunks.pkl"
            if chunk_file.exists():
                with open(chunk_file, 'rb') as f:
                    self.chunk_cache = pickle.load(f)
                print(f"📦 Loaded {len(self.chunk_cache)} cached chunk sets")
            
            # Load extraction cache
            extract_file = self.cache_dir / "extractions.pkl"
            if extract_file.exists():
                with open(extract_file, 'rb') as f:
                    self.extraction_cache = pickle.load(f)
                print(f"📦 Loaded {len(self.extraction_cache)} cached extractions")
                
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    def get_query_result(self, query: str, config: dict) -> Optional[Dict]:
        """Get cached query result if available"""
        cache_key = self.get_cache_key(query, config)
        
        if cache_key in self.query_cache:
            entry = self.query_cache[cache_key]
            if time.time() - entry['timestamp'] < self.query_cache_ttl:
                self.stats['query_hits'] += 1
                # Move to end (LRU)
                self.query_cache.move_to_end(cache_key)
                return entry['result']
            else:
                del self.query_cache[cache_key]
        
        self.stats['query_misses'] += 1
        return None
    
    def set_query_result(self, query: str, config: dict, result: Dict):
        """Cache query result"""
        cache_key = self.get_cache_key(query, config)
        
        # Enforce size limit (LRU eviction)
        if len(self.query_cache) >= self.max_query_cache_size:
            self.query_cache.popitem(last=False)
        
        self.query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def get_embeddings(self, doc_hash: str) -> Optional[np.ndarray]:
        """Get cached embeddings for document"""
        if doc_hash in self.embedding_cache:
            entry = self.embedding_cache[doc_hash]
            if time.time() - entry['timestamp'] < self.embedding_cache_ttl:
                self.stats['embedding_hits'] += 1
                return entry['embeddings']
        
        self.stats['embedding_misses'] += 1
        return None
    
    def set_embeddings(self, doc_hash: str, embeddings: np.ndarray):
        """Cache document embeddings"""
        self.embedding_cache[doc_hash] = {
            'embeddings': embeddings,
            'timestamp': time.time()
        }
        # Auto-save to disk
        self.save_persistent_cache()
    
    def get_chunks(self, doc_hash: str) -> Optional[List[Dict]]:
        """Get cached chunks for document"""
        if doc_hash in self.chunk_cache:
            self.stats['chunk_hits'] += 1
            return self.chunk_cache[doc_hash]
        
        self.stats['chunk_misses'] += 1
        return None
    
    def set_chunks(self, doc_hash: str, chunks: List[Dict]):
        """Cache document chunks"""
        self.chunk_cache[doc_hash] = chunks
        self.save_persistent_cache()
    
    def get_extraction(self, file_path: str, file_mtime: float) -> Optional[str]:
        """Get cached text extraction"""
        if file_path in self.extraction_cache:
            entry = self.extraction_cache[file_path]
            # Check if file hasn't been modified
            if entry['mtime'] == file_mtime:
                self.stats['extraction_hits'] += 1
                return entry['text']
        
        self.stats['extraction_misses'] += 1
        return None
    
    def set_extraction(self, file_path: str, file_mtime: float, text: str):
        """Cache text extraction"""
        self.extraction_cache[file_path] = {
            'text': text,
            'mtime': file_mtime
        }
        self.save_persistent_cache()
    
    def clear_cache(self, cache_type: str = 'all'):
        """Clear specified cache"""
        if cache_type in ['all', 'query']:
            self.query_cache.clear()
        if cache_type in ['all', 'embedding']:
            self.embedding_cache.clear()
        if cache_type in ['all', 'chunk']:
            self.chunk_cache.clear()
        if cache_type in ['all', 'extraction']:
            self.extraction_cache.clear()
        
        if cache_type == 'all':
            # Clear disk cache too
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
        
        return f"✅ Cleared {cache_type} cache"
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        # Calculate cache sizes
        total_size = 0
        
        # Query cache size
        query_size = sum(len(str(v)) for v in self.query_cache.values())
        
        # Embedding cache size (approximate)
        embed_size = sum(
            v['embeddings'].nbytes if isinstance(v['embeddings'], np.ndarray) else 0
            for v in self.embedding_cache.values()
        )
        
        # Update stats
        self.stats['total_cache_size_mb'] = (query_size + embed_size) / (1024 * 1024)
        self.stats['query_cache_entries'] = len(self.query_cache)
        self.stats['embedding_cache_entries'] = len(self.embedding_cache)
        self.stats['chunk_cache_entries'] = len(self.chunk_cache)
        self.stats['extraction_cache_entries'] = len(self.extraction_cache)
        
        # Calculate hit rates
        query_total = self.stats['query_hits'] + self.stats['query_misses']
        self.stats['query_hit_rate'] = (
            f"{(self.stats['query_hits'] / query_total * 100):.1f}%" 
            if query_total > 0 else "N/A"
        )
        
        embed_total = self.stats['embedding_hits'] + self.stats['embedding_misses']
        self.stats['embedding_hit_rate'] = (
            f"{(self.stats['embedding_hits'] / embed_total * 100):.1f}%"
            if embed_total > 0 else "N/A"
        )
        
        return self.stats


class CachedRAG:
    """RAG system with comprehensive caching"""
    
    def __init__(self):
        # Configuration
        self.config = {
            'model': 'llama3.2',
            'base_url': 'http://localhost:11434',
            'similarity_threshold': 0.15,
            'chunk_size': 1500,
            'chunk_overlap': 200,
            'top_k_results': 5,
            'enable_caching': True,
            'cache_embeddings': True,
            'cache_queries': True,
            'cache_extractions': True
        }
        
        # Initialize cache manager
        self.cache = CacheManager()
        
        # Initialize embedder with caching
        print("🧠 Loading semantic search model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Semantic search ready!")
        
        # Document storage
        self.documents = {}
        self.chunks = []
        self.chunk_embeddings = None
        self.chunk_metadata = []
        
        # Performance tracking
        self.last_operation_time = 0
    
    @lru_cache(maxsize=100)
    def _cached_encode(self, text: str) -> np.ndarray:
        """LRU cached encoding for single texts"""
        return self.embedder.encode([text])[0]
    
    def get_document_hash(self, content: str) -> str:
        """Generate hash for document content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def smart_chunk(self, text: str, filename: str) -> List[Dict]:
        """Chunk with caching"""
        doc_hash = self.get_document_hash(text + filename)
        
        # Check cache
        if self.config['enable_caching']:
            cached_chunks = self.cache.get_chunks(doc_hash)
            if cached_chunks:
                print(f"💾 Using cached chunks for {filename}")
                return cached_chunks
        
        # Generate chunks (your existing logic)
        chunks = []
        chunk_size = self.config['chunk_size']
        overlap = self.config['chunk_overlap']
        
        # Resume mode
        if 'resume' in filename.lower() or 'cv' in filename.lower():
            summary_chunk = text[:chunk_size * 2] if len(text) > chunk_size * 2 else text
            chunks.append({
                'text': summary_chunk,
                'source': filename,
                'type': 'resume_full',
                'size': len(summary_chunk)
            })
        
        # Regular chunking
        words = text.split()
        word_chunk_size = chunk_size // 5
        word_overlap = overlap // 5
        
        for i in range(0, len(words), word_chunk_size - word_overlap):
            chunk_text = ' '.join(words[i:i + word_chunk_size])
            if len(chunk_text) > 50:
                chunks.append({
                    'text': chunk_text,
                    'source': filename,
                    'type': 'generic',
                    'size': len(chunk_text)
                })
        
        # Cache the chunks
        if self.config['enable_caching'] and chunks:
            self.cache.set_chunks(doc_hash, chunks)
        
        return chunks
    
    def extract_text(self, file_path: str) -> str:
        """Extract text with caching"""
        # Check extraction cache
        if self.config['cache_extractions']:
            file_mtime = os.path.getmtime(file_path)
            cached_text = self.cache.get_extraction(file_path, file_mtime)
            if cached_text:
                print(f"💾 Using cached extraction for {Path(file_path).name}")
                return cached_text
        
        # Extract based on file type
        text = ""
        filename = Path(file_path).name.lower()
        
        if filename.endswith('.pdf'):
            # PDF extraction logic
            import PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n\n"
            except:
                text = "[PDF extraction failed]"
        
        elif filename.endswith('.docx'):
            # DOCX extraction logic
            from docx import Document
            try:
                doc = Document(file_path)
                text = "\n\n".join([p.text for p in doc.paragraphs if p.text])
            except:
                text = "[DOCX extraction failed]"
        
        else:
            # Plain text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        # Cache the extraction
        if self.config['cache_extractions'] and text:
            self.cache.set_extraction(file_path, file_mtime, text)
        
        return text
    
    def load_documents(self, files):
        """Load documents with caching"""
        start_time = time.time()
        
        if not files:
            return "No files selected", {}
        
        self.documents = {}
        self.chunks = []
        self.chunk_metadata = []
        all_embeddings = []
        
        for file in files:
            filename = Path(file.name).name
            
            # Extract text (with caching)
            content = self.extract_text(file.name)
            
            if content:
                self.documents[filename] = content
                doc_hash = self.get_document_hash(content + filename)
                
                # Get chunks (with caching)
                file_chunks = self.smart_chunk(content, filename)
                
                # Check for cached embeddings
                cached_embeddings = None
                if self.config['cache_embeddings']:
                    cached_embeddings = self.cache.get_embeddings(doc_hash)
                
                if cached_embeddings is not None:
                    print(f"💾 Using cached embeddings for {filename}")
                    all_embeddings.append(cached_embeddings)
                else:
                    # Generate embeddings
                    chunk_texts = [c['text'] for c in file_chunks]
                    if chunk_texts:
                        print(f"🔍 Creating embeddings for {filename}...")
                        embeddings = self.embedder.encode(chunk_texts)
                        all_embeddings.append(embeddings)
                        
                        # Cache embeddings
                        if self.config['cache_embeddings']:
                            self.cache.set_embeddings(doc_hash, embeddings)
                
                # Add chunks to collection
                for chunk in file_chunks:
                    self.chunks.append(chunk['text'])
                    self.chunk_metadata.append({
                        'source': chunk['source'],
                        'type': chunk['type'],
                        'size': chunk['size']
                    })
        
        # Combine all embeddings
        if all_embeddings:
            self.chunk_embeddings = np.vstack(all_embeddings)
        
        self.last_operation_time = time.time() - start_time
        
        return f"✅ Loaded {len(files)} files in {self.last_operation_time:.2f}s", {
            'files': len(files),
            'chunks': len(self.chunks),
            'time': self.last_operation_time,
            'cached': cached_embeddings is not None
        }
    
    def semantic_search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search with query caching"""
        if top_k is None:
            top_k = self.config['top_k_results']
        
        # Check query cache
        if self.config['cache_queries']:
            cache_key = f"{query}_{top_k}_{self.config['similarity_threshold']}"
            cached_result = self.cache.get_query_result(cache_key, self.config)
            if cached_result:
                print(f"💾 Using cached query result")
                return cached_result
        
        # Perform search
        if self.chunk_embeddings is None or len(self.chunks) == 0:
            return []
        
        # Encode query (with function-level caching)
        query_embedding = self._cached_encode(query).reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            if similarities[idx] > self.config['similarity_threshold']:
                results.append({
                    'text': self.chunks[idx],
                    'score': float(similarities[idx]),
                    'metadata': self.chunk_metadata[idx]
                })
        
        # Cache the result
        if self.config['cache_queries'] and results:
            cache_key = f"{query}_{top_k}_{self.config['similarity_threshold']}"
            self.cache.set_query_result(cache_key, self.config, results)
        
        return results
    
    def get_cache_info(self) -> str:
        """Get formatted cache information"""
        stats = self.cache.get_cache_stats()
        
        info = []
        info.append("📊 CACHE STATISTICS")
        info.append("=" * 40)
        info.append(f"Query Cache: {stats.get('query_cache_entries', 0)} entries")
        info.append(f"  Hit Rate: {stats.get('query_hit_rate', 'N/A')}")
        info.append(f"  Hits: {stats['query_hits']} | Misses: {stats['query_misses']}")
        
        info.append(f"\nEmbedding Cache: {stats.get('embedding_cache_entries', 0)} entries")
        info.append(f"  Hit Rate: {stats.get('embedding_hit_rate', 'N/A')}")
        info.append(f"  Hits: {stats['embedding_hits']} | Misses: {stats['embedding_misses']}")
        
        info.append(f"\nChunk Cache: {stats.get('chunk_cache_entries', 0)} entries")
        info.append(f"  Hits: {stats['chunk_hits']} | Misses: {stats['chunk_misses']}")
        
        info.append(f"\nExtraction Cache: {stats.get('extraction_cache_entries', 0)} entries")
        info.append(f"  Hits: {stats['extraction_hits']} | Misses: {stats['extraction_misses']}")
        
        info.append(f"\nTotal Cache Size: {stats['total_cache_size_mb']:.2f} MB")
        info.append(f"Last Operation: {self.last_operation_time:.2f}s")
        
        return "\n".join(info)


def create_cached_interface():
    """Create interface with cache controls"""
    
    rag = CachedRAG()
    
    with gr.Blocks(title="Cached RAG System") as app:
        
        gr.Markdown("# 🚀 Cached RAG System")
        gr.Markdown("High-performance RAG with multi-layer caching")
        
        with gr.Tabs():
            with gr.TabItem("💬 Chat"):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(placeholder="Ask anything...")
                
                with gr.Row():
                    file_upload = gr.File(label="Upload Files", file_count="multiple")
                    upload_btn = gr.Button("Upload")
                    status = gr.Textbox(label="Status")
            
            with gr.TabItem("💾 Cache Management"):
                cache_info = gr.Textbox(label="Cache Statistics", lines=20)
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Stats")
                    clear_query_btn = gr.Button("Clear Query Cache")
                    clear_embed_btn = gr.Button("Clear Embedding Cache")
                    clear_all_btn = gr.Button("Clear All Caches", variant="stop")
                
                with gr.Row():
                    cache_enabled = gr.Checkbox(
                        value=rag.config['enable_caching'],
                        label="Enable Caching"
                    )
                    cache_queries = gr.Checkbox(
                        value=rag.config['cache_queries'],
                        label="Cache Queries"
                    )
                    cache_embeddings = gr.Checkbox(
                        value=rag.config['cache_embeddings'],
                        label="Cache Embeddings"
                    )
                    cache_extractions = gr.Checkbox(
                        value=rag.config['cache_extractions'],
                        label="Cache Extractions"
                    )
        
        # Event handlers
        def update_cache_settings(enable, queries, embeddings, extractions):
            rag.config['enable_caching'] = enable
            rag.config['cache_queries'] = queries
            rag.config['cache_embeddings'] = embeddings
            rag.config['cache_extractions'] = extractions
            return "✅ Cache settings updated"
        
        # Connect events
        upload_btn.click(
            lambda files: (*rag.load_documents(files), rag.get_cache_info()),
            inputs=[file_upload],
            outputs=[status, gr.State(), cache_info]
        )
        
        refresh_btn.click(rag.get_cache_info, outputs=cache_info)
        
        clear_query_btn.click(
            lambda: (rag.cache.clear_cache('query'), rag.get_cache_info()),
            outputs=[status, cache_info]
        )
        
        clear_embed_btn.click(
            lambda: (rag.cache.clear_cache('embedding'), rag.get_cache_info()),
            outputs=[status, cache_info]
        )
        
        clear_all_btn.click(
            lambda: (rag.cache.clear_cache('all'), rag.get_cache_info()),
            outputs=[status, cache_info]
        )
        
        for checkbox in [cache_enabled, cache_queries, cache_embeddings, cache_extractions]:
            checkbox.change(
                update_cache_settings,
                inputs=[cache_enabled, cache_queries, cache_embeddings, cache_extractions],
                outputs=status
            )
    
    return app


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 CACHED RAG SYSTEM")
    print("="*60)
    print("Caching Layers:")
    print("  • Query results (LRU, 1hr TTL)")
    print("  • Document embeddings (24hr TTL)")
    print("  • Text chunks (persistent)")
    print("  • File extractions (persistent)")
    print("  • Embedding function (LRU)")
    print("="*60 + "\n")
    
    app = create_cached_interface()
    app.launch(server_name="0.0.0.0", server_port=7863)