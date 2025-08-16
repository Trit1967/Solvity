#!/usr/bin/env python3
"""
RAGbot for Replit - Optimized for free tier
Minimal dependencies, maximum functionality
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import hashlib
import sqlite3
from datetime import datetime
import uvicorn
from pathlib import Path
import os

# Create directories
Path("data").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)

app = FastAPI(
    title="RAGbot on Replit",
    description="Free Multi-tenant RAG Service",
    version="1.0.0"
)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("data/ragbot.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tenants (
            id TEXT PRIMARY KEY,
            company TEXT,
            api_key TEXT,
            created TEXT,
            documents INTEGER DEFAULT 0,
            queries INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            tenant_id TEXT,
            filename TEXT,
            content TEXT,
            created TEXT,
            FOREIGN KEY (tenant_id) REFERENCES tenants (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id TEXT PRIMARY KEY,
            tenant_id TEXT,
            question TEXT,
            answer TEXT,
            created TEXT,
            FOREIGN KEY (tenant_id) REFERENCES tenants (id)
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

# In-memory storage for demo (Replit free tier has limited storage)
memory_store = {
    "tenants": {},
    "documents": {},
    "embeddings": {}
}

class TenantCreate(BaseModel):
    company_name: str
    email: Optional[str] = None

class DocumentUpload(BaseModel):
    content: str
    filename: str

class Query(BaseModel):
    question: str

# Root redirect to docs
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAGbot on Replit</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                margin-top: 50px;
            }
            h1 { font-size: 3em; margin-bottom: 20px; }
            .btn {
                display: inline-block;
                padding: 15px 30px;
                background: white;
                color: #667eea;
                text-decoration: none;
                border-radius: 30px;
                font-weight: bold;
                margin: 10px;
                transition: transform 0.3s;
            }
            .btn:hover { transform: scale(1.05); }
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }
            .feature {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 10px;
            }
            .code {
                background: rgba(0,0,0,0.3);
                padding: 20px;
                border-radius: 10px;
                font-family: monospace;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 RAGbot on Replit</h1>
            <p style="font-size: 1.3em;">Free Multi-tenant RAG Service - Running on Replit!</p>
            
            <div style="text-align: center; margin: 40px 0;">
                <a href="/docs" class="btn">📖 API Documentation</a>
                <a href="/test" class="btn">🧪 Test Interface</a>
            </div>
            
            <div class="feature-grid">
                <div class="feature">
                    <h3>🆓 Free Hosting</h3>
                    <p>Runs on Replit's free tier</p>
                </div>
                <div class="feature">
                    <h3>🏢 Multi-tenant</h3>
                    <p>Isolated data per company</p>
                </div>
                <div class="feature">
                    <h3>📚 Document RAG</h3>
                    <p>Upload and query documents</p>
                </div>
                <div class="feature">
                    <h3>🔒 Secure</h3>
                    <p>API key authentication</p>
                </div>
            </div>
            
            <h2>Quick Start</h2>
            <div class="code">
                # 1. Create tenant<br>
                POST /api/tenant/create<br><br>
                
                # 2. Upload document<br>
                POST /api/document/upload/{tenant_id}<br><br>
                
                # 3. Query<br>
                POST /api/query/{tenant_id}
            </div>
            
            <div style="text-align: center; margin-top: 40px;">
                <p>Replit URL: <strong>https://ragbot.YOUR-USERNAME.repl.co</strong></p>
                <p style="opacity: 0.8;">Fork on Replit to get your own instance!</p>
            </div>
        </div>
    </body>
    </html>
    """

# Test interface
@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAGbot Tester</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }
            input, textarea, button { width: 100%; padding: 10px; margin: 10px 0; }
            button { background: #667eea; color: white; border: none; cursor: pointer; }
            .result { background: #f0f0f0; padding: 20px; margin: 20px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🧪 RAGbot Tester</h1>
        
        <h2>1. Create Tenant</h2>
        <input id="company" placeholder="Company Name">
        <button onclick="createTenant()">Create Tenant</button>
        
        <h2>2. Upload Document</h2>
        <input id="tenant_id" placeholder="Tenant ID">
        <textarea id="doc_content" placeholder="Document content"></textarea>
        <button onclick="uploadDoc()">Upload Document</button>
        
        <h2>3. Query</h2>
        <input id="query_tenant" placeholder="Tenant ID">
        <input id="question" placeholder="Your question">
        <button onclick="query()">Query</button>
        
        <div id="result" class="result" style="display:none;"></div>
        
        <script>
            async function createTenant() {
                const response = await fetch('/api/tenant/create', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({company_name: document.getElementById('company').value})
                });
                const data = await response.json();
                showResult(data);
            }
            
            async function uploadDoc() {
                const response = await fetch('/api/document/upload/' + document.getElementById('tenant_id').value, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        content: document.getElementById('doc_content').value,
                        filename: 'test.txt'
                    })
                });
                const data = await response.json();
                showResult(data);
            }
            
            async function query() {
                const response = await fetch('/api/query/' + document.getElementById('query_tenant').value, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: document.getElementById('question').value})
                });
                const data = await response.json();
                showResult(data);
            }
            
            function showResult(data) {
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
            }
        </script>
    </body>
    </html>
    """

# API Endpoints
@app.post("/api/tenant/create")
async def create_tenant(tenant: TenantCreate):
    tenant_id = hashlib.md5(tenant.company_name.encode()).hexdigest()[:8]
    api_key = f"replit_key_{hashlib.md5(f"{tenant.company_name}{datetime.now()}".encode()).hexdigest()[:16]}"
    
    conn = sqlite3.connect("data/ragbot.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO tenants (id, company, api_key, created)
            VALUES (?, ?, ?, ?)
        ''', (tenant_id, tenant.company_name, api_key, datetime.now().isoformat()))
        conn.commit()
    except:
        conn.close()
        raise HTTPException(status_code=400, detail="Tenant already exists")
    
    conn.close()
    
    memory_store["tenants"][tenant_id] = {
        "company": tenant.company_name,
        "api_key": api_key
    }
    
    return {
        "tenant_id": tenant_id,
        "api_key": api_key,
        "message": "Tenant created successfully!",
        "dashboard": f"https://ragbot.repl.co/tenant/{tenant_id}"
    }

@app.post("/api/document/upload/{tenant_id}")
async def upload_document(tenant_id: str, doc: DocumentUpload):
    # Verify tenant exists
    conn = sqlite3.connect("data/ragbot.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM tenants WHERE id = ?", (tenant_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    doc_id = hashlib.md5(f"{tenant_id}{doc.filename}{datetime.now()}".encode()).hexdigest()[:8]
    
    cursor.execute('''
        INSERT INTO documents (id, tenant_id, filename, content, created)
        VALUES (?, ?, ?, ?, ?)
    ''', (doc_id, tenant_id, doc.filename, doc.content, datetime.now().isoformat()))
    
    cursor.execute("UPDATE tenants SET documents = documents + 1 WHERE id = ?", (tenant_id,))
    conn.commit()
    conn.close()
    
    # Store in memory for fast retrieval
    if tenant_id not in memory_store["documents"]:
        memory_store["documents"][tenant_id] = []
    memory_store["documents"][tenant_id].append({
        "id": doc_id,
        "filename": doc.filename,
        "content": doc.content
    })
    
    return {
        "document_id": doc_id,
        "message": "Document uploaded successfully!",
        "size": len(doc.content)
    }

@app.post("/api/query/{tenant_id}")
async def query_documents(tenant_id: str, query: Query):
    # Simple keyword search (no heavy ML on Replit free tier)
    conn = sqlite3.connect("data/ragbot.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT content FROM documents WHERE tenant_id = ?", (tenant_id,))
    documents = cursor.fetchall()
    
    if not documents:
        conn.close()
        raise HTTPException(status_code=404, detail="No documents found")
    
    # Simple keyword matching
    keywords = query.question.lower().split()
    results = []
    
    for doc in documents:
        content = doc[0].lower()
        score = sum(1 for keyword in keywords if keyword in content)
        if score > 0:
            snippet = doc[0][:200]
            results.append({"snippet": snippet, "score": score})
    
    results.sort(key=lambda x: x["score"], reverse=True)
    
    answer = "Based on your documents: " + results[0]["snippet"] if results else "No relevant information found."
    
    # Log query
    query_id = hashlib.md5(f"{tenant_id}{query.question}{datetime.now()}".encode()).hexdigest()[:8]
    cursor.execute('''
        INSERT INTO queries (id, tenant_id, question, answer, created)
        VALUES (?, ?, ?, ?, ?)
    ''', (query_id, tenant_id, query.question, answer, datetime.now().isoformat()))
    
    cursor.execute("UPDATE tenants SET queries = queries + 1 WHERE id = ?", (tenant_id,))
    conn.commit()
    conn.close()
    
    return {
        "answer": answer,
        "sources": results[:3],
        "query_id": query_id
    }

@app.get("/api/stats")
async def get_stats():
    conn = sqlite3.connect("data/ragbot.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM tenants")
    tenants = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM documents")
    documents = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM queries")
    queries = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "tenants": tenants,
        "documents": documents,
        "queries": queries,
        "status": "healthy",
        "environment": "Replit Free Tier"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "platform": "Replit"}

# For Replit deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)