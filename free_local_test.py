#!/usr/bin/env python3
"""
Ultra-minimal RAGbot for free local testing
No heavy dependencies, just core functionality
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import hashlib
import sqlite3
from datetime import datetime
import uvicorn
from pathlib import Path

# Create simple directories
Path("data").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)

app = FastAPI(title="RAGbot Free Test", version="1.0.0")

# Simple in-memory storage (no vector DB needed for testing)
storage = {
    "tenants": {},
    "documents": {},
    "queries": []
}

class TenantCreate(BaseModel):
    company_name: str
    email: str

class DocumentUpload(BaseModel):
    content: str
    filename: str

class Query(BaseModel):
    question: str

# Simple tenant management
@app.post("/api/tenant/create")
def create_tenant(tenant: TenantCreate):
    tenant_id = hashlib.md5(tenant.company_name.encode()).hexdigest()[:8]
    api_key = f"sk_test_{hashlib.md5(tenant.email.encode()).hexdigest()[:16]}"
    
    storage["tenants"][tenant_id] = {
        "id": tenant_id,
        "company": tenant.company_name,
        "email": tenant.email,
        "api_key": api_key,
        "created": datetime.now().isoformat()
    }
    
    return {
        "tenant_id": tenant_id,
        "api_key": api_key,
        "message": "Tenant created successfully!"
    }

# Document upload (simplified)
@app.post("/api/document/upload/{tenant_id}")
def upload_document(tenant_id: str, doc: DocumentUpload):
    if tenant_id not in storage["tenants"]:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    doc_id = hashlib.md5(f"{tenant_id}{doc.filename}".encode()).hexdigest()[:8]
    
    if tenant_id not in storage["documents"]:
        storage["documents"][tenant_id] = []
    
    storage["documents"][tenant_id].append({
        "id": doc_id,
        "filename": doc.filename,
        "content": doc.content,
        "uploaded": datetime.now().isoformat()
    })
    
    return {
        "document_id": doc_id,
        "message": "Document uploaded successfully!"
    }

# Simple query (keyword matching for testing)
@app.post("/api/query/{tenant_id}")
def query_documents(tenant_id: str, query: Query):
    if tenant_id not in storage["tenants"]:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    # Simple keyword search (no AI needed for testing)
    docs = storage["documents"].get(tenant_id, [])
    results = []
    
    keywords = query.question.lower().split()
    
    for doc in docs:
        content_lower = doc["content"].lower()
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        if matches > 0:
            results.append({
                "document": doc["filename"],
                "relevance": matches,
                "snippet": doc["content"][:200]
            })
    
    # Sort by relevance
    results.sort(key=lambda x: x["relevance"], reverse=True)
    
    # Store query for analytics
    storage["queries"].append({
        "tenant_id": tenant_id,
        "question": query.question,
        "timestamp": datetime.now().isoformat(),
        "results_found": len(results)
    })
    
    if results:
        answer = f"Found {len(results)} relevant documents. Top result from '{results[0]['document']}': {results[0]['snippet']}..."
    else:
        answer = "No relevant documents found. Please upload documents first."
    
    return {
        "answer": answer,
        "sources": results[:3],
        "query_id": len(storage["queries"])
    }

# Stats endpoint
@app.get("/api/stats")
def get_stats():
    return {
        "total_tenants": len(storage["tenants"]),
        "total_documents": sum(len(docs) for docs in storage["documents"].values()),
        "total_queries": len(storage["queries"]),
        "tenants": list(storage["tenants"].values())
    }

# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "RAGbot Free Test is running!",
        "timestamp": datetime.now().isoformat()
    }

# Home page
@app.get("/")
def home():
    return {
        "message": "🚀 RAGbot Free Test API",
        "endpoints": {
            "Create Tenant": "POST /api/tenant/create",
            "Upload Document": "POST /api/document/upload/{tenant_id}",
            "Query": "POST /api/query/{tenant_id}",
            "Stats": "GET /api/stats",
            "Health": "GET /health"
        },
        "test_steps": [
            "1. Create a tenant with POST /api/tenant/create",
            "2. Upload documents with POST /api/document/upload/{tenant_id}",
            "3. Query with POST /api/query/{tenant_id}",
            "4. Check stats with GET /api/stats"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting RAGbot Free Test Server...")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🏠 Home: http://localhost:8000")
    print("\n✅ No heavy dependencies required!")
    print("✅ No GPU/AI models needed for testing!")
    print("\n🧪 Test with curl or visit /docs for interactive testing")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)