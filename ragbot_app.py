#!/usr/bin/env python3
"""
RAGbot Unified Application
Wraps existing MultiTenantRAG with FastAPI and authentication
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import jwt
import bcrypt
import sqlite3
import json
import uuid
import os
import sys

# Import your existing MultiTenantRAG
from multi_tenant_rag import MultiTenantRAG

# Import existing UI components if needed
sys.path.append('./rag-document-assistant/src')

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configuration
class Settings:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    DATABASE_PATH = "./data/users.db"
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

settings = Settings()

# Initialize FastAPI
app = FastAPI(
    title="RAGbot API",
    description="Multi-tenant RAG Service for SMBs - Built on existing codebase",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize your existing RAG system
rag_system = MultiTenantRAG()

# Security
security = HTTPBearer()

# Pydantic models
class UserSignup(BaseModel):
    email: EmailStr
    password: str
    company_name: str
    plan: Optional[str] = "starter"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class DocumentUpload(BaseModel):
    content: str
    metadata: Dict[str, Any]

class QueryRequest(BaseModel):
    question: str
    context_window: Optional[int] = 5

class TenantUpgrade(BaseModel):
    plan: str

# Database setup for user management
def init_user_database():
    """Initialize user database (extends existing tenant system)"""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
Path("./data").mkdir(exist_ok=True)
init_user_database()

# JWT functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        tenant_id = payload.get("tenant_id")
        if tenant_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return tenant_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# API Endpoints

@app.post("/api/auth/signup")
async def signup(user_data: UserSignup):
    """Create new user and tenant"""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    
    # Check if email exists
    cursor.execute("SELECT id FROM users WHERE email = ?", (user_data.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create tenant using existing system
    tenant_id = rag_system.create_tenant(user_data.company_name, user_data.plan)
    
    # Create user
    user_id = str(uuid.uuid4())
    password_hash = bcrypt.hashpw(user_data.password.encode(), bcrypt.gensalt()).decode()
    
    cursor.execute(
        "INSERT INTO users (id, email, password_hash, tenant_id) VALUES (?, ?, ?, ?)",
        (user_id, user_data.email, password_hash, tenant_id)
    )
    
    conn.commit()
    conn.close()
    
    # Get API key from tenant
    api_key = rag_system.tenants[tenant_id]['api_key']
    
    # Create token
    access_token = create_access_token({"tenant_id": tenant_id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "api_key": api_key,
        "tenant_id": tenant_id
    }

@app.post("/api/auth/login")
async def login(user_data: UserLogin):
    """Login existing user"""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, password_hash, tenant_id FROM users WHERE email = ?",
        (user_data.email,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if not row or not bcrypt.checkpw(user_data.password.encode(), row[1].encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    tenant_id = row[2]
    
    # Get API key from existing tenant system
    api_key = rag_system.tenants.get(tenant_id, {}).get('api_key')
    
    access_token = create_access_token({"tenant_id": tenant_id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "api_key": api_key,
        "tenant_id": tenant_id
    }

@app.post("/api/documents")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    document: DocumentUpload,
    tenant_id: str = Depends(verify_token)
):
    """Upload document using existing add_document method"""
    try:
        doc_id = rag_system.add_document(
            tenant_id,
            document.content,
            document.metadata
        )
        return {"document_id": doc_id, "status": "success"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/file")
@limiter.limit("5/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    tenant_id: str = Depends(verify_token)
):
    """Upload file and process it"""
    # Read file content
    content = await file.read()
    
    # Process based on file type (use existing extraction logic)
    if file.filename.endswith('.txt'):
        text = content.decode('utf-8')
    elif file.filename.endswith('.pdf'):
        # Use existing PDF processing from your dashboards
        import PyPDF2
        from io import BytesIO
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    else:
        text = content.decode('utf-8', errors='ignore')
    
    # Add to RAG system
    doc_id = rag_system.add_document(
        tenant_id,
        text,
        {"filename": file.filename, "size": len(content)}
    )
    
    return {"document_id": doc_id, "filename": file.filename}

@app.post("/api/query")
@limiter.limit("60/minute")
async def query_documents(
    request: Request,
    query: QueryRequest,
    tenant_id: str = Depends(verify_token)
):
    """Query documents using existing query method"""
    try:
        answer = rag_system.query(tenant_id, query.question)
        
        # Get usage stats
        tenant = rag_system.tenants.get(tenant_id, {})
        
        return {
            "answer": answer,
            "usage": tenant.get("usage", {}),
            "remaining_queries": tenant.get("limits", {}).get("max_queries_per_month", 0) - 
                               tenant.get("usage", {}).get("queries", 0)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tenant/info")
async def get_tenant_info(tenant_id: str = Depends(verify_token)):
    """Get tenant information"""
    tenant = rag_system.tenants.get(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    return {
        "company": tenant["company"],
        "plan": tenant["plan"],
        "usage": tenant["usage"],
        "limits": tenant["limits"],
        "created": tenant["created"]
    }

@app.post("/api/tenant/upgrade")
async def upgrade_plan(
    upgrade: TenantUpgrade,
    tenant_id: str = Depends(verify_token)
):
    """Upgrade tenant plan"""
    if tenant_id not in rag_system.tenants:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    # Update plan and limits
    rag_system.tenants[tenant_id]["plan"] = upgrade.plan
    rag_system.tenants[tenant_id]["limits"] = rag_system.get_plan_limits(upgrade.plan)
    rag_system.save_tenants()
    
    return {"status": "success", "new_plan": upgrade.plan}

@app.get("/api/tenant/usage")
async def get_usage(tenant_id: str = Depends(verify_token)):
    """Get current usage statistics"""
    tenant = rag_system.tenants.get(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    return {
        "documents": tenant["usage"]["documents"],
        "queries": tenant["usage"]["queries"],
        "storage_mb": tenant["usage"]["storage_mb"],
        "limits": tenant["limits"]
    }

# API Key authentication (alternative to JWT)
@app.post("/api/query/direct")
async def query_with_api_key(
    api_key: str,
    question: str
):
    """Direct query using API key (for programmatic access)"""
    tenant_id = rag_system.authenticate_tenant(api_key)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        answer = rag_system.query(tenant_id, question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "tenants": len(rag_system.tenants),
        "timestamp": datetime.utcnow().isoformat()
    }

# Landing page
@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Simple landing page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAGbot - AI Document Assistant</title>
        <style>
            body { 
                font-family: -apple-system, system-ui, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                margin-top: 50px;
            }
            h1 { font-size: 3em; margin-bottom: 20px; }
            .subtitle { font-size: 1.5em; opacity: 0.9; margin-bottom: 40px; }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }
            .feature {
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 10px;
            }
            .cta {
                display: inline-block;
                padding: 15px 40px;
                background: white;
                color: #667eea;
                text-decoration: none;
                border-radius: 30px;
                font-weight: bold;
                margin: 20px 10px;
            }
            .cta:hover {
                transform: scale(1.05);
                transition: all 0.3s;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 RAGbot</h1>
            <p class="subtitle">Transform your documents into an intelligent AI assistant</p>
            
            <div class="features">
                <div class="feature">
                    <h3>📚 Multi-Format Support</h3>
                    <p>PDF, Word, Excel, CSV, and more</p>
                </div>
                <div class="feature">
                    <h3>🔒 Secure & Isolated</h3>
                    <p>Each company gets private data storage</p>
                </div>
                <div class="feature">
                    <h3>⚡ Instant Answers</h3>
                    <p>Query your documents in natural language</p>
                </div>
                <div class="feature">
                    <h3>💰 Simple Pricing</h3>
                    <p>Starting at $29/month for small teams</p>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 40px;">
                <a href="/docs" class="cta">View API Docs</a>
                <a href="/api/auth/signup" class="cta">Get Started</a>
            </div>
            
            <div style="text-align: center; margin-top: 60px; opacity: 0.7;">
                <p>Built on proven RAG technology • Powered by local LLMs • No data leaves your instance</p>
            </div>
        </div>
    </body>
    </html>
    """

# Gradio UI Integration (optional - mount existing UI)
@app.get("/ui")
async def redirect_to_ui():
    """Redirect to Gradio UI if running"""
    return {"message": "Start Gradio UI with: python rag-document-assistant/src/user_dashboard.py"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAGbot API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🏠 Landing Page: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)