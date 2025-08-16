#!/usr/bin/env python3
"""
RAGbot - Multi-tenant RAG Chat Service for SMBs
Main FastAPI application with authentication, billing, and API endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import jwt
import bcrypt
import sqlite3
import json
import uuid
import os
from enum import Enum
import asyncio
import aiofiles
from contextlib import asynccontextmanager

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import our RAG engine (to be created)
from rag_engine import RAGEngine

# Environment configuration
from dotenv import load_dotenv
load_dotenv()

# Configuration
class Settings:
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/ragbot.db")
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.md', '.csv', '.xlsx'}

settings = Settings()

# Create necessary directories
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Path("./data").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)

# Database setup
def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            company_name TEXT,
            plan TEXT DEFAULT 'free',
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tenants table (for multi-tenancy)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tenants (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            plan TEXT DEFAULT 'starter',
            max_documents INTEGER DEFAULT 100,
            max_queries_per_month INTEGER DEFAULT 1000,
            max_storage_mb INTEGER DEFAULT 100,
            used_documents INTEGER DEFAULT 0,
            used_queries_this_month INTEGER DEFAULT 0,
            used_storage_mb REAL DEFAULT 0,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            chunk_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (tenant_id) REFERENCES tenants (id)
        )
    ''')
    
    # Queries table (for tracking usage)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            query TEXT NOT NULL,
            response TEXT,
            tokens_used INTEGER DEFAULT 0,
            response_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (tenant_id) REFERENCES tenants (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

# Pydantic models
class PlanType(str, Enum):
    free = "free"
    starter = "starter"
    pro = "pro"
    enterprise = "enterprise"

class UserSignup(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    company_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    context_window: Optional[int] = 5
    temperature: Optional[float] = 0.7

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    tokens_used: int

class TenantInfo(BaseModel):
    id: str
    plan: PlanType
    usage: Dict[str, Any]
    limits: Dict[str, Any]

# Security
security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_tenant_by_user_id(user_id: str) -> Optional[Dict]:
    """Get tenant information by user ID"""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tenants WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        columns = [description[0] for description in cursor.description]
        return dict(zip(columns, row))
    return None

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting RAGbot Service...")
    app.state.rag_engines = {}  # Cache RAG engines per tenant
    yield
    # Shutdown
    print("👋 Shutting down RAGbot Service...")

app = FastAPI(
    title="RAGbot API",
    description="Multi-tenant RAG Chat Service for SMBs",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Authentication endpoints
@app.post("/api/auth/signup", response_model=Dict[str, str])
async def signup(user_data: UserSignup):
    """Register a new user"""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    
    # Check if email exists
    cursor.execute("SELECT id FROM users WHERE email = ?", (user_data.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user_id = str(uuid.uuid4())
    password_hash = bcrypt.hashpw(user_data.password.encode(), bcrypt.gensalt()).decode()
    
    cursor.execute('''
        INSERT INTO users (id, email, password_hash, company_name, plan)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, user_data.email, password_hash, user_data.company_name, 'free'))
    
    # Create tenant
    tenant_id = str(uuid.uuid4())
    api_key = f"sk_{uuid.uuid4().hex[:32]}"
    
    cursor.execute('''
        INSERT INTO tenants (id, user_id, api_key, plan)
        VALUES (?, ?, ?, ?)
    ''', (tenant_id, user_id, api_key, 'starter'))
    
    conn.commit()
    conn.close()
    
    # Create user upload directory
    user_upload_dir = settings.UPLOAD_DIR / tenant_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate tokens
    access_token = create_access_token(data={"sub": user_id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "api_key": api_key
    }

@app.post("/api/auth/login", response_model=Dict[str, str])
async def login(user_data: UserLogin):
    """Login existing user"""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, password_hash FROM users WHERE email = ?", (user_data.email,))
    row = cursor.fetchone()
    
    if not row or not bcrypt.checkpw(user_data.password.encode(), row[1].encode()):
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    user_id = row[0]
    
    # Get API key
    cursor.execute("SELECT api_key FROM tenants WHERE user_id = ?", (user_id,))
    api_key_row = cursor.fetchone()
    conn.close()
    
    access_token = create_access_token(data={"sub": user_id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "api_key": api_key_row[0] if api_key_row else None
    }

# Document management endpoints
@app.post("/api/documents/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Depends(verify_token)
):
    """Upload a document for RAG processing"""
    # Get tenant
    tenant = get_tenant_by_user_id(user_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type {file_ext} not allowed")
    
    # Check usage limits
    if tenant['used_documents'] >= tenant['max_documents']:
        raise HTTPException(status_code=403, detail="Document limit reached. Please upgrade your plan.")
    
    # Save file
    file_id = str(uuid.uuid4())
    file_path = settings.UPLOAD_DIR / tenant['id'] / f"{file_id}{file_ext}"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read and save file
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    # Update database
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO documents (id, tenant_id, filename, file_path, file_size)
        VALUES (?, ?, ?, ?, ?)
    ''', (file_id, tenant['id'], file.filename, str(file_path), len(content)))
    
    cursor.execute('''
        UPDATE tenants 
        SET used_documents = used_documents + 1,
            used_storage_mb = used_storage_mb + ?
        WHERE id = ?
    ''', (len(content) / (1024 * 1024), tenant['id']))
    
    conn.commit()
    conn.close()
    
    # Process document with RAG engine
    if tenant['id'] not in app.state.rag_engines:
        app.state.rag_engines[tenant['id']] = RAGEngine(tenant_id=tenant['id'])
    
    rag_engine = app.state.rag_engines[tenant['id']]
    chunks_created = await rag_engine.process_document(str(file_path), file.filename)
    
    return {
        "document_id": file_id,
        "filename": file.filename,
        "chunks_created": chunks_created,
        "status": "processed"
    }

@app.get("/api/documents")
async def list_documents(user_id: str = Depends(verify_token)):
    """List all documents for a tenant"""
    tenant = get_tenant_by_user_id(user_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, file_size, chunk_count, created_at
        FROM documents
        WHERE tenant_id = ?
        ORDER BY created_at DESC
    ''', (tenant['id'],))
    
    documents = []
    for row in cursor.fetchall():
        documents.append({
            "id": row[0],
            "filename": row[1],
            "file_size": row[2],
            "chunk_count": row[3],
            "created_at": row[4]
        })
    
    conn.close()
    return {"documents": documents}

# Query endpoints
@app.post("/api/query", response_model=QueryResponse)
@limiter.limit("60/minute")
async def query_documents(
    request: Request,
    query_data: QueryRequest,
    user_id: str = Depends(verify_token)
):
    """Query documents using RAG"""
    tenant = get_tenant_by_user_id(user_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    # Check usage limits
    if tenant['used_queries_this_month'] >= tenant['max_queries_per_month']:
        raise HTTPException(status_code=403, detail="Query limit reached. Please upgrade your plan.")
    
    # Get or create RAG engine for tenant
    if tenant['id'] not in app.state.rag_engines:
        app.state.rag_engines[tenant['id']] = RAGEngine(tenant_id=tenant['id'])
    
    rag_engine = app.state.rag_engines[tenant['id']]
    
    # Perform query
    import time
    start_time = time.time()
    
    result = await rag_engine.query(
        query_data.query,
        top_k=query_data.context_window,
        temperature=query_data.temperature
    )
    
    response_time_ms = int((time.time() - start_time) * 1000)
    
    # Update usage
    conn = sqlite3.connect(settings.DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO queries (id, tenant_id, query, response, tokens_used, response_time_ms)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (str(uuid.uuid4()), tenant['id'], query_data.query, result['answer'], 
          result.get('tokens_used', 0), response_time_ms))
    
    cursor.execute('''
        UPDATE tenants 
        SET used_queries_this_month = used_queries_this_month + 1
        WHERE id = ?
    ''', (tenant['id'],))
    
    conn.commit()
    conn.close()
    
    return QueryResponse(
        answer=result['answer'],
        sources=result.get('sources', []),
        confidence=result.get('confidence', 0.0),
        tokens_used=result.get('tokens_used', 0)
    )

# Tenant management endpoints
@app.get("/api/tenant/info", response_model=TenantInfo)
async def get_tenant_info(user_id: str = Depends(verify_token)):
    """Get tenant information and usage"""
    tenant = get_tenant_by_user_id(user_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    return TenantInfo(
        id=tenant['id'],
        plan=tenant['plan'],
        usage={
            "documents": tenant['used_documents'],
            "queries_this_month": tenant['used_queries_this_month'],
            "storage_mb": round(tenant['used_storage_mb'], 2)
        },
        limits={
            "max_documents": tenant['max_documents'],
            "max_queries_per_month": tenant['max_queries_per_month'],
            "max_storage_mb": tenant['max_storage_mb']
        }
    )

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Landing page (temporary - will be replaced with proper frontend)
@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Simple landing page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAGbot - AI Document Assistant for SMBs</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   max-width: 1200px; margin: 0 auto; padding: 20px; }
            .hero { text-align: center; padding: 60px 0; }
            h1 { font-size: 3em; margin-bottom: 20px; }
            .subtitle { font-size: 1.3em; color: #666; margin-bottom: 40px; }
            .cta { display: inline-block; padding: 15px 30px; background: #007bff; 
                   color: white; text-decoration: none; border-radius: 5px; font-size: 1.1em; }
            .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                       gap: 30px; margin-top: 60px; }
            .feature { padding: 20px; border: 1px solid #eee; border-radius: 8px; }
            .pricing { margin-top: 60px; text-align: center; }
            .price-cards { display: flex; justify-content: center; gap: 20px; margin-top: 30px; }
            .price-card { border: 1px solid #ddd; border-radius: 8px; padding: 30px; width: 250px; }
            .price { font-size: 2em; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="hero">
            <h1>🤖 RAGbot</h1>
            <p class="subtitle">Transform your documents into an intelligent AI assistant</p>
            <a href="/docs" class="cta">View API Documentation</a>
        </div>
        
        <div class="features">
            <div class="feature">
                <h3>📚 Document Intelligence</h3>
                <p>Upload PDFs, Word docs, spreadsheets and let AI understand your content</p>
            </div>
            <div class="feature">
                <h3>💬 Natural Conversations</h3>
                <p>Ask questions in plain English and get accurate answers from your documents</p>
            </div>
            <div class="feature">
                <h3>🔒 Secure & Private</h3>
                <p>Your data stays yours. Enterprise-grade security with isolated tenants</p>
            </div>
        </div>
        
        <div class="pricing">
            <h2>Simple, Transparent Pricing</h2>
            <div class="price-cards">
                <div class="price-card">
                    <h3>Starter</h3>
                    <div class="price">$29/mo</div>
                    <ul style="text-align: left;">
                        <li>100 documents</li>
                        <li>1,000 queries/month</li>
                        <li>Email support</li>
                    </ul>
                </div>
                <div class="price-card" style="border-color: #007bff;">
                    <h3>Pro</h3>
                    <div class="price">$99/mo</div>
                    <ul style="text-align: left;">
                        <li>1,000 documents</li>
                        <li>10,000 queries/month</li>
                        <li>Priority support</li>
                        <li>API access</li>
                    </ul>
                </div>
                <div class="price-card">
                    <h3>Enterprise</h3>
                    <div class="price">Custom</div>
                    <ul style="text-align: left;">
                        <li>Unlimited documents</li>
                        <li>Unlimited queries</li>
                        <li>Dedicated support</li>
                        <li>Custom integrations</li>
                    </ul>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)