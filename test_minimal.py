#!/usr/bin/env python3
"""
Minimal test to verify RAGbot setup
Tests core functionality without complex dependencies
"""

import os
import sys
from pathlib import Path

print("🧪 Testing RAGbot Setup")
print("=" * 50)

# Test 1: Check Python version
print(f"✅ Python version: {sys.version}")

# Test 2: Check if virtual environment is active
if 'venv' in sys.prefix:
    print("✅ Virtual environment is active")
else:
    print("⚠️  Virtual environment not active. Run: source venv/bin/activate")

# Test 3: Try importing core dependencies
try:
    import fastapi
    print("✅ FastAPI installed")
except ImportError:
    print("❌ FastAPI not installed")

try:
    import chromadb
    print("✅ ChromaDB installed")
except ImportError:
    print("❌ ChromaDB not installed")

try:
    from cryptography.fernet import Fernet
    print("✅ Cryptography installed")
except ImportError:
    print("❌ Cryptography not installed")

# Test 4: Check directories
dirs_to_check = ['data', 'tenant_data', 'uploads', 'cache', 'logs']
for dir_name in dirs_to_check:
    if Path(dir_name).exists():
        print(f"✅ Directory '{dir_name}' exists")
    else:
        print(f"⚠️  Directory '{dir_name}' missing - creating...")
        Path(dir_name).mkdir(exist_ok=True)

# Test 5: Check .env file
if Path('.env').exists():
    print("✅ .env file exists")
else:
    print("⚠️  .env file missing")

# Test 6: Check Ollama
import subprocess
try:
    result = subprocess.run(['curl', '-s', 'http://localhost:11434/api/tags'], 
                          capture_output=True, timeout=2)
    if result.returncode == 0:
        print("✅ Ollama is running")
    else:
        print("⚠️  Ollama not responding")
except:
    print("⚠️  Cannot connect to Ollama")

# Test 7: Try your existing multi-tenant RAG
try:
    from multi_tenant_rag import MultiTenantRAG
    rag = MultiTenantRAG()
    print("✅ MultiTenantRAG loaded successfully")
    print(f"   Tenants configured: {len(rag.tenants)}")
except Exception as e:
    print(f"⚠️  MultiTenantRAG error: {e}")

print("\n" + "=" * 50)
print("📊 Test Summary")
print("=" * 50)

# Simple API test
print("\n🌐 Starting minimal API test...")

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
    
    app = FastAPI()
    
    @app.get("/")
    def root():
        return {"status": "RAGbot Minimal Test OK"}
    
    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "message": "Minimal RAGbot is running"
        }
    
    print("\n✅ Minimal API created successfully!")
    print("\n🚀 To run the minimal API:")
    print("   python -m uvicorn test_minimal:app --reload")
    print("\n📖 Then visit: http://localhost:8000")
    
except Exception as e:
    print(f"❌ API test failed: {e}")

print("\n" + "=" * 50)
print("✅ Basic tests complete!")
print("\nIf all core components show ✅, you're ready to proceed.")
print("If you see ❌ errors, try:")
print("1. Run: ./quick_setup.sh")
print("2. Or use Docker: ./deploy_ragbot.sh docker")