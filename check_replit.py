#!/usr/bin/env python3
import sys
print("✅ Python version:", sys.version)
try:
    import fastapi
    print("✅ FastAPI installed")
except ImportError:
    print("❌ FastAPI not installed - Replit will install it automatically")
try:
    import uvicorn
    print("✅ Uvicorn installed")
except ImportError:
    print("❌ Uvicorn not installed - Replit will install it automatically")
print("\n🚀 Ready for Replit deployment!")
print("📝 Main file: main_replit.py")
print("📦 Requirements: requirements.txt")
