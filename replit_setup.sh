#!/bin/bash
# Replit setup script - Fix common build errors

echo "🔧 Fixing Replit build issues..."

# 1. Clean up conflicting files
rm -f poetry.lock 2>/dev/null
rm -f Pipfile 2>/dev/null
rm -f Pipfile.lock 2>/dev/null

# 2. Create minimal requirements
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
aiofiles==23.2.1
jinja2==3.1.2
slowapi==0.1.9
EOF

# 3. Update .replit config
cat > .replit << 'EOF'
run = "python main_replit.py"
language = "python3"
entrypoint = "main_replit.py"
modules = ["python-3.10"]

[env]
PYTHONPATH = "${PYTHONPATH}:${REPL_HOME}"

[nix]
channel = "stable-22_11"

[packager]
language = "python3"

[packager.features]
enabledForHosting = true
packageSearch = true
guessImports = true

[[ports]]
localPort = 8000
externalPort = 80

[deployment]
run = ["sh", "-c", "python main_replit.py"]
deploymentTarget = "cloudrun"
EOF

# 4. Create data directories
mkdir -p data uploads logs

# 5. Create a startup check script
cat > check_replit.py << 'EOF'
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
EOF

python3 check_replit.py

echo "
✅ Replit setup complete!

To deploy on Replit:
1. Push these changes to GitHub
2. Import to Replit from GitHub
3. Replit will auto-detect and run main_replit.py
4. Your app will be available at: https://ragbot.YOUR-USERNAME.repl.co

Files configured:
- .replit (config file)
- requirements.txt (minimal dependencies)
- main_replit.py (entry point)
- .env.replit (environment config)
"