# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAGbot is a secure multi-tenant RAG (Retrieval-Augmented Generation) service designed for SMBs. It features enterprise-grade security with AES-256 encryption per tenant, complete data isolation, and audit logging. The system uses Ollama for local LLM inference (data never leaves your infrastructure) and ChromaDB for vector storage.

## Architecture

The application has multiple implementations, with the primary production-ready versions being:
- `ragbot_secure_app.py` - FastAPI app with encryption and enhanced security
- `ragbot_app.py` - Standard FastAPI implementation with authentication
- `security_enhanced_rag.py` - Core secure RAG engine with encryption layer
- `multi_tenant_rag.py` - Base multi-tenant RAG implementation

Key architectural components:
- **API Layer**: FastAPI with JWT authentication and rate limiting
- **Security Layer**: Per-tenant AES-256 encryption, audit logging, input validation
- **RAG Engine**: ChromaDB for vector storage, sentence-transformers for embeddings
- **LLM**: Ollama (local) with llama3.2 model as default

## Common Development Commands

### Setup & Installation
```bash
# Initial setup with virtual environment
./setup_secure.sh

# Quick setup without Ollama installation
./quick_setup.sh

# Activate virtual environment (required before running)
source venv/bin/activate
```

### Running the Application
```bash
# Run secure API server (production)
python ragbot_secure_app.py

# Run standard API server
python ragbot_app.py

# Deploy with Docker
./deploy_ragbot.sh docker

# Run with UI
./deploy_ragbot.sh local ui
```

### Testing
```bash
# Run basic tests
python test_rag.py

# Test minimal functionality
python test_minimal.py

# Test document processing with specific file
python test_rag.py /path/to/document.pdf
```

### Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull required model
ollama pull llama3.2

# Check available models
./check_models.sh
```

### Deployment
```bash
# Deploy to production VPS
./deploy_ragbot.sh production

# Push to GitHub
./push_to_github.sh

# Quick push with auto-commit
./quick_push.sh
```

## Critical File Locations

- **Environment Configuration**: `.env` (copy from `.env.ragbot` template)
- **Secure Data Storage**: `secure_data/` (chmod 700, encrypted tenant data)
- **Tenant Data**: `tenant_data/` (per-tenant document storage)
- **Audit Logs**: `logs/audit.log` (immutable audit trail)
- **User Database**: `data/users.db` (SQLite user management)

## API Endpoints

Main endpoints in `ragbot_secure_app.py`:
- `POST /api/auth/signup` - Create new tenant with encryption
- `POST /api/auth/login` - Authenticate and get JWT token
- `POST /api/documents/upload` - Upload document with encryption
- `POST /api/query` - Query documents with context
- `GET /api/tenant/info` - Get tenant details and usage

## Security Considerations

1. **Always generate new keys** for production using `setup_secure.sh`
2. **Never commit** `.env` file or any files in `secure_data/`
3. **Audit logs** are immutable - check `logs/audit.log` for security events
4. **Rate limiting** is enabled by default (100 requests/minute)
5. **Input validation** happens at API and RAG engine levels

## Dependencies Management

Core requirements are in:
- `requirements_ragbot.txt` - Production dependencies
- `requirements_minimal.txt` - Minimal setup
- `requirements_api.txt` - API-specific packages

Key dependencies:
- FastAPI for API framework
- ChromaDB for vector database
- sentence-transformers for embeddings
- cryptography for AES-256 encryption
- pydantic for data validation
- slowapi for rate limiting

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.ragbot.yml up -d

# View logs
docker logs ragbot-api -f

# Stop services
docker-compose -f docker-compose.ragbot.yml down
```

## Working with Tenants

The system is multi-tenant by design. Each tenant has:
- Isolated vector database collection
- Encrypted data storage with unique key
- API key for authentication
- Usage tracking and rate limits
- Immutable audit trail

When debugging tenant issues, check:
1. `secure_data/tenants.json` - Tenant registry (encrypted)
2. `logs/audit.log` - Security events
3. `tenant_data/{tenant_id}/` - Document storage

## Performance Optimization

- Document processing handles ~1000 pages/minute
- Query response time < 2 seconds average
- Supports 100+ concurrent users on minimal VPS
- ChromaDB persistence reduces memory usage
- Caching layer available in `cached_rag.py`

## Common Issues & Solutions

1. **Ollama not running**: Start with `ollama serve &`
2. **Model not found**: Pull with `ollama pull llama3.2`
3. **Permission denied on secure_data**: Run `chmod 700 secure_data`
4. **Virtual environment issues**: Deactivate and reactivate with `deactivate && source venv/bin/activate`
5. **Port already in use**: Change port in `.env` or kill existing process