# 🚀 RAGbot Launch Guide - Using Your Existing Code

## Quick Start (3 Commands)

```bash
# 1. Setup environment
cp .env.ragbot .env

# 2. Install and run
pip install -r requirements_ragbot.txt

# 3. Launch!
./deploy_ragbot.sh local
```

**That's it!** RAGbot is now running at http://localhost:8000

---

## What We Built Using Your Code

### 1. **ragbot_app.py** - API wrapper around your MultiTenantRAG
- Uses your existing `multi_tenant_rag.py` class
- Adds JWT authentication
- Exposes your methods as REST endpoints
- No changes to your core logic!

### 2. **Integration with Your Existing Features**
- ✅ Multi-tenant isolation (already in your code)
- ✅ ChromaDB vector storage (already configured)
- ✅ Usage tracking (already implemented)
- ✅ Plan limits (starter/pro/enterprise ready)
- ✅ Gradio UIs (can run alongside)

---

## Deployment Options

### Option 1: Local Development
```bash
./deploy_ragbot.sh local
```
- Runs on http://localhost:8000
- Uses your existing Ollama setup
- Perfect for testing

### Option 2: Docker (Production Ready)
```bash
./deploy_ragbot.sh docker
```
- Containerized deployment
- Auto-starts Ollama
- Ready for VPS deployment

### Option 3: Docker with Your Existing UIs
```bash
./deploy_ragbot.sh docker-ui
```
- API on port 8000
- User Dashboard on port 7860 (your existing UI)
- Admin Dashboard on port 7861 (your existing UI)

---

## Testing the API

### 1. Create an Account
```bash
curl -X POST http://localhost:8000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "testpass123",
    "company_name": "Test Company"
  }'
```

### 2. Upload a Document
```bash
curl -X POST http://localhost:8000/api/documents \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "This is a test document about RAG systems.",
    "metadata": {"filename": "test.txt"}
  }'
```

### 3. Query Your Documents
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

---

## Your Existing Code Structure

```
What You Built:
├── multi_tenant_rag.py     → Core multi-tenant logic
├── rag-document-assistant/ → Complete UI system
├── cached_rag.py          → Performance optimization
├── smart_rag.py           → Intelligent chunking
└── ollama_rag.py          → LLM integration

What We Added:
├── ragbot_app.py          → FastAPI wrapper
├── .env.ragbot            → Configuration
├── requirements_ragbot.txt → Combined dependencies
└── deploy_ragbot.sh       → One-click deployment
```

---

## Deploy to Production (VPS)

### 1. Get a VPS ($20/month)
- **Hetzner**: CX21 (2 vCPU, 4GB RAM) - €5.83/month
- **DigitalOcean**: Basic Droplet - $24/month
- **Linode**: Shared CPU - $20/month

### 2. Deploy
```bash
# On your VPS
git clone your-repo
cd rag
./deploy_ragbot.sh docker
```

### 3. Add Domain (Optional)
```bash
# Install nginx
sudo apt install nginx certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com
```

---

## Monitor Usage

### Check System Status
```bash
./deploy_ragbot.sh status
```

### View Logs
```bash
# Docker logs
docker-compose -f docker-compose.ragbot.yml logs -f

# API logs
tail -f logs/ragbot.log
```

### View Tenant Data
Your existing `tenants.json` file tracks all customer data!

---

## Revenue Tracking

With your existing plan structure:
- **Starter ($29/mo)**: 100 docs, 1K queries
- **Pro ($99/mo)**: 1000 docs, 10K queries  
- **Enterprise (Custom)**: Unlimited

### Quick Math:
- 10 customers = $290-990/month
- 50 customers = $1,450-4,950/month
- 100 customers = $2,900-9,900/month

---

## Next Steps to Launch

### Today (Day 1):
1. ✅ Run `./deploy_ragbot.sh local`
2. ✅ Test the API endpoints
3. ✅ Verify multi-tenant isolation works

### Tomorrow (Day 2):
1. Deploy to a $20 VPS
2. Point a domain (optional)
3. Share with 5 beta users

### This Week:
1. Add Stripe (code is ready, just add keys)
2. Create a simple landing page
3. Post on ProductHunt/HackerNews

---

## FAQ

**Q: Do I need to modify my existing code?**
A: No! The API wrapper uses your code as-is.

**Q: Can I still use my Gradio UIs?**
A: Yes! They can run alongside the API.

**Q: What about my existing data?**
A: It's preserved. The API reads your `tenants.json`.

**Q: How much will hosting cost?**
A: $20-40/month for a VPS that can handle 100+ customers.

---

## Support & Troubleshooting

### API Not Starting?
```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check port 8000
lsof -i :8000
```

### Docker Issues?
```bash
# Rebuild
docker-compose -f docker-compose.ragbot.yml build --no-cache

# Reset
docker-compose -f docker-compose.ragbot.yml down -v
```

---

## 🎉 You're Ready to Launch!

Your existing code + our API wrapper = **Production-ready SaaS**

Start with `./deploy_ragbot.sh local` and you can be live today!