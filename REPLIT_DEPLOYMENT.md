# 🚀 Deploying RAGbot to Replit

## Quick Deploy (Easiest)

### Option 1: Fork from GitHub
1. Go to [Replit](https://replit.com)
2. Click "Create Repl" → "Import from GitHub"
3. Paste: `https://github.com/Trit1967/Solvity`
4. Click "Import from GitHub"
5. Your RAGbot will auto-deploy!

### Option 2: Direct Fork
[![Run on Replit](https://replit.com/badge/github/Trit1967/Solvity)](https://replit.com/github/Trit1967/Solvity)

## What You Get

### Free Tier Features
- ✅ Multi-tenant RAG system
- ✅ SQLite database (no external dependencies)
- ✅ FastAPI REST API
- ✅ Web UI for testing
- ✅ API key authentication
- ✅ Document upload & query
- ✅ Auto-scaling with Replit

### URLs After Deployment
- Main App: `https://solvity-YOUR-USERNAME.repl.co`
- API Docs: `https://solvity-YOUR-USERNAME.repl.co/docs`
- Test UI: `https://solvity-YOUR-USERNAME.repl.co/test`

## Configuration

### Environment Variables (Optional)
In Replit's Secrets tab, you can add:
```
PORT=8000
MAX_UPLOAD_SIZE=10485760  # 10MB
RATE_LIMIT=100  # requests per minute
```

### File Structure
```
Solvity/
├── main_replit.py      # Main FastAPI app (optimized for Replit)
├── .replit             # Replit configuration
├── replit.nix          # Nix packages
├── requirements_minimal.txt  # Minimal dependencies
└── data/               # SQLite database (auto-created)
```

## Testing Your Deployment

### 1. Create a Tenant
```bash
curl -X POST https://solvity-YOUR-USERNAME.repl.co/api/tenant/create \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Test Company"}'
```

### 2. Upload Document
```bash
curl -X POST https://solvity-YOUR-USERNAME.repl.co/api/document/upload/{tenant_id} \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "test.txt",
    "content": "This is a test document about AI and machine learning."
  }'
```

### 3. Query
```bash
curl -X POST https://solvity-YOUR-USERNAME.repl.co/api/query/{tenant_id} \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## Scaling on Replit

### Free Tier Limits
- 0.5 vCPU
- 512MB RAM
- Always-on with activity
- SQLite database (local)
- ~100 concurrent users

### Paid Upgrade Options
1. **Hacker Plan ($7/month)**
   - 2 vCPUs
   - 2GB RAM
   - Priority hosting
   - ~500 concurrent users

2. **Pro Plan ($20/month)**
   - 4 vCPUs
   - 4GB RAM
   - Custom domains
   - ~1000 concurrent users

## Optimization Tips

### For Better Performance
1. Keep documents under 1MB each
2. Use the web UI for testing
3. Batch document uploads
4. Enable Replit's "Always On" in settings

### Database Management
```python
# To reset database (if needed)
import os
os.remove("data/ragbot.db")
# Restart the Repl to recreate
```

## Monitoring

Check system stats:
```
GET https://solvity-YOUR-USERNAME.repl.co/api/stats
```

Returns:
```json
{
  "tenants": 5,
  "documents": 42,
  "queries": 156,
  "status": "healthy",
  "environment": "Replit Free Tier"
}
```

## Troubleshooting

### Common Issues

1. **"Module not found" error**
   - Click "Packages" in Replit
   - Search and add: fastapi, uvicorn, pydantic

2. **"Port already in use"**
   - Stop and restart the Repl
   - Check .replit config

3. **"Database locked"**
   - SQLite limitation with concurrent writes
   - Upgrade to Hacker plan for better performance

## Next Steps

### After Deployment
1. ✅ Test with the web UI
2. ✅ Create your first tenant
3. ✅ Upload sample documents
4. ✅ Share your API endpoint
5. ✅ Monitor usage via /api/stats

### Production Ready
When ready to scale:
1. Upgrade Replit plan
2. Add PostgreSQL database
3. Implement vector embeddings
4. Add user authentication frontend

## Support

- GitHub Issues: [Trit1967/Solvity](https://github.com/Trit1967/Solvity/issues)
- Replit Community: [@YOUR-USERNAME](https://replit.com/@YOUR-USERNAME)

---

**Ready to Deploy?** Just fork to Replit and your RAGbot is live in 30 seconds! 🎉