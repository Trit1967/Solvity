# 🤖 RAGbot - Multi-Tenant RAG Chat Service for SMBs

> Transform your documents into an intelligent AI assistant. Built for small and medium businesses.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 🚀 Quick Start (5 minutes)

### Option 1: Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/ragbot.git
cd ragbot

# Copy environment file
cp .env.example .env

# Start with Docker
./deploy.sh --type docker

# Access at http://localhost:8000
```

### Option 2: Local Development
```bash
# Install dependencies
pip install -r requirements_api.txt

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# Run the application
./deploy.sh --type local
```

## 📋 Features

### MVP Features (Available Now)
- ✅ **Multi-tenant Architecture** - Isolated data per customer
- ✅ **Document Processing** - PDF, DOCX, TXT, CSV, XLSX support
- ✅ **Intelligent RAG** - Semantic search with embeddings
- ✅ **JWT Authentication** - Secure user sessions
- ✅ **Usage Tracking** - Monitor queries and storage
- ✅ **Rate Limiting** - Prevent abuse
- ✅ **Docker Ready** - One-command deployment
- ✅ **REST API** - Full API documentation

### Coming Soon (Phase 2)
- 💳 Stripe billing integration
- 📊 Analytics dashboard
- 🔄 Webhook notifications
- 👥 Team collaboration
- 🎨 White-label options

## 🏗️ Architecture

```
┌─────────────────┐
│   FastAPI App   │ ← REST API + Auth
├─────────────────┤
│   RAG Engine    │ ← Document Processing
├─────────────────┤
│     Ollama      │ ← Local LLM (Free)
├─────────────────┤
│     SQLite      │ ← User Data
└─────────────────┘
```

## 💰 Pricing Plans

| Plan | Price | Documents | Queries/Month | Storage |
|------|-------|-----------|---------------|---------|
| Starter | $29/mo | 100 | 1,000 | 100 MB |
| Pro | $99/mo | 1,000 | 10,000 | 1 GB |
| Enterprise | Custom | Unlimited | Unlimited | Unlimited |

## 🛠️ API Documentation

Once running, visit http://localhost:8000/docs for interactive API documentation.

### Authentication
```bash
# Sign up
curl -X POST http://localhost:8000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepass123", "company_name": "ACME Corp"}'

# Response includes access_token and api_key
```

### Upload Documents
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@document.pdf"
```

### Query Documents
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the document?"}'
```

## 🚢 Deployment Guide

### Deploy to VPS (Hetzner, DigitalOcean, etc.)

```bash
# SSH into your VPS
ssh root@your-server-ip

# Clone and deploy
git clone https://github.com/yourusername/ragbot.git
cd ragbot
./deploy.sh --type vps --domain yourdomain.com --email your@email.com
```

### Deploy to AWS EC2

1. Launch EC2 instance (t3.medium recommended)
2. Security group: Open ports 80, 443, 8000
3. SSH and run:
```bash
./deploy.sh --type vps --domain yourdomain.com
```

### Deploy to Google Cloud

```bash
# Create VM instance
gcloud compute instances create ragbot-vm \
  --machine-type=e2-medium \
  --image-family=ubuntu-2004-lts \
  --boot-disk-size=20GB

# SSH and deploy
gcloud compute ssh ragbot-vm
# Then follow VPS deployment steps
```

## 📁 Project Structure

```
ragbot/
├── app.py              # Main FastAPI application
├── rag_engine.py       # Core RAG functionality
├── requirements_api.txt # Python dependencies
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Multi-container setup
├── nginx.conf         # Reverse proxy config
├── deploy.sh          # Deployment script
├── .env.example       # Environment template
└── data/             # SQLite database
```

## 🔧 Configuration

Edit `.env` file:

```env
# Essential settings
SECRET_KEY=your-secret-key-minimum-32-chars
OLLAMA_HOST=http://localhost:11434

# Optional
STRIPE_SECRET_KEY=sk_test_...
REDIS_URL=redis://localhost:6379
```

## 📊 Monitoring

### Check Health
```bash
curl http://localhost:8000/health
```

### View Logs
```bash
# Docker logs
docker-compose logs -f ragbot

# Local logs
tail -f logs/ragbot.log
```

### Database Queries
```bash
sqlite3 data/ragbot.db "SELECT COUNT(*) FROM users;"
```

## 🐛 Troubleshooting

### Ollama not responding
```bash
# Restart Ollama
ollama serve
ollama pull llama3.2
```

### Port already in use
```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>
```

### Docker issues
```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📈 Roadmap

### Phase 1 (Current) ✅
- [x] Core RAG functionality
- [x] Multi-tenant support
- [x] Authentication system
- [x] Docker deployment
- [x] Basic API

### Phase 2 (Q1 2024)
- [ ] Stripe billing
- [ ] Admin dashboard
- [ ] Email notifications
- [ ] API SDK (Python, JS)

### Phase 3 (Q2 2024)
- [ ] Team collaboration
- [ ] Advanced analytics
- [ ] Custom model training
- [ ] Mobile apps

### Phase 4 (Q3 2024)
- [ ] Enterprise features
- [ ] SOC2 compliance
- [ ] SAML/SSO
- [ ] On-premise option

## 💳 Billing Integration (Coming Soon)

To enable Stripe billing:

1. Get Stripe API keys from https://dashboard.stripe.com
2. Add to `.env`:
```env
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
```
3. Uncomment Stripe code in `app.py`

## 🔒 Security

- JWT authentication with refresh tokens
- Rate limiting on all endpoints
- SQL injection protection
- XSS prevention
- CORS configuration
- SSL/TLS in production
- Isolated tenant data

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Ollama](https://ollama.ai/) - Local LLM inference
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [Docker](https://www.docker.com/) - Containerization

## 📧 Support

- 📖 [Documentation](https://docs.ragbot.ai) (coming soon)
- 💬 [Discord Community](https://discord.gg/ragbot) (coming soon)
- 📧 Email: support@ragbot.ai
- 🐛 [Issue Tracker](https://github.com/yourusername/ragbot/issues)

---

**Built with ❤️ for SMBs** | [Website](https://ragbot.ai) | [API Docs](http://localhost:8000/docs)