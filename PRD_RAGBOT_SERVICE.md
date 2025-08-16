# Product Requirements Document: RAGbot Chat Service for SMBs
## Leveraging Existing Codebase

---

## Executive Summary

Transform your existing multi-implementation RAG project into a production-ready SaaS platform targeting SMBs. The codebase already contains 90% of required functionality across multiple implementations - we need to consolidate, productize, and add billing/auth layers.

## Current Asset Analysis

### ✅ What You Already Have

#### Core RAG Implementations (Multiple Options Available):
1. **`multi_tenant_rag.py`** - Multi-tenant architecture with ChromaDB
   - Tenant isolation already implemented
   - API key generation system
   - Usage tracking per tenant
   - Three deployment options (shared, container-per-tenant, K8s)
   - Plan-based limits (starter/pro/enterprise)

2. **`rag-document-assistant/`** - Complete UI System
   - User dashboard (`user_dashboard.py`) - simplified interface
   - Admin dashboard (`admin_dashboard.py`) - full control panel
   - Docker deployment ready
   - Ollama integration configured

3. **Working Implementations**:
   - `final_rag.py` - Production-ready with caching
   - `cached_rag.py` - Advanced caching system
   - `smart_rag.py` - Intelligent chunking
   - `modern_rag.py` - Modern UI with Gradio
   - `ollama_rag.py` - Local LLM integration
   - `openai_rag.py` - OpenAI API option

4. **Infrastructure**:
   - Docker configurations exist
   - Deployment scripts available
   - Gradio UI implementations
   - Multiple embedding options

### 🔧 What Needs Integration

1. **Authentication Layer** - Add JWT to existing system
2. **Payment Processing** - Stripe integration
3. **API Consolidation** - Unify endpoints
4. **Database Schema** - Centralize user/tenant data
5. **Usage Metering** - Enhance existing tracking

---

## Phase 1: MVP Launch (Week 1) - $0-50/month
**Goal**: Consolidate existing code into unified service

### Technical Implementation Plan

#### 1.1 Core Service Consolidation
```python
# Merge these existing components:
- multi_tenant_rag.py (backend logic)
- rag-document-assistant/src/user_dashboard.py (UI)
- rag-document-assistant/src/admin_dashboard.py (admin panel)
- cached_rag.py (performance optimization)
```

**Action Items**:
- [ ] Create `main.py` that imports MultiTenantRAG class
- [ ] Add FastAPI wrapper around existing functions
- [ ] Keep existing Gradio UI as frontend option
- [ ] Use existing ChromaDB setup for vector storage

#### 1.2 Authentication Integration
```python
# Add to existing multi_tenant_rag.py:
- JWT token generation using existing api_key system
- Session management for Gradio interface
- API authentication middleware
```

**Files to Modify**:
- `multi_tenant_rag.py`: Add `create_user()`, `login()` methods
- Create `auth_middleware.py` using existing `authenticate_tenant()`

#### 1.3 Database Setup
```sql
-- Extend existing tenant structure with:
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE,
    password_hash TEXT,
    tenant_id TEXT REFERENCES tenants(id)
);

-- Your existing tenant tracking:
- tenants.json → SQLite database
- Keep existing usage tracking structure
```

### Deployment Strategy

**Use Existing Docker Setup**:
```yaml
# Modify existing docker-compose.yml:
services:
  ragbot:
    build: .
    # Use existing ollama service
    # Add environment variables for auth
```

**Quick Launch Commands**:
```bash
# Use existing scripts with modifications:
./setup_rag.sh  # Modify to include auth setup
./quick_start.sh # Add user creation flow
```

---

## Phase 2: Growth (Weeks 2-3) - $100-200/month
**Goal**: Add payment and enhance existing features

### 2.1 Payment Integration

**Integrate Stripe with Existing Plan System**:
```python
# multi_tenant_rag.py already has:
- get_plan_limits() method
- Plan tiers (starter/pro/enterprise)
- Usage tracking

# Add:
- stripe_customer_id to tenant record
- Webhook handler for subscription events
- Update plan on payment
```

### 2.2 API Enhancement

**Expose Existing Functions as REST API**:
```python
# Current methods to expose:
- add_document() → POST /api/documents
- query() → POST /api/query
- create_tenant() → POST /api/tenants
```

### 2.3 UI Improvements

**Enhance Existing Gradio Interfaces**:
- Add billing page to admin dashboard
- Show usage metrics in user dashboard
- Keep existing document processing UI

---

## Phase 3: Scale (Weeks 4-6) - $500-1000/month
**Goal**: Production optimization using existing code

### 3.1 Performance Optimization

**Leverage Existing Caching**:
- `cached_rag.py` has complete caching system
- `final_rag.py` has production optimizations
- Merge cache implementations

### 3.2 Advanced Features

**Activate Existing Capabilities**:
- Multiple model support (already in codebase)
- Advanced chunking from `smart_rag.py`
- Debug mode from admin dashboard

### 3.3 Deployment Options

**Use Existing Multi-Deployment Code**:
```python
# multi_tenant_rag.py already supports:
1. Shared infrastructure (Option 1)
2. Container per tenant (Option 2)
3. Kubernetes namespaces (Option 3)

# Start with Option 1, scale to Option 2/3 as needed
```

---

## Migration Path from Current Codebase

### Step 1: File Consolidation (Day 1)
```bash
# Create new structure:
ragbot/
├── app.py (new - FastAPI wrapper)
├── core/
│   ├── multi_tenant.py (from multi_tenant_rag.py)
│   ├── rag_engine.py (merge smart_rag + cached_rag)
│   └── auth.py (new - simple JWT)
├── ui/
│   ├── user_dashboard.py (existing)
│   └── admin_dashboard.py (existing)
└── docker-compose.yml (existing, modified)
```

### Step 2: Minimal Code Changes (Day 2)
```python
# app.py - New file wrapping existing code:
from fastapi import FastAPI
from core.multi_tenant import MultiTenantRAG

app = FastAPI()
rag = MultiTenantRAG()  # Your existing class

@app.post("/api/query")
async def query(tenant_id: str, question: str):
    return rag.query(tenant_id, question)  # Existing method
```

### Step 3: Add Authentication (Day 3)
```python
# Extend existing authenticate_tenant():
def authenticate_user(email: str, password: str):
    # Check credentials
    # Return existing tenant_id and api_key
    return {
        "tenant_id": tenant_id,
        "api_key": existing_api_key,
        "token": generate_jwt(tenant_id)
    }
```

### Step 4: Deploy (Day 4)
```bash
# Use existing deployment scripts:
./install_ollama.sh  # Already exists
docker-compose up    # Already configured
```

---

## Cost Analysis Using Existing Infrastructure

### Current Setup Costs:
- **Ollama**: Free (local LLM)
- **ChromaDB**: Free (embedded)
- **Gradio**: Free (UI framework)
- **Hosting**: $20-40/month (single VPS)

### Revenue Projections:
- **Month 1**: 10 customers × $29 = $290 MRR
- **Month 2**: 50 customers × $29-99 = $3,000 MRR
- **Month 3**: 150 customers = $10,000 MRR

---

## Implementation Checklist

### Week 1: MVP
- [ ] Merge `multi_tenant_rag.py` with FastAPI
- [ ] Add basic auth to existing code
- [ ] Deploy using existing Docker setup
- [ ] Test with 5 pilot customers

### Week 2: Payments
- [ ] Add Stripe to existing billing structure
- [ ] Create pricing page
- [ ] Implement usage limits enforcement

### Week 3: Polish
- [ ] Enhance existing UI
- [ ] Add monitoring
- [ ] Create landing page

### Week 4: Launch
- [ ] ProductHunt submission
- [ ] Content marketing
- [ ] Direct outreach to SMBs

---

## Key Advantages of Your Existing Code

1. **Multi-tenancy already built** - Complete isolation per company
2. **Multiple UI options** - Admin and user dashboards ready
3. **Caching implemented** - Performance optimization done
4. **Docker ready** - Deployment configuration exists
5. **Multiple RAG strategies** - Can A/B test different implementations
6. **Ollama integration** - Free LLM costs

---

## Next Immediate Steps

1. **Today**: 
   - Copy `multi_tenant_rag.py` to new `app.py`
   - Add FastAPI endpoints
   - Test locally

2. **Tomorrow**:
   - Add JWT authentication
   - Create user signup flow
   - Deploy to $20 VPS

3. **Day 3**:
   - Create landing page
   - Add Stripe (can be commented out initially)
   - Launch to first customers

---

## Conclusion

Your existing codebase is 90% ready for production. The main work is:
1. **Consolidation** - Merge best parts of each implementation
2. **Authentication** - Add simple JWT layer
3. **Packaging** - Wrap in FastAPI for REST API
4. **Deployment** - Use existing Docker configuration

You can literally launch an MVP in 3-4 days by leveraging what you've already built!