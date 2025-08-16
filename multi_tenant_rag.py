#!/usr/bin/env python3
"""
Multi-Tenant RAG Service
Each company gets isolated data and queries
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import gradio as gr
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import requests

# ==========================================
# OPTION 1: Shared Infrastructure (Recommended)
# ==========================================

class MultiTenantRAG:
    """
    Single instance, multiple companies
    Most cost-effective approach
    """
    
    def __init__(self):
        # Each company gets a separate collection in vector DB
        self.chroma_client = chromadb.PersistentClient(
            path="./tenant_data",
            settings=Settings(anonymized_telemetry=False)
        )
        self.model = "mistral"  # Commercial-friendly
        self.tenants = self.load_tenants()
    
    def load_tenants(self) -> Dict:
        """Load tenant configuration"""
        tenant_file = Path("tenants.json")
        if tenant_file.exists():
            with open(tenant_file) as f:
                return json.load(f)
        return {}
    
    def create_tenant(self, company_name: str, plan: str = "starter") -> str:
        """Create new tenant/company"""
        tenant_id = str(uuid.uuid4())
        
        # Create isolated vector collection
        collection_name = f"tenant_{tenant_id}"
        self.chroma_client.create_collection(
            name=collection_name,
            metadata={"company": company_name}
        )
        
        # Create tenant record
        self.tenants[tenant_id] = {
            "id": tenant_id,
            "company": company_name,
            "plan": plan,
            "created": datetime.now().isoformat(),
            "collection": collection_name,
            "api_key": self.generate_api_key(),
            "limits": self.get_plan_limits(plan),
            "usage": {
                "documents": 0,
                "queries": 0,
                "storage_mb": 0
            }
        }
        
        # Create isolated storage directory
        tenant_dir = Path(f"./data/{tenant_id}")
        tenant_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_tenants()
        return tenant_id
    
    def generate_api_key(self) -> str:
        """Generate unique API key for tenant"""
        return f"sk_{hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:32]}"
    
    def get_plan_limits(self, plan: str) -> Dict:
        """Get limits based on plan"""
        plans = {
            "starter": {
                "max_documents": 100,
                "max_queries_per_month": 1000,
                "max_storage_mb": 100,
                "max_users": 5
            },
            "pro": {
                "max_documents": 1000,
                "max_queries_per_month": 10000,
                "max_storage_mb": 1000,
                "max_users": 20
            },
            "enterprise": {
                "max_documents": -1,  # Unlimited
                "max_queries_per_month": -1,
                "max_storage_mb": -1,
                "max_users": -1
            }
        }
        return plans.get(plan, plans["starter"])
    
    def save_tenants(self):
        """Save tenant configuration"""
        with open("tenants.json", "w") as f:
            json.dump(self.tenants, f, indent=2)
    
    def authenticate_tenant(self, api_key: str) -> Optional[str]:
        """Authenticate and return tenant_id"""
        for tenant_id, tenant in self.tenants.items():
            if tenant["api_key"] == api_key:
                return tenant_id
        return None
    
    def add_document(self, tenant_id: str, document: str, metadata: Dict):
        """Add document to tenant's collection"""
        tenant = self.tenants[tenant_id]
        
        # Check limits
        if tenant["limits"]["max_documents"] != -1:
            if tenant["usage"]["documents"] >= tenant["limits"]["max_documents"]:
                raise HTTPException(402, "Document limit exceeded. Please upgrade.")
        
        # Get tenant's collection
        collection = self.chroma_client.get_collection(tenant["collection"])
        
        # Add document with tenant isolation
        doc_id = str(uuid.uuid4())
        collection.add(
            documents=[document],
            metadatas=[{**metadata, "tenant_id": tenant_id}],
            ids=[doc_id]
        )
        
        # Update usage
        tenant["usage"]["documents"] += 1
        self.save_tenants()
        
        return doc_id
    
    def query(self, tenant_id: str, question: str) -> str:
        """Query only tenant's documents"""
        tenant = self.tenants[tenant_id]
        
        # Check query limits
        if tenant["limits"]["max_queries_per_month"] != -1:
            if tenant["usage"]["queries"] >= tenant["limits"]["max_queries_per_month"]:
                raise HTTPException(402, "Query limit exceeded. Please upgrade.")
        
        # Get tenant's collection
        collection = self.chroma_client.get_collection(tenant["collection"])
        
        # Search only in tenant's data
        results = collection.query(
            query_texts=[question],
            n_results=3,
            where={"tenant_id": tenant_id}  # Critical: Filter by tenant
        )
        
        # Build context from tenant's documents only
        context = "\n".join(results["documents"][0]) if results["documents"] else ""
        
        # Query Ollama with tenant's context
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
                "stream": False
            }
        )
        
        # Update usage
        tenant["usage"]["queries"] += 1
        self.save_tenants()
        
        return response.json().get("response", "No answer")

# ==========================================
# OPTION 2: Container Per Company
# ==========================================

class ContainerPerTenant:
    """
    Each company gets their own Docker container
    Better isolation, higher cost
    """
    
    def provision_tenant(self, company: str, plan: str):
        """Spin up new container for company"""
        
        # Docker Compose template
        docker_compose = f"""
version: '3.8'
services:
  rag_{company}:
    image: your-rag-image:latest
    container_name: rag_{company}
    environment:
      - TENANT_ID={company}
      - MODEL=mistral
      - PORT={self.get_next_port()}
    volumes:
      - ./data/{company}:/app/data
    ports:
      - "{self.get_next_port()}:7860"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '{self.get_cpu_limit(plan)}'
          memory: {self.get_memory_limit(plan)}
"""
        
        # Write docker-compose file
        with open(f"docker-compose-{company}.yml", "w") as f:
            f.write(docker_compose)
        
        # Start container
        os.system(f"docker-compose -f docker-compose-{company}.yml up -d")
    
    def get_cpu_limit(self, plan: str) -> str:
        """CPU limits per plan"""
        limits = {
            "starter": "0.5",    # Half CPU
            "pro": "2.0",        # 2 CPUs
            "enterprise": "4.0"  # 4 CPUs
        }
        return limits.get(plan, "0.5")
    
    def get_memory_limit(self, plan: str) -> str:
        """Memory limits per plan"""
        limits = {
            "starter": "2g",
            "pro": "8g",
            "enterprise": "16g"
        }
        return limits.get(plan, "2g")
    
    def get_next_port(self) -> int:
        """Get next available port for tenant"""
        # Track used ports in database
        # Start from 8000, increment for each tenant
        return 8000 + len(self.list_tenants())

# ==========================================
# OPTION 3: Kubernetes Namespaces
# ==========================================

class KubernetesTenant:
    """
    Each company gets a Kubernetes namespace
    Best for scale, complex setup
    """
    
    def create_tenant_namespace(self, company: str):
        """Create K8s namespace and resources"""
        
        # Kubernetes manifest
        k8s_manifest = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: tenant-{company}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-deployment
  namespace: tenant-{company}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rag
  template:
    metadata:
      labels:
        app: rag
        tenant: {company}
    spec:
      containers:
      - name: rag
        image: your-rag:latest
        env:
        - name: TENANT_ID
          value: "{company}"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: rag-service
  namespace: tenant-{company}
spec:
  selector:
    app: rag
  ports:
  - port: 80
    targetPort: 7860
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  namespace: tenant-{company}
spec:
  rules:
  - host: {company}.yourservice.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-service
            port:
              number: 80
"""
        
        # Apply to cluster
        with open(f"k8s-{company}.yaml", "w") as f:
            f.write(k8s_manifest)
        
        os.system(f"kubectl apply -f k8s-{company}.yaml")

# ==========================================
# FastAPI Implementation
# ==========================================

app = FastAPI(title="Multi-Tenant RAG API")
rag = MultiTenantRAG()

class CreateTenantRequest(BaseModel):
    company_name: str
    plan: str = "starter"
    admin_email: str

class DocumentRequest(BaseModel):
    content: str
    metadata: Dict = {}

class QueryRequest(BaseModel):
    question: str

@app.post("/api/tenants")
async def create_tenant(request: CreateTenantRequest):
    """Create new tenant/company"""
    tenant_id = rag.create_tenant(request.company_name, request.plan)
    tenant = rag.tenants[tenant_id]
    
    return {
        "tenant_id": tenant_id,
        "api_key": tenant["api_key"],
        "plan": request.plan,
        "limits": tenant["limits"]
    }

@app.post("/api/documents")
async def add_document(
    request: DocumentRequest,
    api_key: str = Header(None)
):
    """Add document to tenant's collection"""
    tenant_id = rag.authenticate_tenant(api_key)
    if not tenant_id:
        raise HTTPException(401, "Invalid API key")
    
    doc_id = rag.add_document(tenant_id, request.content, request.metadata)
    return {"document_id": doc_id, "status": "added"}

@app.post("/api/query")
async def query_documents(
    request: QueryRequest,
    api_key: str = Header(None)
):
    """Query tenant's documents"""
    tenant_id = rag.authenticate_tenant(api_key)
    if not tenant_id:
        raise HTTPException(401, "Invalid API key")
    
    answer = rag.query(tenant_id, request.question)
    tenant = rag.tenants[tenant_id]
    
    return {
        "answer": answer,
        "usage": tenant["usage"],
        "limits": tenant["limits"]
    }

@app.get("/api/usage")
async def get_usage(api_key: str = Header(None)):
    """Get tenant's usage stats"""
    tenant_id = rag.authenticate_tenant(api_key)
    if not tenant_id:
        raise HTTPException(401, "Invalid API key")
    
    tenant = rag.tenants[tenant_id]
    return {
        "company": tenant["company"],
        "plan": tenant["plan"],
        "usage": tenant["usage"],
        "limits": tenant["limits"]
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🏢 Multi-Tenant RAG Service")
    print("="*60)
    print("\nArchitecture Options:")
    print("1. Shared Infrastructure (this file) - Most cost-effective")
    print("2. Container per tenant - Better isolation")
    print("3. Kubernetes namespaces - Best for scale")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)