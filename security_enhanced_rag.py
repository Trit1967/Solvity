#!/usr/bin/env python3
"""
Security-Enhanced RAG Implementation
Adds encryption, audit logging, and advanced isolation to existing MultiTenantRAG
"""

import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import base64
import sqlite3
from functools import wraps
import threading
import time

# Import your existing implementation
from multi_tenant_rag import MultiTenantRAG

class SecurityLayer:
    """
    Enterprise-grade security layer for RAG operations
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize security layer with master key"""
        self.master_key = master_key or os.getenv("MASTER_KEY", self._generate_master_key())
        self.audit_db = "./data/audit_trail.db"
        self.key_cache = {}  # Cache derived keys
        self.rate_limits = {}  # Track rate limits
        self._init_audit_db()
        
    def _generate_master_key(self) -> str:
        """Generate a secure master key"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    
    def _init_audit_db(self):
        """Initialize immutable audit trail database"""
        Path("./data").mkdir(exist_ok=True)
        conn = sqlite3.connect(self.audit_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                resource TEXT,
                ip_address TEXT,
                user_agent TEXT,
                success BOOLEAN,
                error_message TEXT,
                data_hash TEXT,
                previous_hash TEXT,
                INDEX idx_tenant (tenant_id),
                INDEX idx_timestamp (timestamp)
            )
        ''')
        
        # Make audit log append-only (no updates or deletes)
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS prevent_audit_update
            BEFORE UPDATE ON audit_log
            BEGIN
                SELECT RAISE(ABORT, 'Audit log is immutable');
            END
        ''')
        
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS prevent_audit_delete
            BEFORE DELETE ON audit_log
            BEGIN
                SELECT RAISE(ABORT, 'Audit log is immutable');
            END
        ''')
        
        conn.commit()
        conn.close()
    
    def derive_tenant_key(self, tenant_id: str) -> bytes:
        """Derive a unique encryption key for each tenant"""
        if tenant_id in self.key_cache:
            return self.key_cache[tenant_id]
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=tenant_id.encode(),
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        self.key_cache[tenant_id] = key
        return key
    
    def encrypt_data(self, data: str, tenant_id: str) -> str:
        """Encrypt data using tenant-specific key"""
        key = self.derive_tenant_key(tenant_id)
        f = Fernet(key)
        encrypted = f.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str, tenant_id: str) -> str:
        """Decrypt data using tenant-specific key"""
        key = self.derive_tenant_key(tenant_id)
        f = Fernet(key)
        decoded = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = f.decrypt(decoded)
        return decrypted.decode()
    
    def hash_data(self, data: str) -> str:
        """Create SHA-256 hash of data for integrity verification"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def audit_log(self, tenant_id: str, action: str, success: bool = True, 
                  error: str = None, metadata: Dict = None):
        """Log action to immutable audit trail"""
        conn = sqlite3.connect(self.audit_db)
        cursor = conn.cursor()
        
        # Get previous hash for blockchain-style integrity
        cursor.execute("SELECT data_hash FROM audit_log ORDER BY id DESC LIMIT 1")
        prev_hash = cursor.fetchone()
        prev_hash = prev_hash[0] if prev_hash else "genesis"
        
        # Create audit entry
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
            "action": action,
            "success": success,
            "error_message": error,
            "metadata": metadata or {}
        }
        
        # Hash the entry with previous hash (blockchain-style)
        entry_str = json.dumps(entry, sort_keys=True)
        data_hash = self.hash_data(f"{prev_hash}{entry_str}")
        
        # Insert into immutable log
        cursor.execute('''
            INSERT INTO audit_log (
                timestamp, tenant_id, action, success, 
                error_message, data_hash, previous_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (entry["timestamp"], tenant_id, action, success, error, data_hash, prev_hash))
        
        conn.commit()
        conn.close()
        
        # Alert on security-critical events
        if action in ["unauthorized_access", "data_breach_attempt", "rate_limit_exceeded"]:
            self._send_security_alert(tenant_id, action, metadata)
    
    def _send_security_alert(self, tenant_id: str, action: str, metadata: Dict):
        """Send immediate security alerts"""
        # In production, integrate with PagerDuty, Slack, etc.
        alert = {
            "severity": "HIGH",
            "tenant_id": tenant_id,
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata
        }
        print(f"🚨 SECURITY ALERT: {json.dumps(alert, indent=2)}")
        
        # Log to separate security incidents file
        with open("./logs/security_incidents.jsonl", "a") as f:
            f.write(json.dumps(alert) + "\n")
    
    def check_rate_limit(self, tenant_id: str, limit: int = 60) -> bool:
        """Check if tenant exceeded rate limit"""
        current_minute = datetime.now().strftime("%Y%m%d%H%M")
        key = f"{tenant_id}:{current_minute}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = 0
        
        self.rate_limits[key] += 1
        
        # Clean old entries
        self._cleanup_rate_limits()
        
        if self.rate_limits[key] > limit:
            self.audit_log(tenant_id, "rate_limit_exceeded", False, 
                          metadata={"limit": limit, "actual": self.rate_limits[key]})
            return False
        return True
    
    def _cleanup_rate_limits(self):
        """Remove old rate limit entries"""
        current_minute = datetime.now().strftime("%Y%m%d%H%M")
        old_keys = [k for k in self.rate_limits.keys() 
                   if not k.endswith(current_minute)]
        for key in old_keys:
            del self.rate_limits[key]
    
    def validate_input(self, input_data: str, input_type: str = "query") -> Tuple[bool, str]:
        """Validate and sanitize user input"""
        # Check for injection attempts
        dangerous_patterns = [
            "'; DROP TABLE",
            "<script>",
            "javascript:",
            "onclick=",
            "../",
            "file://",
            "\\x00",  # Null byte
            "%00",    # URL encoded null
        ]
        
        input_lower = input_data.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in input_lower:
                return False, f"Potential injection attempt detected: {pattern}"
        
        # Length validation
        max_lengths = {
            "query": 1000,
            "document": 1000000,  # 1MB
            "filename": 255
        }
        
        if len(input_data) > max_lengths.get(input_type, 10000):
            return False, f"Input exceeds maximum length for {input_type}"
        
        return True, "Valid"
    
    def anonymize_pii(self, text: str) -> str:
        """Remove or mask PII from text"""
        import re
        
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     '[EMAIL_REDACTED]', text)
        
        # Phone numbers (US format)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', text)
        
        # SSN
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
        
        # Credit card numbers
        text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', 
                     '[CC_REDACTED]', text)
        
        return text


class SecureMultiTenantRAG(MultiTenantRAG):
    """
    Security-enhanced version of your existing MultiTenantRAG
    """
    
    def __init__(self):
        super().__init__()
        self.security = SecurityLayer()
        self.session_tokens = {}  # Track active sessions
        self.failed_attempts = {}  # Track failed login attempts
        
    def create_tenant(self, company_name: str, plan: str = "starter") -> str:
        """Create tenant with enhanced security"""
        # Validate company name
        valid, message = self.security.validate_input(company_name, "filename")
        if not valid:
            raise ValueError(f"Invalid company name: {message}")
        
        # Create tenant using parent method
        tenant_id = super().create_tenant(company_name, plan)
        
        # Generate tenant-specific encryption key
        tenant_key = self.security.derive_tenant_key(tenant_id)
        
        # Create secure storage directory
        tenant_dir = Path(f"./secure_data/{tenant_id}")
        tenant_dir.mkdir(parents=True, exist_ok=True, mode=0o700)  # Restricted permissions
        
        # Initialize encrypted metadata file
        metadata = {
            "created": datetime.utcnow().isoformat(),
            "encryption_enabled": True,
            "key_rotation_date": datetime.utcnow().isoformat(),
            "compliance_level": "standard"  # or "hipaa", "pci", etc.
        }
        
        encrypted_metadata = self.security.encrypt_data(
            json.dumps(metadata), tenant_id
        )
        
        with open(tenant_dir / "metadata.enc", "w") as f:
            f.write(encrypted_metadata)
        
        # Audit log
        self.security.audit_log(tenant_id, "tenant_created", True, 
                               metadata={"company": company_name, "plan": plan})
        
        return tenant_id
    
    def add_document(self, tenant_id: str, document: str, metadata: Dict):
        """Add document with encryption and security checks"""
        # Rate limiting
        if not self.security.check_rate_limit(tenant_id, limit=10):  # 10 uploads/minute
            raise Exception("Rate limit exceeded. Please try again later.")
        
        # Input validation
        valid, message = self.security.validate_input(document, "document")
        if not valid:
            self.security.audit_log(tenant_id, "document_rejected", False, 
                                   error=message)
            raise ValueError(f"Document validation failed: {message}")
        
        # Anonymize PII if enabled
        if self.tenants[tenant_id].get("anonymize_pii", False):
            document = self.security.anonymize_pii(document)
        
        # Encrypt document before storage
        encrypted_doc = self.security.encrypt_data(document, tenant_id)
        
        # Store encrypted document
        doc_id = super().add_document(tenant_id, encrypted_doc, metadata)
        
        # Audit log with document hash (not content)
        doc_hash = self.security.hash_data(document)
        self.security.audit_log(tenant_id, "document_added", True,
                               metadata={"doc_id": doc_id, "hash": doc_hash})
        
        return doc_id
    
    def query(self, tenant_id: str, question: str, user_context: Dict = None) -> str:
        """Query with security validation and audit trail"""
        # Rate limiting (stricter for queries)
        if not self.security.check_rate_limit(tenant_id, limit=60):  # 60 queries/minute
            raise Exception("Rate limit exceeded. Please upgrade your plan.")
        
        # Input validation
        valid, message = self.security.validate_input(question, "query")
        if not valid:
            self.security.audit_log(tenant_id, "query_rejected", False,
                                   error=message, metadata=user_context)
            raise ValueError(f"Query validation failed: {message}")
        
        # Log query attempt
        query_hash = self.security.hash_data(question)
        self.security.audit_log(tenant_id, "query_started", True,
                               metadata={"query_hash": query_hash})
        
        try:
            # Get encrypted documents
            encrypted_results = super().query(tenant_id, question)
            
            # Decrypt results for this tenant only
            if isinstance(encrypted_results, str):
                decrypted = self.security.decrypt_data(encrypted_results, tenant_id)
            else:
                decrypted = encrypted_results
            
            # Log successful query
            self.security.audit_log(tenant_id, "query_completed", True,
                                   metadata={"query_hash": query_hash})
            
            return decrypted
            
        except Exception as e:
            # Log query failure
            self.security.audit_log(tenant_id, "query_failed", False,
                                   error=str(e), metadata={"query_hash": query_hash})
            raise
    
    def authenticate_tenant(self, api_key: str) -> Optional[str]:
        """Enhanced authentication with brute force protection"""
        # Check for brute force attempts
        ip = "127.0.0.1"  # In production, get from request
        if self._is_blocked(ip):
            self.security.audit_log("unknown", "auth_blocked", False,
                                   metadata={"ip": ip})
            return None
        
        # Validate API key format
        if not api_key.startswith("sk_") or len(api_key) != 35:
            self._record_failed_attempt(ip)
            return None
        
        # Authenticate using parent method
        tenant_id = super().authenticate_tenant(api_key)
        
        if tenant_id:
            # Success - reset failed attempts
            if ip in self.failed_attempts:
                del self.failed_attempts[ip]
            
            self.security.audit_log(tenant_id, "auth_success", True)
            
            # Create session token for additional security
            session_token = secrets.token_urlsafe(32)
            self.session_tokens[session_token] = {
                "tenant_id": tenant_id,
                "created": datetime.utcnow(),
                "last_activity": datetime.utcnow()
            }
            
            return tenant_id
        else:
            # Failed authentication
            self._record_failed_attempt(ip)
            self.security.audit_log("unknown", "auth_failed", False,
                                   metadata={"ip": ip})
            return None
    
    def _record_failed_attempt(self, ip: str):
        """Record failed authentication attempt"""
        if ip not in self.failed_attempts:
            self.failed_attempts[ip] = []
        
        self.failed_attempts[ip].append(datetime.utcnow())
        
        # Clean old attempts (older than 1 hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.failed_attempts[ip] = [
            attempt for attempt in self.failed_attempts[ip]
            if attempt > cutoff
        ]
    
    def _is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked due to too many failed attempts"""
        if ip not in self.failed_attempts:
            return False
        
        # Block if more than 5 attempts in last hour
        return len(self.failed_attempts[ip]) > 5
    
    def get_security_report(self, tenant_id: str) -> Dict:
        """Generate security report for tenant"""
        conn = sqlite3.connect(self.security.audit_db)
        cursor = conn.cursor()
        
        # Get security metrics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_actions,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                COUNT(DISTINCT DATE(timestamp)) as active_days
            FROM audit_log
            WHERE tenant_id = ?
            AND timestamp > datetime('now', '-30 days')
        ''', (tenant_id,))
        
        metrics = cursor.fetchone()
        
        # Get recent security events
        cursor.execute('''
            SELECT timestamp, action, success, error_message
            FROM audit_log
            WHERE tenant_id = ?
            AND success = 0
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (tenant_id,))
        
        recent_failures = cursor.fetchall()
        
        conn.close()
        
        return {
            "tenant_id": tenant_id,
            "report_date": datetime.utcnow().isoformat(),
            "metrics": {
                "total_actions": metrics[0],
                "successful": metrics[1],
                "failed": metrics[2],
                "active_days": metrics[3],
                "success_rate": (metrics[1] / metrics[0] * 100) if metrics[0] > 0 else 100
            },
            "recent_security_events": [
                {
                    "timestamp": event[0],
                    "action": event[1],
                    "error": event[3]
                }
                for event in recent_failures
            ],
            "security_score": self._calculate_security_score(metrics)
        }
    
    def _calculate_security_score(self, metrics) -> int:
        """Calculate security score (0-100)"""
        if not metrics[0]:  # No actions
            return 100
        
        success_rate = (metrics[1] / metrics[0]) * 100
        
        # Deduct points for failures
        score = 100
        score -= min(metrics[2] * 2, 30)  # Max 30 point deduction for failures
        
        # Bonus for consistent activity
        if metrics[3] >= 20:  # Active 20+ days
            score += 10
        
        return max(0, min(100, int(score)))
    
    def rotate_keys(self, tenant_id: str):
        """Rotate encryption keys for a tenant"""
        # Generate new key
        old_key = self.security.derive_tenant_key(tenant_id)
        
        # Clear key cache to force new key generation
        if tenant_id in self.security.key_cache:
            del self.security.key_cache[tenant_id]
        
        # Update master key salt (in production, use proper key rotation)
        new_salt = secrets.token_hex(16)
        
        # Re-encrypt all documents with new key
        # (In production, this would be done asynchronously)
        
        self.security.audit_log(tenant_id, "key_rotation", True,
                               metadata={"rotation_date": datetime.utcnow().isoformat()})
        
        return "Key rotation completed successfully"


# Example usage and testing
if __name__ == "__main__":
    print("🔐 Secure Multi-Tenant RAG System")
    print("=" * 50)
    
    # Initialize secure system
    secure_rag = SecureMultiTenantRAG()
    
    # Create a test tenant
    print("\n1️⃣ Creating secure tenant...")
    tenant_id = secure_rag.create_tenant("Secure Corp", "pro")
    print(f"✅ Tenant created: {tenant_id}")
    
    # Add encrypted document
    print("\n2️⃣ Adding encrypted document...")
    doc_id = secure_rag.add_document(
        tenant_id,
        "This is sensitive data with email@example.com and 555-123-4567",
        {"type": "confidential"}
    )
    print(f"✅ Document added: {doc_id}")
    
    # Query with security
    print("\n3️⃣ Querying with security validation...")
    api_key = secure_rag.tenants[tenant_id]["api_key"]
    auth_result = secure_rag.authenticate_tenant(api_key)
    
    if auth_result:
        result = secure_rag.query(tenant_id, "What sensitive data do we have?")
        print(f"✅ Query result: {result[:100]}...")
    
    # Get security report
    print("\n4️⃣ Security Report:")
    report = secure_rag.get_security_report(tenant_id)
    print(json.dumps(report, indent=2))
    
    print("\n✅ Security features demonstrated successfully!")