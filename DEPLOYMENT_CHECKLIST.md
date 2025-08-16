# 🚀 RAGbot Deployment Checklist

## Pre-Deployment (Today)

### Local Testing ✅
- [ ] Run `./setup_secure.sh`
- [ ] Test `python ragbot_secure_app.py`
- [ ] Create test tenant via API
- [ ] Upload test document
- [ ] Query and verify encryption
- [ ] Check audit logs

### Security Verification
- [ ] Verify .env has strong keys
- [ ] Test rate limiting works
- [ ] Confirm audit logging
- [ ] Check encryption/decryption
- [ ] Test authentication flow

---

## Deployment Day (Tomorrow)

### 1. Get VPS (30 mins)
Choose one:

#### Option A: Hetzner (Recommended - Cheapest)
```bash
# CX21: 2 vCPU, 4GB RAM, 40GB SSD
# €5.83/month (~$6.50)
# Location: Germany/Finland
```
1. Go to https://www.hetzner.com/cloud
2. Create account
3. Launch CX21 instance with Ubuntu 22.04
4. Note the IP address

#### Option B: DigitalOcean
```bash
# Basic Droplet: 2 vCPU, 4GB RAM
# $24/month
# Multiple locations
```
1. Go to https://www.digitalocean.com
2. Create droplet with Ubuntu 22.04
3. Add SSH key
4. Note the IP address

#### Option C: Linode
```bash
# Shared CPU: 2GB RAM
# $12/month
```

### 2. Initial Server Setup (20 mins)
```bash
# SSH into your server
ssh root@YOUR_SERVER_IP

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose
apt install docker-compose -y

# Setup firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8000/tcp
ufw --force enable

# Create app user
adduser ragbot
usermod -aG docker ragbot
```

### 3. Deploy Application (15 mins)
```bash
# As ragbot user
su - ragbot

# Clone your repository
git clone YOUR_REPO_URL
cd rag

# Setup environment
cp .env.ragbot .env
nano .env  # Add your secure keys

# Deploy with Docker
./deploy_ragbot.sh docker

# Or use secure version
docker-compose -f docker-compose.ragbot.yml up -d
```

### 4. Setup Domain & SSL (Optional - 20 mins)
```bash
# Install nginx and certbot
apt install nginx certbot python3-certbot-nginx -y

# Configure nginx
nano /etc/nginx/sites-available/ragbot

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable site
ln -s /etc/nginx/sites-available/ragbot /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx

# Get SSL certificate
certbot --nginx -d your-domain.com
```

### 5. Setup Monitoring (10 mins)
```bash
# Install basic monitoring
apt install htop ncdu -y

# Setup UptimeRobot (free)
# 1. Go to https://uptimerobot.com
# 2. Add monitor for http://YOUR_IP:8000/health

# Check logs
docker-compose logs -f ragbot
```

### 6. Automated Backups (15 mins)
```bash
# Create backup script
nano /home/ragbot/backup.sh

#!/bin/bash
# Daily backup script
BACKUP_DIR="/home/ragbot/backups"
DATE=$(date +%Y%m%d)

# Create backup
cd /home/ragbot/rag
tar -czf $BACKUP_DIR/ragbot-$DATE.tar.gz data/ tenant_data/ tenants.json

# Encrypt backup
gpg -c $BACKUP_DIR/ragbot-$DATE.tar.gz
rm $BACKUP_DIR/ragbot-$DATE.tar.gz

# Upload to S3/B2 (optional)
# aws s3 cp $BACKUP_DIR/ragbot-$DATE.tar.gz.gpg s3://your-bucket/

# Keep only last 7 days
find $BACKUP_DIR -name "*.gpg" -mtime +7 -delete

# Make executable
chmod +x /home/ragbot/backup.sh

# Add to crontab
crontab -e
# Add: 0 2 * * * /home/ragbot/backup.sh
```

---

## Post-Deployment (Day 2)

### Verification
- [ ] API accessible at http://YOUR_IP:8000
- [ ] Health check returns OK
- [ ] Can create account via API
- [ ] Can upload document
- [ ] Can query successfully
- [ ] Audit logs working
- [ ] Backups running

### Security Hardening
- [ ] Change SSH port
- [ ] Disable root login
- [ ] Setup fail2ban
- [ ] Configure log rotation
- [ ] Review firewall rules

### Performance Tuning
- [ ] Monitor CPU/Memory usage
- [ ] Optimize Docker resources
- [ ] Configure swap if needed
- [ ] Setup Redis cache (Phase 2)

---

## Launch Week Tasks

### Day 1-2: Infrastructure
- [x] Deploy to VPS
- [x] Setup monitoring
- [x] Configure backups
- [ ] Test everything

### Day 3-4: Beta Users
- [ ] Create onboarding guide
- [ ] Invite 5 beta users
- [ ] Provide API keys
- [ ] Gather feedback

### Day 5-7: Marketing Prep
- [ ] Create landing page
- [ ] Write launch post
- [ ] Prepare ProductHunt
- [ ] Plan HackerNews post

---

## Quick Commands Reference

### Check Status
```bash
# SSH to server
ssh ragbot@YOUR_IP

# Check containers
docker ps

# View logs
docker-compose logs -f ragbot

# Check disk space
df -h

# Monitor resources
htop
```

### Troubleshooting
```bash
# Restart services
docker-compose down
docker-compose up -d

# Check Ollama
docker exec -it ragbot-ollama ollama list

# View audit logs
sqlite3 data/audit_trail.db "SELECT * FROM audit_log ORDER BY id DESC LIMIT 10;"

# Check security logs
tail -f logs/security_incidents.jsonl
```

### Backup & Restore
```bash
# Manual backup
./backup.sh

# Restore from backup
gpg -d backup-20240101.tar.gz.gpg | tar -xzf -
```

---

## Support Contacts

### Infrastructure
- Hetzner: https://console.hetzner.cloud
- DigitalOcean: https://cloud.digitalocean.com
- UptimeRobot: https://uptimerobot.com

### Emergency
- Create incident: `echo "INCIDENT" >> logs/incidents.log`
- Alert webhook: Configure in .env
- Backup contact: your-email@domain.com

---

## Success Metrics

### Day 1
- ✅ Deployed successfully
- ✅ API responding
- ✅ Monitoring active

### Week 1
- [ ] 5 beta users onboarded
- [ ] 100+ API calls
- [ ] Zero security incidents
- [ ] <2s average response time

### Month 1
- [ ] 50 paying customers
- [ ] 99.9% uptime
- [ ] Positive user feedback
- [ ] Ready to scale

---

**Remember: Start small, secure from day one, scale when ready!** 🚀🔐