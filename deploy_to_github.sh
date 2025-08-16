#!/bin/bash
# Deploy RAGbot to GitHub with all features enabled

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Deploying RAGbot to GitHub${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if git is initialized
if [ ! -d .git ]; then
    echo -e "${YELLOW}Initializing git repository...${NC}"
    git init
    git branch -M main
fi

# Check if GitHub CLI is installed
if command -v gh &> /dev/null; then
    echo -e "${GREEN}✅ GitHub CLI found${NC}"
    USE_GH=true
else
    echo -e "${YELLOW}⚠️  GitHub CLI not found. Using standard git.${NC}"
    echo "   Install with: https://cli.github.com/"
    USE_GH=false
fi

# Add all files
echo -e "${YELLOW}Adding files to git...${NC}"
git add .
git commit -m "🚀 Initial RAGbot deployment" || true

# Create GitHub repository
if [ "$USE_GH" = true ]; then
    echo -e "${YELLOW}Creating GitHub repository...${NC}"
    
    # Prompt for repo name
    read -p "Enter repository name (default: ragbot): " REPO_NAME
    REPO_NAME=${REPO_NAME:-ragbot}
    
    # Create repo
    gh repo create $REPO_NAME --public --source=. --remote=origin --push \
        --description "Secure Multi-Tenant RAG Service for SMBs" \
        --homepage "https://github.com/$USER/$REPO_NAME" || {
        echo -e "${YELLOW}Repository might already exist. Pushing to existing...${NC}"
        git remote add origin "https://github.com/$USER/$REPO_NAME.git" 2>/dev/null || true
        git push -u origin main
    }
    
    # Enable GitHub Pages
    echo -e "${YELLOW}Enabling GitHub Pages...${NC}"
    gh repo edit --enable-wiki --enable-issues --enable-projects || true
    
    # Create initial issues
    echo -e "${YELLOW}Creating starter issues...${NC}"
    gh issue create --title "📋 Setup development environment" \
        --body "- [ ] Clone repository
- [ ] Run setup_secure.sh
- [ ] Test local deployment
- [ ] Verify all endpoints work" || true
    
    gh issue create --title "🚀 Deploy to production" \
        --body "- [ ] Choose hosting provider
- [ ] Setup domain
- [ ] Configure SSL
- [ ] Deploy application
- [ ] Setup monitoring" || true
    
    # Setup secrets for Actions (if you have them)
    echo -e "${YELLOW}Setting up GitHub Secrets...${NC}"
    echo "Add these secrets in GitHub Settings > Secrets:"
    echo "  - DOCKER_REGISTRY_TOKEN"
    echo "  - DEPLOY_KEY"
    
    GITHUB_URL="https://github.com/$USER/$REPO_NAME"
    
else
    # Manual setup
    echo -e "${YELLOW}Please complete GitHub setup:${NC}"
    echo "1. Go to https://github.com/new"
    echo "2. Create a repository named 'ragbot'"
    echo "3. Run these commands:"
    echo ""
    echo "   git remote add origin https://github.com/YOUR_USERNAME/ragbot.git"
    echo "   git push -u origin main"
    echo ""
    read -p "Enter your GitHub username: " GH_USER
    git remote add origin "https://github.com/$GH_USER/ragbot.git" 2>/dev/null || true
    
    echo -e "${YELLOW}Pushing to GitHub...${NC}"
    git push -u origin main
    
    GITHUB_URL="https://github.com/$GH_USER/ragbot"
fi

# Create a simple GitHub Pages site
echo -e "${YELLOW}Creating GitHub Pages documentation...${NC}"
mkdir -p docs
cat > docs/index.md << 'EOF'
# RAGbot Documentation

## Quick Start
1. Click "Use this template" on GitHub
2. Open in Codespaces
3. Run `./setup_secure.sh`
4. Start with `./run_ragbot.sh`

## Features
- 🔐 Enterprise security
- 🏢 Multi-tenancy
- 📚 Document processing
- 🤖 Local LLM with Ollama

## API Documentation
Visit `/docs` endpoint when running

## Support
Create an issue on GitHub for help
EOF

# Push everything
git add .
git commit -m "📚 Add documentation" || true
git push

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Successfully deployed to GitHub!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}📦 Repository:${NC} $GITHUB_URL"
echo -e "${BLUE}🚀 Codespaces:${NC} $GITHUB_URL/codespaces"
echo -e "${BLUE}📖 Actions:${NC} $GITHUB_URL/actions"
echo -e "${BLUE}🐛 Issues:${NC} $GITHUB_URL/issues"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Open in GitHub Codespaces for free development"
echo "2. Enable GitHub Pages in Settings > Pages"
echo "3. Add secrets for CI/CD in Settings > Secrets"
echo "4. Invite collaborators in Settings > Collaborators"
echo ""
echo -e "${GREEN}🎉 Your RAGbot is now on GitHub!${NC}"