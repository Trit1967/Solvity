#!/bin/bash
# RAGbot Deployment Script
# Leverages your existing codebase

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════╗"
echo "║       RAGbot Deployment Script        ║"
echo "║   Multi-tenant RAG Service for SMBs   ║"
echo "╚═══════════════════════════════════════╝"
echo -e "${NC}"

# Function to print colored messages
print_msg() {
    echo -e "${2}${1}${NC}"
}

# Check command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_msg "❌ $1 is not installed" "$RED"
        return 1
    else
        print_msg "✅ $1 is installed" "$GREEN"
        return 0
    fi
}

# Parse arguments
DEPLOY_TYPE=${1:-local}
DEPLOY_UI=${2:-no}

# Main deployment function
deploy() {
    print_msg "\n🚀 Starting RAGbot Deployment ($DEPLOY_TYPE mode)" "$YELLOW"
    
    # Step 1: Check prerequisites
    print_msg "\n📋 Checking prerequisites..." "$YELLOW"
    
    if [ "$DEPLOY_TYPE" == "docker" ]; then
        check_command docker || exit 1
        check_command docker-compose || check_command docker compose || exit 1
    else
        check_command python3 || exit 1
        check_command pip || check_command pip3 || exit 1
    fi
    
    # Step 2: Setup environment
    print_msg "\n🔧 Setting up environment..." "$YELLOW"
    
    if [ ! -f .env ]; then
        cp .env.ragbot .env
        print_msg "Created .env file - please edit with your settings" "$GREEN"
        
        # Generate secure secret key
        SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/change-this-to-a-secure-random-string-minimum-32-characters/$SECRET_KEY/" .env
        else
            sed -i "s/change-this-to-a-secure-random-string-minimum-32-characters/$SECRET_KEY/" .env
        fi
        print_msg "Generated secure SECRET_KEY" "$GREEN"
    fi
    
    # Create necessary directories
    mkdir -p data tenant_data uploads cache logs
    print_msg "Created data directories" "$GREEN"
    
    # Step 3: Deploy based on type
    case $DEPLOY_TYPE in
        local)
            deploy_local
            ;;
        docker)
            deploy_docker
            ;;
        docker-ui)
            deploy_docker_with_ui
            ;;
        vps)
            deploy_vps
            ;;
        *)
            print_msg "Invalid deployment type: $DEPLOY_TYPE" "$RED"
            print_msg "Options: local, docker, docker-ui, vps" "$YELLOW"
            exit 1
            ;;
    esac
}

# Local deployment
deploy_local() {
    print_msg "\n🏠 Deploying locally..." "$BLUE"
    
    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_msg "Starting Ollama..." "$YELLOW"
        
        # Install Ollama if needed
        if ! check_command ollama; then
            print_msg "Installing Ollama..." "$YELLOW"
            curl -fsSL https://ollama.com/install.sh | sh
        fi
        
        # Start Ollama
        ollama serve &
        OLLAMA_PID=$!
        sleep 5
        
        # Pull models
        print_msg "Pulling LLM models..." "$YELLOW"
        ollama pull llama3.2
        ollama pull mistral
    else
        print_msg "Ollama is already running" "$GREEN"
    fi
    
    # Install Python dependencies
    print_msg "Installing Python dependencies..." "$YELLOW"
    pip install -r requirements_ragbot.txt
    
    # Start the application
    print_msg "\n🎉 Starting RAGbot API..." "$GREEN"
    print_msg "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$BLUE"
    print_msg "API URL: http://localhost:8000" "$GREEN"
    print_msg "API Docs: http://localhost:8000/docs" "$GREEN"
    print_msg "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$BLUE"
    
    python ragbot_app.py
}

# Docker deployment
deploy_docker() {
    print_msg "\n🐳 Deploying with Docker..." "$BLUE"
    
    # Build and start containers
    print_msg "Building Docker images..." "$YELLOW"
    docker-compose -f docker-compose.ragbot.yml build
    
    print_msg "Starting containers..." "$YELLOW"
    docker-compose -f docker-compose.ragbot.yml up -d
    
    # Wait for services
    print_msg "Waiting for services to start..." "$YELLOW"
    sleep 10
    
    # Check health
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_msg "\n✅ RAGbot is running!" "$GREEN"
        print_msg "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$BLUE"
        print_msg "API URL: http://localhost:8000" "$GREEN"
        print_msg "API Docs: http://localhost:8000/docs" "$GREEN"
        print_msg "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$BLUE"
        print_msg "\nView logs: docker-compose -f docker-compose.ragbot.yml logs -f" "$YELLOW"
    else
        print_msg "⚠️ Service may still be starting. Check logs:" "$YELLOW"
        print_msg "docker-compose -f docker-compose.ragbot.yml logs -f" "$YELLOW"
    fi
}

# Docker with UI deployment
deploy_docker_with_ui() {
    print_msg "\n🐳 Deploying with Docker + UI..." "$BLUE"
    
    # Build and start all services including UI
    print_msg "Building Docker images..." "$YELLOW"
    docker-compose -f docker-compose.ragbot.yml --profile with-ui build
    
    print_msg "Starting containers with UI..." "$YELLOW"
    docker-compose -f docker-compose.ragbot.yml --profile with-ui up -d
    
    # Wait for services
    print_msg "Waiting for services to start..." "$YELLOW"
    sleep 15
    
    print_msg "\n✅ RAGbot with UI is running!" "$GREEN"
    print_msg "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$BLUE"
    print_msg "API URL: http://localhost:8000" "$GREEN"
    print_msg "API Docs: http://localhost:8000/docs" "$GREEN"
    print_msg "User Dashboard: http://localhost:7860" "$GREEN"
    print_msg "Admin Dashboard: http://localhost:7861" "$GREEN"
    print_msg "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$BLUE"
}

# VPS deployment
deploy_vps() {
    print_msg "\n☁️ VPS Deployment Guide" "$BLUE"
    print_msg "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$YELLOW"
    
    cat << EOF
1. SSH into your VPS:
   ssh root@your-server-ip

2. Clone your repository:
   git clone your-repo-url
   cd rag

3. Run this script on the VPS:
   ./deploy_ragbot.sh docker

4. Configure nginx (optional):
   sudo apt install nginx certbot
   # Configure reverse proxy to port 8000

5. Setup SSL (optional):
   certbot --nginx -d yourdomain.com

6. Monitor:
   docker-compose -f docker-compose.ragbot.yml logs -f
EOF
    
    print_msg "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$YELLOW"
}

# Stop services
stop_services() {
    print_msg "\n🛑 Stopping RAGbot services..." "$YELLOW"
    
    if [ -f docker-compose.ragbot.yml ]; then
        docker-compose -f docker-compose.ragbot.yml down
        print_msg "Docker services stopped" "$GREEN"
    fi
    
    # Kill local Ollama if running
    pkill ollama 2>/dev/null || true
    print_msg "Local services stopped" "$GREEN"
}

# Show status
show_status() {
    print_msg "\n📊 RAGbot Status" "$BLUE"
    print_msg "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$YELLOW"
    
    # Check API
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        HEALTH=$(curl -s http://localhost:8000/health)
        print_msg "✅ API is running" "$GREEN"
        echo "$HEALTH" | python3 -m json.tool
    else
        print_msg "❌ API is not running" "$RED"
    fi
    
    # Check Docker containers
    if command -v docker &> /dev/null; then
        print_msg "\nDocker containers:" "$YELLOW"
        docker ps --filter "name=ragbot"
    fi
    
    # Check Ollama
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_msg "\n✅ Ollama is running" "$GREEN"
    else
        print_msg "\n❌ Ollama is not running" "$RED"
    fi
}

# Show help
show_help() {
    print_msg "\n📖 RAGbot Deployment Script" "$BLUE"
    print_msg "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$YELLOW"
    echo "
Usage: ./deploy_ragbot.sh [command]

Commands:
  local       - Deploy locally with Python
  docker      - Deploy with Docker (API only)
  docker-ui   - Deploy with Docker + Gradio UIs
  vps        - Show VPS deployment guide
  stop       - Stop all services
  status     - Show service status
  help       - Show this help message

Examples:
  ./deploy_ragbot.sh local
  ./deploy_ragbot.sh docker
  ./deploy_ragbot.sh docker-ui
  ./deploy_ragbot.sh status
"
}

# Main execution
case ${1:-help} in
    local|docker|docker-ui|vps)
        deploy
        ;;
    stop)
        stop_services
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_msg "Unknown command: $1" "$RED"
        show_help
        exit 1
        ;;
esac