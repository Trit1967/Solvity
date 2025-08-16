#!/bin/bash
# RAGbot Deployment Script
# Supports local, VPS, and cloud deployments

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ragbot"
DOMAIN=""
EMAIL=""
DEPLOYMENT_TYPE=""

# Function to print colored output
print_color() {
    printf "${2}${1}${NC}\n"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./deploy.sh --type [local|vps|docker] [--domain yourdomain.com] [--email your@email.com]"
            echo ""
            echo "Options:"
            echo "  --type    Deployment type: local, vps, or docker"
            echo "  --domain  Your domain name (for VPS deployment)"
            echo "  --email   Your email (for SSL certificates)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate deployment type
if [ -z "$DEPLOYMENT_TYPE" ]; then
    print_color "Error: Deployment type is required. Use --type [local|vps|docker]" "$RED"
    exit 1
fi

# Function to setup environment
setup_environment() {
    print_color "Setting up environment..." "$YELLOW"
    
    # Copy .env.example to .env if it doesn't exist
    if [ ! -f .env ]; then
        cp .env.example .env
        print_color "Created .env file. Please edit it with your configuration." "$GREEN"
        
        # Generate a secure secret key
        SECRET_KEY=$(openssl rand -hex 32)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s/your-secret-key-change-in-production-minimum-32-chars/$SECRET_KEY/" .env
        else
            # Linux
            sed -i "s/your-secret-key-change-in-production-minimum-32-chars/$SECRET_KEY/" .env
        fi
        print_color "Generated secure SECRET_KEY" "$GREEN"
    fi
    
    # Create necessary directories
    mkdir -p data uploads cache logs
    print_color "Created data directories" "$GREEN"
}

# Function for local deployment
deploy_local() {
    print_color "Starting local deployment..." "$YELLOW"
    
    setup_environment
    
    # Check Python version
    if ! command_exists python3; then
        print_color "Python 3 is required but not installed." "$RED"
        exit 1
    fi
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_color "Creating virtual environment..." "$YELLOW"
        python3 -m venv venv
    fi
    
    # Activate virtual environment and install dependencies
    source venv/bin/activate
    print_color "Installing dependencies..." "$YELLOW"
    pip install --upgrade pip
    pip install -r requirements_api.txt
    
    # Check if Ollama is installed
    if ! command_exists ollama; then
        print_color "Installing Ollama..." "$YELLOW"
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    
    # Start Ollama and pull model
    print_color "Starting Ollama and pulling llama3.2 model..." "$YELLOW"
    ollama serve &
    OLLAMA_PID=$!
    sleep 5
    ollama pull llama3.2
    
    # Start the application
    print_color "Starting RAGbot application..." "$GREEN"
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
}

# Function for Docker deployment
deploy_docker() {
    print_color "Starting Docker deployment..." "$YELLOW"
    
    setup_environment
    
    # Check if Docker is installed
    if ! command_exists docker; then
        print_color "Installing Docker..." "$YELLOW"
        curl -fsSL https://get.docker.com | sh
    fi
    
    # Check if Docker Compose is installed
    if ! command_exists docker-compose; then
        print_color "Installing Docker Compose..." "$YELLOW"
        sudo apt-get update
        sudo apt-get install -y docker-compose
    fi
    
    # Build and start containers
    print_color "Building Docker images..." "$YELLOW"
    docker-compose build
    
    print_color "Starting Docker containers..." "$GREEN"
    docker-compose up -d
    
    # Wait for services to be ready
    print_color "Waiting for services to be ready..." "$YELLOW"
    sleep 10
    
    # Check health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_color "✅ RAGbot is running at http://localhost:8000" "$GREEN"
    else
        print_color "⚠️ RAGbot may still be starting up. Check logs with: docker-compose logs -f" "$YELLOW"
    fi
}

# Function for VPS deployment
deploy_vps() {
    print_color "Starting VPS deployment..." "$YELLOW"
    
    if [ -z "$DOMAIN" ]; then
        print_color "Error: Domain is required for VPS deployment. Use --domain yourdomain.com" "$RED"
        exit 1
    fi
    
    setup_environment
    
    # Update system
    print_color "Updating system packages..." "$YELLOW"
    sudo apt-get update
    sudo apt-get upgrade -y
    
    # Install required packages
    print_color "Installing required packages..." "$YELLOW"
    sudo apt-get install -y \
        docker.io \
        docker-compose \
        nginx \
        certbot \
        python3-certbot-nginx \
        ufw
    
    # Configure firewall
    print_color "Configuring firewall..." "$YELLOW"
    sudo ufw allow 22/tcp
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    sudo ufw allow 8000/tcp
    sudo ufw --force enable
    
    # Start Docker deployment
    deploy_docker
    
    # Configure Nginx
    print_color "Configuring Nginx..." "$YELLOW"
    sudo tee /etc/nginx/sites-available/ragbot > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
    
    sudo ln -sf /etc/nginx/sites-available/ragbot /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t
    sudo systemctl restart nginx
    
    # Setup SSL with Let's Encrypt
    if [ ! -z "$EMAIL" ]; then
        print_color "Setting up SSL certificate..." "$YELLOW"
        sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email $EMAIL
    else
        print_color "Skipping SSL setup. Run 'sudo certbot --nginx -d $DOMAIN' to set it up later." "$YELLOW"
    fi
    
    # Create systemd service for auto-start
    print_color "Creating systemd service..." "$YELLOW"
    sudo tee /etc/systemd/system/ragbot.service > /dev/null <<EOF
[Unit]
Description=RAGbot Docker Compose Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable ragbot
    
    print_color "✅ VPS deployment complete!" "$GREEN"
    print_color "RAGbot is available at: http://$DOMAIN" "$GREEN"
    if [ ! -z "$EMAIL" ]; then
        print_color "SSL is configured. Access via: https://$DOMAIN" "$GREEN"
    fi
}

# Function to show deployment status
show_status() {
    print_color "\n📊 Deployment Status:" "$YELLOW"
    
    # Check if running locally
    if pgrep -f "uvicorn app:app" > /dev/null; then
        print_color "✅ Local deployment is running" "$GREEN"
    fi
    
    # Check Docker containers
    if command_exists docker; then
        if docker ps | grep -q ragbot; then
            print_color "✅ Docker deployment is running" "$GREEN"
            docker ps --filter "name=ragbot" --filter "name=ollama"
        fi
    fi
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_color "✅ API is healthy" "$GREEN"
    else
        print_color "⚠️ API is not responding" "$YELLOW"
    fi
}

# Main execution
print_color "\n🚀 RAGbot Deployment Script\n" "$GREEN"

case $DEPLOYMENT_TYPE in
    local)
        deploy_local
        ;;
    docker)
        deploy_docker
        ;;
    vps)
        deploy_vps
        ;;
    *)
        print_color "Invalid deployment type: $DEPLOYMENT_TYPE" "$RED"
        print_color "Use: local, docker, or vps" "$YELLOW"
        exit 1
        ;;
esac

show_status

print_color "\n✅ Deployment complete!" "$GREEN"
print_color "📖 Check the README.md for next steps and usage instructions." "$YELLOW"