#!/bin/bash
# Setup GitHub authentication for pushing

echo "🔐 GitHub Authentication Setup"
echo "=============================="
echo ""
echo "Choose an authentication method:"
echo ""
echo "1. Personal Access Token (Recommended)"
echo "2. SSH Key"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "📝 Setting up Personal Access Token:"
    echo "1. Go to: https://github.com/settings/tokens/new"
    echo "2. Give it a name: 'RAGbot Deploy'"
    echo "3. Select scopes: 'repo' (all repo permissions)"
    echo "4. Click 'Generate token'"
    echo "5. Copy the token (starts with ghp_)"
    echo ""
    read -p "Paste your token here: " TOKEN
    
    # Configure git to use token
    git remote set-url origin https://$TOKEN@github.com/trit1967/Solvity.git
    
    echo "✅ Token configured! Pushing to GitHub..."
    git push -u origin main
    
elif [ "$choice" = "2" ]; then
    echo ""
    echo "🔑 Setting up SSH Key:"
    
    # Check if SSH key exists
    if [ ! -f ~/.ssh/id_rsa ]; then
        echo "Generating SSH key..."
        ssh-keygen -t rsa -b 4096 -C "trit1967@example.com" -f ~/.ssh/id_rsa -N ""
    fi
    
    echo ""
    echo "📋 Copy this SSH key and add it to GitHub:"
    echo "----------------------------------------"
    cat ~/.ssh/id_rsa.pub
    echo "----------------------------------------"
    echo ""
    echo "1. Go to: https://github.com/settings/keys"
    echo "2. Click 'New SSH key'"
    echo "3. Paste the key above"
    echo "4. Click 'Add SSH key'"
    echo ""
    read -p "Press Enter after adding the key to GitHub..."
    
    # Change remote to SSH
    git remote set-url origin git@github.com:trit1967/Solvity.git
    
    echo "✅ SSH configured! Pushing to GitHub..."
    git push -u origin main
fi

echo ""
echo "🎉 Your code is now on GitHub!"
echo "Repository: https://github.com/trit1967/Solvity"
echo ""
echo "🚀 Next steps:"
echo "1. Open in Codespaces: https://github.com/trit1967/Solvity/codespaces"
echo "2. Check Actions: https://github.com/trit1967/Solvity/actions"
echo "3. Enable Pages: https://github.com/trit1967/Solvity/settings/pages"