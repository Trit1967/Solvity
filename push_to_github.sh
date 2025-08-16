#!/bin/bash
# Push RAGbot to your existing GitHub repository

echo "🚀 Pushing RAGbot to GitHub"
echo "=========================="

# Get GitHub username
read -p "Enter your GitHub username: " GH_USER
read -p "Enter your repository name (default: rag): " REPO_NAME
REPO_NAME=${REPO_NAME:-rag}

# Add remote
echo "Adding GitHub remote..."
git remote add origin "https://github.com/$GH_USER/$REPO_NAME.git" 2>/dev/null || {
    echo "Remote already exists, updating..."
    git remote set-url origin "https://github.com/$GH_USER/$REPO_NAME.git"
}

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main || {
    echo ""
    echo "If this is your first push, you may need to:"
    echo "1. Create a personal access token at: https://github.com/settings/tokens"
    echo "2. Use the token as your password when prompted"
    echo ""
    echo "Or use SSH:"
    echo "git remote set-url origin git@github.com:$GH_USER/$REPO_NAME.git"
}

echo ""
echo "✅ Done! Your repository is at:"
echo "https://github.com/$GH_USER/$REPO_NAME"
echo ""
echo "🚀 To use GitHub Codespaces (FREE 60 hours/month):"
echo "1. Go to: https://github.com/$GH_USER/$REPO_NAME"
echo "2. Click 'Code' → 'Codespaces' → 'Create codespace'"
echo "3. Everything runs in your browser!"