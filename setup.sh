#!/bin/bash
set -e  # Arrêter si erreur critique

echo "🚀 Setting up development environment..."

# Configure Git LFS
echo "🔧 Configuring Git LFS..."
if command -v git-lfs >/dev/null 2>&1; then
    echo "✅ Git LFS already installed via devcontainer feature"
    git lfs install >/dev/null 2>&1
else
    echo "⚠️  Git LFS not found, installing as fallback..."
    sudo apt update >/dev/null 2>&1
    sudo apt install -y git-lfs >/dev/null 2>&1
    git lfs install >/dev/null 2>&1
fi

# Install Python dependencies
echo "🐍 Installing Python packages..."
if [ -f "requirements.txt" ]; then
    pip3 install --user -r requirements.txt >/dev/null 2>&1 && \
        echo "✅ Python dependencies installed" || \
        echo "⚠️  Some dependencies may have failed to install"
else
    echo "⚠️  requirements.txt not found"
fi

# Make scripts executable
echo "🔑 Setting permissions..."
chmod +x *.sh 2>/dev/null && echo "✅ Scripts made executable" || true

echo "✅ Setup complete!"
echo "🎯 Run './start.sh' to launch the application"