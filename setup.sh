#!/bin/bash
echo "🚀 Setting up development environment..."

# Install Git LFS
echo "📦 Installing Git LFS..."
sudo apt update >/dev/null 2>&1
sudo apt install -y git-lfs >/dev/null 2>&1

# Configure Git LFS
echo "🔧 Configuring Git LFS..."
git lfs install

# Install Python dependencies
echo "🐍 Installing Python packages..."
pip3 install --user -r app/requirements.txt >/dev/null 2>&1
pip3 install --user streamlit >/dev/null 2>&1

# Make scripts executable
echo "🔑 Setting permissions..."
[ -f start.sh ] && chmod +x start.sh
chmod +x setup.sh

echo "✅ Setup complete!"
echo "🎯 Run './start.sh' to launch the application"