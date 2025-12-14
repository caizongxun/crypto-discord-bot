#!/bin/bash

# Crypto Discord Bot - VM Deployment Script
# This script automates the deployment process on a Linux VM

set -e

echo "========================================="
echo "🤖 Crypto Discord Bot - VM Deployment"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ️${NC} $1"
}

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found. Please install Python 3.8+"
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python $PYTHON_VERSION found"
echo ""

# Create virtual environment
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

pip install -r requirements.txt
print_success "Dependencies installed"
echo ""

# Create .env if not exists
echo "Checking configuration..."
if [ ! -f ".env" ]; then
    print_info ".env file not found"
    cp .env.example .env
    print_success ".env created from template"
    print_error "IMPORTANT: Edit .env with your tokens before running the bot"
    print_info "Run: nano .env"
    echo ""
    exit 0
else
    print_success ".env file found"
fi
echo ""

# Verify .env has tokens
echo "Verifying configuration..."
if grep -q "your_discord_bot_token_here" .env; then
    print_error "DISCORD_BOT_TOKEN not configured in .env"
    print_info "Please edit .env and add your Discord bot token"
    exit 1
fi

if grep -q "your_discord_channel_id_here" .env; then
    print_error "DISCORD_CHANNEL_ID not configured in .env"
    print_info "Please edit .env and add your Discord channel ID"
    exit 1
fi

print_success "Configuration verified"
echo ""

# Test imports
echo "Testing imports..."
python3 -c "import discord; import torch; from huggingface_hub import hf_hub_download" 2>/dev/null
print_success "All imports successful"
echo ""

# Create logs directory
mkdir -p logs
print_success "Logs directory ready"
echo ""

# Summary
echo "========================================="
echo "✅ Deployment successful!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Make sure .env is configured with your tokens"
echo "2. Run the bot: python bot.py"
echo "3. Or use systemd: sudo systemctl start crypto-discord-bot"
echo ""
echo "For background execution:"
echo "  - systemd (recommended): See README.md"
echo "  - screen: screen -S crypto-bot && python bot.py"
echo "  - nohup: nohup python bot.py > bot.log 2>&1 &"
echo ""
echo "View logs:"
echo "  tail -f bot.log"
echo ""
