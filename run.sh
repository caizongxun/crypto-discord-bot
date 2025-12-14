#!/bin/bash

# Crypto Discord Bot - All-in-one Launcher Script
# Starts both Discord bot and web dashboard

set -e

echo "🚀 Crypto Discord Bot Launcher"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "${YELLOW}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install/upgrade requirements
echo -e "${YELLOW}Checking dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found${NC}"
    echo -e "${YELLOW}Creating .env from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env with your Discord token!${NC}"
    cat .env
    exit 1
fi

# Check Discord token
if ! grep -q "^DISCORD_TOKEN=" .env || grep "^DISCORD_TOKEN=$" .env > /dev/null 2>&1; then
    echo -e "${RED}❌ DISCORD_TOKEN not set in .env${NC}"
    exit 1
fi

echo -e "${GREEN}✓ .env configuration valid${NC}"

# Start services
echo -e "\n${GREEN}Starting services...${NC}"
echo "================================"

# Start Discord bot in background
echo -e "${YELLOW}Starting Discord bot...${NC}"
python3 bot.py &
BOT_PID=$!
echo -e "${GREEN}✓ Discord bot PID: $BOT_PID${NC}"

# Wait a moment
sleep 2

# Start Flask dashboard in background (optional)
if [ "$1" == "--with-dashboard" ] || [ "$1" == "-d" ]; then
    echo -e "${YELLOW}Starting web dashboard...${NC}"
    DASHBOARD_PORT=$(grep "DASHBOARD_PORT" .env | cut -d '=' -f2 || echo "5000")
    python3 dashboard.py &
    DASHBOARD_PID=$!
    echo -e "${GREEN}✓ Dashboard running on http://localhost:$DASHBOARD_PORT${NC}"
    echo -e "${GREEN}✓ Dashboard PID: $DASHBOARD_PID${NC}"
else
    DASHBOARD_PID=""
    echo -e "${YELLOW}To start dashboard, run: $0 --with-dashboard${NC}"
fi

echo "================================"
echo -e "${GREEN}✅ All services started!${NC}"
echo -e "${YELLOW}Bot logs above. Press Ctrl+C to stop all services.${NC}"

# Trap Ctrl+C
trap 'cleanup' INT

cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    kill $BOT_PID 2>/dev/null || true
    if [ -n "$DASHBOARD_PID" ]; then
        kill $DASHBOARD_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}✓ Services stopped${NC}"
    exit 0
}

# Wait for processes
wait
