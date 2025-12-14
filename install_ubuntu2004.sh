#!/bin/bash

# 🐧 Ubuntu 20.04 LTS 一键安裝脚本
# 使用方法: bash install_ubuntu2004.sh

set -e

echo "================================================"
echo "  Ubuntu 20.04 LTS - Crypto Bot 粗全安裝"
echo "================================================"
echo ""

# 颜色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}≪步驟 1: 更新系統...${NC}"
sudo apt update && sudo apt upgrade -y
echo -e "${GREEN}✓ 完成${NC}"
echo ""

echo -e "${YELLOW}≪步驟 2: 安裝細他基礎工具...${NC}"
sudo apt install -y software-properties-common curl wget git htop tmux python3-distutils
echo -e "${GREEN}✓ 完成${NC}"
echo ""

echo -e "${YELLOW}≪步驟 3: 添加 Deadsnakes PPA...${NC}"
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
echo -e "${GREEN}✓ 完成${NC}"
echo ""

echo -e "${YELLOW}≪步驟 4: 安裝 Python 3.11...${NC}"
sudo apt install -y python3.11 python3.11-venv python3.11-dev
echo -e "${GREEN}✓ 完成${NC}"
echo ""

echo -e "${YELLOW}≪步驟 5: 設置 Python 3.11 為默認...${NC}"
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
echo -e "${GREEN}✓ 完成${NC}"
echo ""

echo -e "${YELLOW}≪步驟 6: 驗證 Python 3.11...${NC}
python --version
echo -e "${GREEN}✓ 完成${NC}"
echo ""

echo -e "${YELLOW}≪步驟 7: 升級 pip...${NC}"
python3.11 -m pip install --upgrade pip
echo -e "${GREEN}✓ 完成${NC}"
echo ""

echo -e "${YELLOW}≪步驟 8: 克隆倉庫...${NC}"
cd /home/$USER
if [ ! -d "crypto-discord-bot" ]; then
    git clone https://github.com/caizongxun/crypto-discord-bot.git
    echo -e "${GREEN}✓ 克隆完成${NC}"
else
    echo -e "${GREEN}✓ 倉庫已存在${NC}"
    cd crypto-discord-bot
    git pull origin main
    cd /home/$USER
fi
echo ""

cd crypto-discord-bot

echo -e "${YELLOW}≪步驟 9: 創建虛擬環境...${NC}"
python3.11 -m venv venv
source venv/bin/activate
echo -e "${GREEN}✓ 完成${NC}"
echo ""

echo -e "${YELLOW}≪步驟 10: 安裝 Python 依賴...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ 完成${NC}"
echo ""

echo -e "${YELLOW}≪步驟 11: 設置 .env...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${RED}✗ 請編輯 .env 檔並添加 Discord Token${NC}"
    echo -e "${GREEN}nano /home/$USER/crypto-discord-bot/.env${NC}"
    echo ""
    echo -e "${RED}✗ 詳細可詳詙查 UBUNTU_2004_GUIDE.md${NC}"
    exit 1
else
    echo -e "${GREEN}✓ .env 已存在${NC}"
fi
echo ""

echo "================================================"
echo -e "${GREEN}✅ Ubuntu 20.04 安裝完成!${NC}"
echo "================================================"
echo ""
echo -e "${GREEN}下一步:${NC}"
echo ""
echo -e "${YELLOW}1. 編輯 .env 檔:${NC}"
echo -e "${GREEN}   nano /home/$USER/crypto-discord-bot/.env${NC}"
echo ""
echo -e "${YELLOW}2. 選擇部署方式 (Systemd 推薦):${NC}"
echo -e "${GREEN}   查看 UBUNTU_2004_GUIDE.md 的 '部署方式' 部分${NC}"
echo ""
echo -e "${YELLOW}3. 啟動機器人:${NC}"
echo -e "${GREEN}   sudo systemctl start crypto-bot${NC}"
echo ""
echo -e "${YELLOW}4. 查看日誌:${NC}"
echo -e "${GREEN}   sudo journalctl -u crypto-bot -f${NC}"
echo ""
