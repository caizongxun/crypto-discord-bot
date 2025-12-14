#!/bin/bash

# 🌐 GCP VM 一键部署脚本
# 使用方法: bash deploy_gcp_vm.sh

set -e

echo "============================================"
echo "  🌐 Crypto Discord Bot - GCP 部署"
echo "============================================"
echo ""

# 颜色
正常='\033[0;32m'  # 绿色警='\033[1;33m'  # 黃色
错误='\033[0;31m'  # 红色
不='\033[0m'         # 残位

# 棄下低斤新更新
echo -e "${\u8b66}\u226a更新\u7cfb\u7d71...${\u4e0d}"
sudo apt update && sudo apt upgrade -y

echo -e "${\u8b66}\u226a安裝 Python 3.11...${\u4e0d}"
sudo apt install -y python3.11 python3.11-venv python3.11-dev git curl htop tmux

echo -e "${\u8b66}\u226a設\u7f6e Python alias...${\u4e0d}"
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 克隆倉庫
echo ""
echo -e "${\u8b66}\u226a克\u9686倉庫...${\u4e0d}"
cd /home/$USER
if [ ! -d "crypto-discord-bot" ]; then
    git clone https://github.com/caizongxun/crypto-discord-bot.git
else
    echo -e "${\u6b63}\u2713 倉庫\u5df2\u5b58\u5728${\u4e0d}"
    cd crypto-discord-bot
    git pull origin main
    cd /home/$USER
fi

# 安\u88dd下佌管理器
echo ""
echo -e "${\u8b66}\u226a基建 Python 虛\u62ec環境...${\u4e0d}"
cd crypto-discord-bot
python -m venv venv
source venv/bin/activate

echo -e "${\u8b66}\u226a\u5b89\u88dd\u4f9d\u8cf4...${\u4e0d}"
pip install --upgrade pip
pip install -r requirements.txt

# 回趨\u7b80\u7701 .env
echo ""
echo -e "${\u6b63}\u2705 \u7d93\u5e78\u4f60\u5df2\u7d93\u4e86 .env ${\u4e0d}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${\u6b63}\u2705 .env \u5275\u5efa\u5b8c\u6210${\u4e0d}"
    echo ""
    echo -e "${\u9519}\u274c \u4f60\u9700\u8981\u7f16\u8f2f .env ${\u4e0d}"
    echo -e "${\u9519}\u274c \u6dfb\u52a0\u4f60\u7684 Discord Token${\u4e0d}"
    echo ""
    echo -e "${\u6b63}sudo nano /home/$USER/crypto-discord-bot/.env${\u4e0d}"
    echo ""
    exit 1
else
    echo -e "${\u6b63}\u2705 .env \u5df2\u5b58\u5728 - \u68c4下\u4f59\u91cd\u8b80\u53d6${\u4e0d}"
fi

# \u9078\u9805: \u6d4b\u8a66\u6a5f\u5668\u4eba
echo ""
echo -e "${\u8b66}\u226a\u6d4b\u8a66\u6a5f\u5668\u4eba\u7c97\u7a0b\u5ea6${\u4e0d}"
echo -e "${\u6b63}python bot_predictor.py${\u4e0d}"

# \u9078\u9805 A: \u4f7f\u7528 Systemd (推\u85a6)
echo ""
echo -e "${\u8b66}\u226a\u9078\u64c7\u90e8\u7f72\u6a21\u5f0f:${\u4e0d}"
echo -e "${\u6b63}1. Systemd (推\u85a6, \u81ea\u52d5\u91cd\u65b0\u555f\u52d5)${\u4e0d}"
echo -e "${\u6b63}2. Screen (\u7c97\u7a0b\u5ea6)${\u4e0d}"
echo -e "${\u6b63}3. Tmux (平\u8861\u6027\u80fd)${\u4e0d}"
echo -e "${\u6b63}4. Docker (\u7c97\u7a0b\u5ea6)${\u4e0d}"
echo ""

read -p "請\u9078\u64c7 (1-4) [\u9810\u8a2d: 1]: " choice
choice=${choice:-1}

if [ "$choice" = "1" ]; then
    echo -e "${\u8b66}\u226a\u8a2d\u7f6e Systemd...${\u4e0d}"
    
    # Bot \u670d\u52d9
    sudo tee /etc/systemd/system/crypto-bot.service > /dev/null <<EOF
[Unit]
Description=Crypto Discord Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/crypto-discord-bot
ExecStart=/home/$USER/crypto-discord-bot/venv/bin/python bot.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment="PATH=/home/$USER/crypto-discord-bot/venv/bin"

[Install]
WantedBy=multi-user.target
EOF
    
    # Dashboard \u670d\u52d9
    sudo tee /etc/systemd/system/crypto-dashboard.service > /dev/null <<EOF
[Unit]
Description=Crypto Dashboard
After=network.target crypto-bot.service
Wants=crypto-bot.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/crypto-discord-bot
ExecStart=/home/$USER/crypto-discord-bot/venv/bin/python dashboard.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment="PATH=/home/$USER/crypto-discord-bot/venv/bin"

[Install]
WantedBy=multi-user.target
EOF
    
    # \u5553\u6e96\u6240\u6709服\u52d9
    sudo systemctl daemon-reload
    sudo systemctl start crypto-bot
    sudo systemctl enable crypto-bot
    sudo systemctl start crypto-dashboard
    sudo systemctl enable crypto-dashboard
    
    echo -e "${\u6b63}\u2705 Systemd \u670d\u52d9\u5df2\u8a2d\u7f6e${\u4e0d}"
    echo ""
    echo -e "${\u6b63}\u5b50\u548c\u5f66:${\u4e0d}"
    echo -e "${\u6b63}  \u67e5\u770b\u72c0\u614b: sudo systemctl status crypto-bot${\u4e0d}"
    echo -e "${\u6b63}  \u67e5\u770b\u65e5\u8a8c: sudo journalctl -u crypto-bot -f${\u4e0d}"
    echo -e "${\u6b63}  \u505c\u6b62: sudo systemctl stop crypto-bot${\u4e0d}"
    echo -e "${\u6b63}  \u91cd\u8a93\u95dc\u5df2: sudo systemctl restart crypto-bot${\u4e0d}"
    
elif [ "$choice" = "2" ]; then
    echo -e "${\u8b66}\u226a\u4f7f\u7528 Screen...${\u4e0d}"
    screen -dmS bot bash -c "cd /home/$USER/crypto-discord-bot && source venv/bin/activate && python bot.py"
    echo -e "${\u6b63}\u2705 Bot \u5df2\u5728 Screen 'bot' \u4e2d\u555f\u52d5${\u4e0d}"
    echo -e "${\u6b63}  \u91cd\u65b0\u9023\u63a5: screen -r bot${\u4e0d}"
    
elif [ "$choice" = "3" ]; then
    echo -e "${\u8b66}\u226a\u4f7f\u7528 Tmux...${\u4e0d}"
    tmux new-session -d -s bot -x 200 -y 50 "cd /home/$USER/crypto-discord-bot && source venv/bin/activate && python bot.py"
    echo -e "${\u6b63}\u2705 Bot \u5df2\u5728 Tmux 'bot' \u4e2d\u555f\u52d5${\u4e0d}"
    echo -e "${\u6b63}  \u91cd\u65b0\u9023\u63a5: tmux attach -t bot${\u4e0d}"
    
elif [ "$choice" = "4" ]; then
    echo -e "${\u8b66}\u226a\u4f7f\u7528 Docker...${\u4e0d}"
    docker-compose up -d
    echo -e "${\u6b63}\u2705 Docker \u5bb9\u5668\u5df2\u555f\u52d5${\u4e0d}"
    echo -e "${\u6b63}  \u67e5\u770b\u65e5\u8a8c: docker-compose logs -f${\u4e0d}"
fi

# \u6c42\u4e0d\u4f1a\u7bc4
echo ""
echo "============================================"
echo -e "${\u6b63}\u2705 \u90e8\u7f72\u5b8c\u6210!${\u4e0d}"
echo "============================================"
echo ""
echo -e "${\u6b63}\u4e0b\u4e00\u6b65:${\u4e0d}"
echo -e "${\u6b63}1. \u7de8\u8f2f .env \u4e26\u6dfb\u52a0 Discord Token:${\u4e0d}"
echo -e "${\u6b63}   nano /home/$USER/crypto-discord-bot/.env${\u4e0d}"
echo ""
echo -e "${\u6b63}2. \u6e2c\u8a66\u6a5f\u5668\u4eba:${\u4e0d}"
echo -e "${\u6b63}   .models${\u4e0d}"
echo -e "${\u6b63}   .predict${\u4e0d}"
echo -e "${\u6b63}   .signal${\u4e0d}"
echo ""
echo -e "${\u6b63}3. \u8a73\u7d30\u90e8\u7f72\u8a73\u8a59\u67e5\u8a93 GCP_DEPLOYMENT.md${\u4e0d}"
echo ""
