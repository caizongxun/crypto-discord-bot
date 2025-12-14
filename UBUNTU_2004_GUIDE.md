# 🐧 Ubuntu 20.04 LTS 專用部署指南

## ⚠️ 重要注意

Ubuntu 20.04 LTS 內置 **Python 3.8**，我們需要升級到 **Python 3.11**。

### 版本信息

```
Ubuntu 20.04 LTS
├── 默認 Python: 3.8.10 (不支持)
├── 需要升級到: 3.11.x
└── 預期安裝時間: 15-20 分鐘
```

---

## 🚀 方法 1: 使用 Deadsnakes PPA (推薦 ⭐)

### 步驟 1: 更新系統

```bash
sudo apt update && sudo apt upgrade -y
```

### 步驟 2: 添加 Deadsnakes PPA

```bash
# Deadsnakes 是官方 Python 團隊維護的 PPA
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
```

### 步驟 3: 安裝 Python 3.11

```bash
# 安裝 Python 3.11 及開發工具
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils

# 驗證
python3.11 --version
# 應輸出: Python 3.11.x
```

### 步驟 4: 設置 Python 3.11 為默認

```bash
# 查看當前 python3 指向
which python3

# 設置 alternatives
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# 驗證
python --version
python3 --version
# 都應輸出: Python 3.11.x
```

### 步驟 5: 安裝 pip

```bash
# 升級 pip 到最新版本
python3.11 -m pip install --upgrade pip

# 驗證
pip --version
# 應輸出: pip 23.x.x from ... (python 3.11)
```

---

## 🔧 方法 2: 使用源代碼編譯 (高級)

如果 PPA 不可用，可以從源代碼編譯：

```bash
# 安裝依賴
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl

# 下載 Python 3.11
cd /tmp
curl -O https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tar.xz
tar -xf Python-3.11.7.tar.xz
cd Python-3.11.7

# 編譯 (約 5-10 分鐘)
./configure --enable-optimizations
make -j$(nproc)
sudo make install

# 驗證
python3.11 --version
```

---

## 📦 完整安裝腳本 (Ubuntu 20.04)

### 創建自動化腳本

```bash
# 創建腳本
cat > ~/install_ubuntu2004.sh << 'EOF'
#!/bin/bash

set -e

echo "================================================"
echo "  Ubuntu 20.04 LTS - Crypto Bot 部署腳本"
echo "================================================"
echo ""

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}▸ 更新系統...${NC}"
sudo apt update && sudo apt upgrade -y

echo -e "${YELLOW}▸ 安裝基礎工具...${NC}"
sudo apt install -y software-properties-common curl wget git htop tmux

echo -e "${YELLOW}▸ 添加 Deadsnakes PPA...${NC}"
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

echo -e "${YELLOW}▸ 安裝 Python 3.11...${NC}"
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils

echo -e "${YELLOW}▸ 設置 Python 3.11 為默認...${NC}"
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

echo -e "${YELLOW}▸ 升級 pip...${NC}"
python3.11 -m pip install --upgrade pip

# 克隆倉庫
echo ""
echo -e "${YELLOW}▸ 克隆倉庫...${NC}"
cd /home/$USER
if [ ! -d "crypto-discord-bot" ]; then
    git clone https://github.com/caizongxun/crypto-discord-bot.git
else
    echo -e "${GREEN}✓ 倉庫已存在${NC}"
fi

cd crypto-discord-bot

# 創建虛擬環境
echo -e "${YELLOW}▸ 創建虛擬環境...${NC}"
python3.11 -m venv venv
source venv/bin/activate

# 安裝依賴
echo -e "${YELLOW}▸ 安裝 Python 依賴...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# 設置 .env
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}▸ 創建 .env 文件...${NC}"
    cp .env.example .env
    echo -e "${RED}✗ 請編輯 .env 文件並添加 Discord Token${NC}"
    echo -e "${GREEN}  nano .env${NC}"
    exit 1
fi

echo ""
echo "================================================"
echo -e "${GREEN}✓ 安裝完成!${NC}"
echo "================================================"
echo ""
echo -e "${GREEN}下一步:${NC}"
echo -e "${GREEN}1. 選擇部署方式 (Systemd/Screen/Docker)${NC}"
echo -e "${GREEN}2. 編輯 .env 添加 Discord Token${NC}"
echo -e "${GREEN}3. 啟動機器人${NC}"
echo ""
EOF

# 添加執行權限
chmod +x ~/install_ubuntu2004.sh

# 運行腳本
bash ~/install_ubuntu2004.sh
```

---

## ✅ 驗證安裝

### 檢查 Python 版本

```bash
# 應該都輸出 3.11.x
python --version
python3 --version
python3.11 --version

# 檢查 pip
pip --version

# 檢查虛擬環境
source venv/bin/activate
python --version
```

### 測試必要的包

```bash
cd crypto-discord-bot
source venv/bin/activate

# 測試 Discord.py
python -c "import discord; print(f'discord.py {discord.__version__}')"

# 測試 PyTorch (CPU)
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 測試 CCXT
python -c "import ccxt; print(f'CCXT {ccxt.__version__}')"

# 測試 Flask
python -c "import flask; print(f'Flask {flask.__version__}')"
```

---

## 🚀 部署方式

選擇以下任一方式：

### 方式 A: Systemd (推薦 ⭐)

```bash
# 創建服務文件
sudo tee /etc/systemd/system/crypto-bot.service > /dev/null <<'EOF'
[Unit]
Description=Crypto Discord Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/crypto-discord-bot
# 重要: 使用虛擬環境中的 Python
ExecStart=/home/$USER/crypto-discord-bot/venv/bin/python bot.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 重新加載 systemd
sudo systemctl daemon-reload

# 啟動機器人
sudo systemctl start crypto-bot
sudo systemctl enable crypto-bot

# 查看狀態
sudo systemctl status crypto-bot

# 查看日誌
sudo journalctl -u crypto-bot -f
```

### 方式 B: Screen

```bash
cd crypto-discord-bot
source venv/bin/activate
screen -S bot
# 在 screen 中:
python bot.py
# 離開: Ctrl+A 然後 D

# 重新連接: screen -r bot
```

### 方式 C: Tmux

```bash
cd crypto-discord-bot
source venv/bin/activate
tmux new-session -d -s bot "python bot.py"

# 連接: tmux attach -t bot
# 離開: Ctrl+B 然後 D
```

---

## 🔍 常見問題 & 解決方案

### 問題 1: `python: command not found`

```bash
# 解決方案
python3.11 --version
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
python --version
```

### 問題 2: `pip: command not found`

```bash
# 使用 pip3.11
pip3.11 install -r requirements.txt

# 或設置別名
alias pip=pip3.11
```

### 問題 3: `venv` 無法激活

```bash
# 重新創建虛擬環境
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 問題 4: PyTorch 安裝失敗

```bash
# 檢查 pip 版本
pip --version

# 升級 pip
python3.11 -m pip install --upgrade pip

# 重試安裝
pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### 問題 5: `discord.py` 導入失敗

```bash
# 確認虛擬環境已激活
source venv/bin/activate

# 重新安裝
pip uninstall discord.py -y
pip install discord.py==2.4.0
```

---

## 📊 Ubuntu 20.04 vs 22.04 對比

| 功能 | 20.04 LTS | 22.04 LTS | 20.04 解決方案 |
|------|-----------|-----------|---------------|
| **默認 Python** | 3.8.10 | 3.10.x | 升級到 3.11 |
| **Systemd** | ✓ | ✓ | 完全相同 |
| **apt 包管理** | ✓ | ✓ | 完全相同 |
| **安裝難度** | 中等 | 簡單 | 使用本指南 |
| **性能** | ✓ | ✓ | 相同 |
| **支持期限** | 至 2030 年 | 至 2032 年 | 足夠長 |

---

## 🔐 安全建議 (Ubuntu 20.04 特定)

### 1. 定期更新

```bash
# 自動安全更新
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure unattended-upgrades

# 查看更新日誌
sudo tail -f /var/log/unattended-upgrades/unattended-upgrades.log
```

### 2. 防火牆配置

```bash
# 啟用 UFW
sudo apt install ufw -y
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 5000/tcp  # Dashboard
sudo ufw enable

# 查看規則
sudo ufw status
```

### 3. 監控磁盤空間

```bash
# 檢查當前使用
df -h

# 查看大文件
du -sh /home/$USER/crypto-discord-bot/*

# 清理舊日誌
sudo journalctl --vacuum=30d
```

---

## 📝 完整部署檢查清單

- [ ] Ubuntu 20.04 LTS 系統已更新
- [ ] Python 3.11 已安裝
- [ ] Python 3.11 已設為默認版本
- [ ] pip 已升級
- [ ] 倉庫已克隆
- [ ] 虛擬環境已創建
- [ ] 所有 Python 依賴已安裝
- [ ] .env 文件已編輯並添加 Discord Token
- [ ] 已選擇部署方式 (Systemd/Screen/Tmux)
- [ ] `sudo systemctl status crypto-bot` 顯示 active
- [ ] `sudo journalctl -u crypto-bot` 無錯誤
- [ ] Discord 命令 `.models` 可正常執行

---

## 📞 故障排除

### 查看完整日誌

```bash
# 最後 100 行
sudo journalctl -u crypto-bot -n 100

# 實時日誌
sudo journalctl -u crypto-bot -f

# 特定時間段
journalctl -u crypto-bot --since "2025-12-14 08:00:00" --until "2025-12-14 09:00:00"
```

### 調試 Python 問題

```bash
cd crypto-discord-bot
source venv/bin/activate

# 測試機器人加載
python bot_predictor.py

# 測試 Discord 連接
python -c "import discord; print(discord.__version__)"

# 測試 HuggingFace
python -c "from huggingface_hub import list_repo_files; print(list(list_repo_files('zongowo111/crypto_model', repo_type='model'))[:3])"
```

---

## 🎓 學習資源

- [Python 3.11 官方文檔](https://docs.python.org/3.11/)
- [Ubuntu 20.04 升級指南](https://ubuntu.com/blog/python-3-11-and-ubuntu)
- [Deadsnakes PPA 文檔](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa)

---

## ✨ 下一步

1. 按照本指南完成 Python 3.11 升級
2. 返回 **GCP_QUICKSTART.md** 繼續部署
3. 或查看 **GCP_DEPLOYMENT.md** 了解詳細配置

---

**Ubuntu 20.04 機器人已準備就緒！** 🚀
