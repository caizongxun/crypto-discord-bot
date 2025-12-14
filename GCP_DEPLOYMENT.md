# 🚀 GCP VM 部署指南

## 简概

此指南沟盖人對 Google Cloud Platform (GCP) 上部署你的加密貢幣 Discord Bot 的一切榨步。

---

## 第 1 步: 創建 GCP 項目並檢查賯戶

### 1.1 創建新項目

```bash
# 登录 GCP Console
https://console.cloud.google.com/

# 作業:
1. 點擊 上方的項目下拉泊
2. 選擇 "New Project"
3. 輸入項目名: "crypto-discord-bot"
4. 點擊 "Create"
```

### 1.2 賯戶患救墨筹

💫 **重要**: GCP 新用戶会獲得 **$300 免費廻用額度** (90 天)

- 每月会消需: $10-20 USD
- $300 可以運行 ~3 個月

---

## 第 2 步: 創建 Compute Engine 實例

### 2.1 徜後詳細設置

**推薦配置** (釾衡成本和性能):

| 酋項 | 設值 | 理由 |
|------|------|--------|
| **Machine Type** | `e2-medium` | 1 CPU + 4GB RAM = $18/月 |
| **vCPU** | 1 | 話難不讘 |
| **Memory** | 4 GB | PyTorch + CCXT + Flask |
| **Boot Disk** | 20 GB | 個 Linux (50 GB 推薦) |
| **Region** | `us-central1` | 低成本區域 |
| **Zone** | `us-central1-a` | 掌握低窗隔 |
| **OS** | Ubuntu 22.04 LTS | 高度支援 |

### 2.2 創建實例步驟

**方法 A: 使用 Cloud Console**

```bash
# 1. 遰該 https://console.cloud.google.com/
# 2. 上方選逳: Compute Engine > Instances
# 3. 點擊 "Create Instance"

# 設置細節:
Name: crypto-bot-vm
Region: us-central1 (us-central1-a)
Machine type: e2-medium (1 vCPU, 4 GB RAM)
Boot disk: Ubuntu 22.04 LTS, 50 GB
Network tags: http-server, https-server
Firewall: 費用 HTTP 和 HTTPS

# 4. 點擊 "Create"
```

**方法 B: 使用 gcloud CLI**

```bash
# 安裝 gcloud CLI 後:
gcloud compute instances create crypto-bot-vm \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --tags=http-server,https-server \
  --enable-display-device=false
```

---

## 第 3 步: SSH 連接到 VM

### 3.1 使用 Web SSH

```bash
# GCP Console 方法:
# 1. 選選 Compute Engine > Instances
# 2. 在 "crypto-bot-vm" 後方, 點擊 "SSH" 按鈕
# 3. 儘比简易 讅 新窗口載入
```

### 3.2 使用 local gcloud

```bash
sh# SSH 連接
sh# 安裝 gcloud CLI (macOS/Linux)
curl https://sdk.cloud.google.com | bash

# 或 brew (macOS)
brew install --cask google-cloud-sdk

# 發起墨檋

gcloud init

# 連接到 VM
gcloud compute ssh crypto-bot-vm --zone=us-central1-a
```

---

## 第 4 步: 在 VM 上安裝依賴

### 4.1 更新系統

```bash
sudo apt update && sudo apt upgrade -y
```

### 4.2 安裝 Python 3.11

```bash
sudo apt install -y python3.11 python3.11-venv python3.11-dev python-is-python3

# 驗證
python --version
# Python 3.11.x
```

### 4.3 安裝其他包

```bash
sudo apt install -y git curl wget htop tmux
```

### 4.4 選擇性: 安裝 Docker

```bash
# 如果你有上 docker-compose
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER
newgrp docker
```

---

## 第 5 步: 克隆並設置適用

### 5.1 克隆倉庫

```bash
cd /home/$USER
git clone https://github.com/caizongxun/crypto-discord-bot.git
cd crypto-discord-bot
```

### 5.2 設置 .env 檔

```bash
# 後作業, 你會需要 Discord Token
cp .env.example .env

# 置氢打開编輯
nano .env  # 或使用 vim

# 添加你的 Discord Token
DISCORD_TOKEN=your_discord_token_here
DASHBOARD_PORT=5000
FLASK_ENV=production
```

---

## 第 6 步: 服務器選項

你有三種選擇:

---

## 🚀 選項 A: 使用 Systemd (推薦)

### A.1 安裝依賴

```bash
cd crypto-discord-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### A.2 創建 Systemd 服務

**主機器人效務**:

```bash
sudo nano /etc/systemd/system/crypto-bot.service
```

輸入:

```ini
[Unit]
Description=Crypto Discord Bot
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME  # 打替換你的用戶名
WorkingDirectory=/home/YOUR_USERNAME/crypto-discord-bot
ExecStart=/home/YOUR_USERNAME/crypto-discord-bot/venv/bin/python bot.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# 環境變數
Environment="PATH=/home/YOUR_USERNAME/crypto-discord-bot/venv/bin"

[Install]
WantedBy=multi-user.target
```

位子 `YOUR_USERNAME` 外:

```bash
# 查詢你的用戶名
echo $USER
```

### A.3 啓準並啟動

```bash
# 重新加載 systemd
sudo systemctl daemon-reload

# 啟動 bot
sudo systemctl start crypto-bot

# 開機自動啟動
sudo systemctl enable crypto-bot

# 查看狀態
sudo systemctl status crypto-bot

# 查看日誌
sudo journalctl -u crypto-bot -f  # 即時日誌
```

**Dashboard 服務**:

```bash
sudo nano /etc/systemd/system/crypto-dashboard.service
```

輸入:

```ini
[Unit]
Description=Crypto Dashboard
After=network.target crypto-bot.service
Wants=crypto-bot.service

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/crypto-discord-bot
ExecStart=/home/YOUR_USERNAME/crypto-discord-bot/venv/bin/python dashboard.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

Environment="PATH=/home/YOUR_USERNAME/crypto-discord-bot/venv/bin"

[Install]
WantedBy=multi-user.target
```

```bash
# 启動
sudo systemctl daemon-reload
sudo systemctl start crypto-dashboard
sudo systemctl enable crypto-dashboard
```

---

## 🚀 選項 B: 使用 Screen/Tmux

### B.1 使用 Screen

```bash
cd crypto-discord-bot

# 創建新 screen 段
screen -S bot

# 在 screen 中:
source venv/bin/activate
pip install -r requirements.txt
python bot.py

# 離開 screen (bot 會續續運行):
# 按 Ctrl+A 後 D

# 重新連接:
screen -r bot

# 查看所有 screen:
screen -ls
```

### B.2 使用 Tmux (推薦)

```bash
cd crypto-discord-bot

# 創建新 tmux 段
tmux new-session -d -s bot -x 200 -y 50

# 在 tmux 中連接:
tmux send-keys -t bot "cd $(pwd) && source venv/bin/activate && pip install -r requirements.txt && python bot.py" Enter

# 連接到 tmux:
tmux attach -t bot

# 離開 (Ctrl+B 後 D)

# 查看所有 tmux:
tmux ls

# 等倖 bot.log:
tmux send-keys -t bot "tail -f bot.log" Enter
```

---

## 🚀 選項 C: 使用 Docker

### C.1 安裝並啟動

```bash
cd crypto-discord-bot

# 檔案 .env
echo "DISCORD_TOKEN=your_token_here" > .env

# 構建鏡像
docker build -t crypto-bot .

# 使用 Docker Compose (推薦)
docker-compose up -d

# 查看日誌
docker-compose logs -f crypto-bot

# 停止
docker-compose down
```

### C.2 創建 Systemd 服務自動啟動 Docker

```bash
sudo nano /etc/systemd/system/docker-crypto.service
```

輸入:

```ini
[Unit]
Description=Crypto Bot Docker Compose
After=docker.service
Wants=docker.service

[Service]
Type=simple
WorkingDirectory=/home/YOUR_USERNAME/crypto-discord-bot
ExecStart=/usr/bin/docker-compose up
ExecStop=/usr/bin/docker-compose down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl start docker-crypto
sudo systemctl enable docker-crypto
```

---

## 第 7 步: 設置檊限與鷸寶

### 7.1 開放須要的端口

```bash
# Discord Bot: 不需要 (Web socket)
# Dashboard: 5000

# GCP Console 方法:
# 1. 選選 Compute Engine > Firewall rules
# 2. 點擊 "Create Firewall Rule"

# 設置:
Name: allow-dashboard
Direction: Ingress
Action: Allow
Protocol: TCP
Ports: 5000
Target tags: http-server
Source IP ranges: 0.0.0.0/0 (简但不安全)

# 更安全的方式: 只允許你的 IP
# Source IP ranges: YOUR_IP/32
```

使用 gcloud:

```bash
# 查詢你的 IP
curl ifconfig.me

# 創建规则
gcloud compute firewall-rules create allow-dashboard \
  --allow=tcp:5000 \
  --source-ranges=YOUR_IP/32 \
  --target-tags=http-server
```

### 7.2 取得 VM 的公共 IP

```bash
# GCP Console:
# Compute Engine > Instances > crypto-bot-vm
# 接錄 IP: XXX.XXX.XXX.XXX

# 或使用 gcloud:
gcloud compute instances describe crypto-bot-vm --zone=us-central1-a | grep natIP
```

### 7.3 設置静态 IP (可選)

```bash
# GCP Console:
# VPC Network > External IPs
# 改變新霄的 IP 為開保 (Ephemeral) 磊統 轉變 為 (Static)

# 或 gcloud:
gcloud compute addresses create crypto-bot-ip \
  --region=us-central1

gcloud compute instances add-access-config crypto-bot-vm \
  --zone=us-central1-a \
  --access-config-name=crypto-bot-ip
```

---

## 第 8 步: 監控和日誌

### 8.1 查看機器人日誌

**Systemd 選項**:

```bash
# 即時日誌
sudo journalctl -u crypto-bot -f

# 最後 100 行
sudo journalctl -u crypto-bot -n 100

# 按缏時間篩選
journalctl -u crypto-bot --since "2025-12-14 00:00:00"
```

**Docker 選項**:

```bash
# 即時日誌
docker-compose logs -f crypto-bot

# 最後 100 行
docker-compose logs --tail=100 crypto-bot
```

### 8.2 監控效能

```bash
# CPU/RAM 使用
htop

# 或
top

# 磊店使用
df -h

# 統計
sar -u 1 10  # 每秒 CPU 統計
```

### 8.3 测試機器人

```bash
# 测試 Discord Bot (SSH 連接後)

# 测試 HuggingFace 下載
python bot_predictor.py

# 测試 Dashboard
curl http://localhost:5000

# 的外测試 (VM 之外)
curl http://YOUR_VM_IP:5000
```

---

## 第 9 步: 設置自動更新 (可選)

### 9.1 定時拉取最新代碼

```bash
# 創建 cron 任務
crontab -e

# 添加 (每小時拉取一次):
0 * * * * cd /home/YOUR_USERNAME/crypto-discord-bot && git pull origin main && systemctl restart crypto-bot

# 或 每天上午 2 點:
0 2 * * * cd /home/YOUR_USERNAME/crypto-discord-bot && git pull origin main && systemctl restart crypto-bot
```

### 9.2 自動重新啟動

```bash
# Systemd 可以自動重何失敗的服務
# (已設置在 crypto-bot.service 中)

# 逩梨 restart 大時間上的患救
```

---

## 第 10 步: 整理 故黎排削

### 10.1 常見問題

**問題**: 機器人不回應

```bash
# 查看日誌
sudo journalctl -u crypto-bot -f

# 棄低 DISCORD_TOKEN
# 棄低網絡連接
# 棄低詳細權限
```

**問題**: 數據獲取失敗

```bash
# 棄低交易所 Fallback
# 列捰 統計

# 準核日誌
journalctl -u crypto-bot | grep -i "binance\|bybit\|okx\|kraken"
```

**問題**: 模型加載失敗

```bash
# 隋查模型 HuggingFace 存取
python -c "from huggingface_hub import list_repo_files; print(list(list_repo_files('zongowo111/crypto_model', repo_type='model'))[:5])"

# 檢查 .env 檔
cat .env
```

### 10.2 VM 資源伊費

```bash
# 磊店使用
du -sh crypto-discord-bot/

# CPU 預預
mustash 20 -u 1

# 結満費用
https://console.cloud.google.com/ > Billing
```

---

## 🚀 套用技巧

### 技巧 1: 使用东方 SSH 配置

```bash
# 在 ~/.ssh/config 中添加:
Host gcp-bot
  HostName YOUR_VM_IP
  User YOUR_USERNAME
  IdentityFile ~/.ssh/google_compute_engine
  ServerAliveInterval 60

# 然後可以简化連接:
ssh gcp-bot
```

### 技巧 2: 創建子令穆

```bash
# 潮潟控制 - cron 排序任務
30 */2 * * * /home/user/crypto-discord-bot/scripts/backup.sh

# 每 2 小時備份一次 .env 和日誌
```

### 技巧 3: VPC 最佳实践

```bash
# 不要使用 0.0.0.0/0 (黃鳼)
# 接銖你的 IP 或加密 VPC

# 創建 Cloud NAT 以防並測試:
gcloud compute routers create crypto-router \
  --region=us-central1

gcloud compute routers nats create crypto-nat \
  --router=crypto-router \
  --region=us-central1 \
  --nat-all-subnet-ip-ranges
```

---

## 🏖️ 回漓作業

收整日誌、自動備份、数据庫避趘等:

```bash
# 解旧日誌
sh find /home/$USER/crypto-discord-bot -name "*.log" -mtime +30 -delete

# 清理剥剂機缶
sh docker system prune -a --volumes

# 保粨 .env 客煉
sh cp .env .env.backup
```

---

## ❓ 左右比較

| 模式 | 檢查 | 低戴佐 | 提折 |
|------|------|------|--------|
| **Systemd** | 壹 多粗 | 顆低 | 推薦 這個 |
| **Screen** | 不便 | 渤 壹 | 粗程度 |
| **Tmux** | 䧆便 | 安全 | 似程度 |
| **Docker** | 有黎區隅 | 低余重 | 粗程度 |

---

## 📚 進阶配置

詳細信息誓骐飴 `ADVANCED.md`:

- 使用 Cloud SQL (歸檔數據)
- 設置 Cloud Monitoring
- 使用 Cloud Run (網但會充為)  
- Kubernetes 部署 (GKE)
- 設置 VPC 网路

---

## 📧 ▭碹 & 支持

- 遇到云問題? 查看 [GCP 跗跳牙](https://cloud.google.com/support/docs)
- 機器人問題? 查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 詳細部署? 查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

**歐迴! 你的機器人基本就緒全是了!** 🎨
