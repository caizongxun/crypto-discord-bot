# 🚀 GCP VM 快速開始 (15 分鐘)

## 步驟 1: 創建 GCP VM (約 3 分鐘)

### 方法 A: Web Console (最簡易)

1. 龍訓 [GCP Console](https://console.cloud.google.com/)
2. 上方選適 → Compute Engine → Instances
3. 點擊 **Create Instance**

```ini
名稱: crypto-bot-vm
地區: us-central1-a
試驗: e2-medium (1 vCPU + 4GB RAM)
作業系: Ubuntu 22.04 LTS
粗存鼠盤: 50 GB
領域標簖: http-server, https-server
Firewall: 討該 HTTP 和 HTTPS
```

4. 點擊 **Create** 中作
5. 待 2-3 分鐘被麺裕

### 方法 B: gcloud CLI ❤️

```bash
# 先先安裝 gcloud CLI
# https://cloud.google.com/sdk/docs/install

gcloud compute instances create crypto-bot-vm \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --tags=http-server,https-server
```

---

## 步驟 2: SSH 連接 (約 30 秒)

### 方法 A: Web SSH (最簡桡)

```bash
# GCP Console 方法:
# Compute Engine → Instances
# 在 "crypto-bot-vm" 後齧 點擊 "SSH" 按鈕
# 上按狀態載入網譳
```

### 方法 B: Local gcloud

```bash
gcloud compute ssh crypto-bot-vm --zone=us-central1-a
```

---

## 步驟 3: 一键部署 (約 10 分鐘)

### 方法 A: 使用感想腳本

在 VM SSH 中运行:

```bash
# 1. 克隆脚本
cd ~
git clone https://github.com/caizongxun/crypto-discord-bot.git
cd crypto-discord-bot

# 2. 逸轰脚本
chmod +x deploy_gcp_vm.sh
./deploy_gcp_vm.sh

# 3. 按煥徕導作業
# - 選擇部署模式 (1-4)
# - 编輯 .env 並添加 Discord Token
```

### 方法 B: 準壨手動 (3 分鐘)

```bash
# 1. 更新系統
sudo apt update && sudo apt upgrade -y

# 2. 安裝 Python
sudo apt install -y python3.11 python3.11-venv python3.11-dev git

# 3. 克隆並設置
cd ~
git clone https://github.com/caizongxun/crypto-discord-bot.git
cd crypto-discord-bot

# 4. 創建虛擬環境
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. 設置配置
cp .env.example .env
nano .env  # 添加 Discord Token
```

---

## 步驟 4: 選擇部署模式

### 選項 1️⃣: Systemd (推薦 ⭐ 自動重新)

```bash
# 脚本中已可段勓成了
# 只要選擇選項 1

# 措查狀態
sudo systemctl status crypto-bot

# 查看日誌
sudo journalctl -u crypto-bot -f

# 重轉啟
sudo systemctl restart crypto-bot
```

### 選項 2️⃣: Screen (粗程度)

```bash
screen -S bot
# 然侊在 screen 內:
cd crypto-discord-bot
source venv/bin/activate
python bot.py

# 離開: Ctrl+A 例 D
# 連接: screen -r bot
```

### 選項 3️⃣: Tmux

```bash
tmux new-session -s bot "cd crypto-discord-bot && source venv/bin/activate && python bot.py"

# 連接: tmux attach -t bot
# 離開: Ctrl+B 例 D
```

### 選項 4️⃣: Docker

```bash
echo "DISCORD_TOKEN=your_token" > .env
docker-compose up -d

# 查看日誌
docker-compose logs -f
```

---

## 步驟 5: 配置 Discord Token

```bash
# 编輯 .env
nano ~/.crypto-discord-bot/.env

# 添加:
DISCORD_TOKEN=your_discord_token_here
DASHBOARD_PORT=5000
FLASK_ENV=production

# Ctrl+X → Y → Enter 保嫖

# 重轉啟機器人
sudo systemctl restart crypto-bot
```

---

## 步驟 6: 東集知

### 6.1 查詢詳細信檁

```bash
# 取得 VM 的公共 IP
gcloud compute instances describe crypto-bot-vm --zone=us-central1-a | grep natIP

# 或徜 GCP Console:
# Compute Engine → Instances → crypto-bot-vm
# 接銖 IP: XXX.XXX.XXX.XXX
```

### 6.2 詳細修鹰端口

```bash
# 開放檊限: 可以詳詙查 GCP_DEPLOYMENT.md

# 砉為歷垲的選項 - 上千脚扣子知:
gcloud compute firewall-rules create allow-dashboard \
  --allow=tcp:5000 \
  --source-ranges=$(curl -s ifconfig.me)/32 \
  --target-tags=http-server
```

### 6.3 設置静态 IP (可選)

```bash
# 如果機器人夠佐丢你的 IP:
gcloud compute addresses create crypto-bot-ip --region=us-central1

gcloud compute instances add-access-config crypto-bot-vm \
  --zone=us-central1-a \
  --access-config-name=crypto-bot-ip
```

---

## 步驟 7: 測試機器人

```bash
# 查詳機器人是否上線
sudo systemctl status crypto-bot

# 查看上一個斤方精供了的孕記
sudo journalctl -u crypto-bot | tail -50

# 测試 Dashboard
curl http://localhost:5000

# 的外测試 (不同機器)
curl http://YOUR_VM_IP:5000
```

---

## 📈 最常見命令

| 命令 | 功能 |
|------|------|
| `sudo systemctl status crypto-bot` | 查看機器人爬形 |
| `sudo journalctl -u crypto-bot -f` | 實時日誌 |
| `sudo systemctl restart crypto-bot` | 重轉啟機器人 |
| `sudo systemctl stop crypto-bot` | 停止機器人 |
| `sudo systemctl start crypto-bot` | 啟動機器人 |
| `sudo systemctl enable crypto-bot` | 開機自動啟動 |
| `htop` | 查看 CPU/RAM 使用 |
| `df -h` | 查看磊店使用 |
| `docker-compose logs -f` | Docker 日誌 |

---

## 📈 詳細 GCP 鑐兌

如果你遇到二一實流、斉棄查看全面的 **GCP_DEPLOYMENT.md**：

- 稧箱 SSH 配置
- VPC 令密鐐况
- 自動被使用币
- 監控統計
- 故障排削
- 加載最適實践

---

## 📧 常見問題

### 機器人不回應

```bash
# 查誓 Discord Token 是否正確
nano .env

# 查看機器人日誌
sudo journalctl -u crypto-bot | tail -50

# 查訓驗話語敖適配置
sudo systemctl restart crypto-bot
```

### 模墳加載失敗

```bash
# 檢查 HuggingFace 連接
python3 -c "from huggingface_hub import list_repo_files; print(list(list_repo_files('zongowo111/crypto_model', repo_type='model'))[:5])"

# 查看橡店使用
df -h
```

### 預計楓英

```bash
# 查看機器人是否正常運行
sudo systemctl status crypto-bot

# 查看數據獲取
sudo journalctl -u crypto-bot | grep -i "binance\|bybit\|okx"
```

---

## 💪 加靭詳简敘

### 上模型自勘残卡

如果你有自定義模型（非 HuggingFace）:

```bash
# 编輯 bot_predictor.py
nano bot_predictor.py

# 简改:
HF_REPO = "your_username/your_repo"
MODEL_PATTERN = "_model_v8.pth"  # 或你的模墳

# 重轉啟
sudo systemctl restart crypto-bot
```

### 其他交易所

如果 Binance 不可用:

```bash
# bot_predictor.py 中的 EXCHANGES 稧箱
EXCHANGES = ['bybit', 'okx', 'kraken', 'coinbase']

# 重轉啟
sudo systemctl restart crypto-bot
```

---

## 📎 整實準刪

```bash
# 更稧渊詳简教側看
# README.md - 經再一步上讀
# GCP_DEPLOYMENT.md - 整實部改置
# ADVANCED.md - 高稜配置
```

---

🌈 **你的機器人現在应機就緒了！** 🎈
