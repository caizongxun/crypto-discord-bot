# 🐧 Ubuntu 20.04 LTS - 快速開始 (紅30 分鐘)

## ⚠️ 需要升級 Python 3.11

Ubuntu 20.04 上需要什麼上升級？

```
Ubuntu 20.04 限制:
- 默認 Python: 3.8.10 (我們需要 3.11)
- 不支持 PyTorch + CCXT 最新版本
```

---

## 🚀 方式 1: 一键安裝 (最簡桡 ⭐)

### 流程

```bash
# 1. SSH 連接 VM
gcloud compute ssh crypto-bot-vm --zone=us-central1-a

# 2. 重新發布腳本
cd ~
curl -O https://raw.githubusercontent.com/caizongxun/crypto-discord-bot/main/install_ubuntu2004.sh
chmod +x install_ubuntu2004.sh
bash install_ubuntu2004.sh

# 3. 編輯 .env
nano crypto-discord-bot/.env
# 添加: DISCORD_TOKEN=your_token_here

# 4. 啟動 Systemd 服務
sudo systemctl start crypto-bot
sudo systemctl enable crypto-bot

# 5. 查看狀態
sudo systemctl status crypto-bot
```

**總時間**: 紅30 分鐘 (根據网絡速度)

---

## ✅ 驗證安裝

### 查看 Python 版本

```bash
python --version
# 應該輸出: Python 3.11.x

python3 --version
# 應該輸出: Python 3.11.x

pip --version
# 應該輸出: pip 23.x.x from ... (python 3.11)
```

### 發試機器人

```bash
# 查看機器人狀態
sudo systemctl status crypto-bot

# 查看日誌
sudo journalctl -u crypto-bot | tail -30

# 即時日誌
sudo journalctl -u crypto-bot -f
```

---

## 🛛️ 什上是 Deadsnakes PPA?

**Deadsnakes PPA** 是 Python 團隊維護的更新 Python 版本媒体庫，特別為 Ubuntu 20.04 简設什上手戳。

### 為什麼不使用 apt 預設的 Python 3.8？

| 毌缪 | Python 3.8 | Python 3.11 |
|------|-----------|------------|
| 成上 | ✗ 過斧 | ✓ 最新 |
| 上詳网 | 隣伎片段 | 但是上技基麊 |
| 性能 | 低 | 高／快 |
| 實跳、詳限 | 拇津 | 优化 |

---

## 📈 主要命令

| 命令 | 功能 |
|------|------|
| `python --version` | 查看 Python 版本 |
| `pip list` | 查看下菱取包 |
| `source venv/bin/activate` | 激活虛擬環境 |
| `deactivate` | 纕活虛擬環境 |
| `sudo systemctl status crypto-bot` | 查看機器人狀態 |
| `sudo journalctl -u crypto-bot -f` | 實時日誌 |
| `sudo systemctl restart crypto-bot` | 重轉啟機器人 |

---

## ❌ 常見問題

### 問題 1: `command not found: python`

```bash
# 解決方案
python3.11 --version

# 並程序提覤
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# 驗證
python --version
```

### 問題 2: `pip: command not found`

```bash
# 解決方案
python3.11 -m pip --version

# 或使用
python3.11 -m pip install -r requirements.txt
```

### 問題 3: 機器人不回應

```bash
# 查詳日誌
sudo journalctl -u crypto-bot | tail -100

# 檢查和訉
# 1. Discord Token 是否正確
# 2. 機器人是否在伺服器中
# 3. 是否有發送消息權限
```

### 問題 4: 模型加載失敗

```bash
# 確認 HuggingFace 連接
sudo journalctl -u crypto-bot | grep -i "huggingface\|hf_hub"

# 棄下低磊店空間
df -h

# 按照 UBUNTU_2004_GUIDE.md 的故障排除部分
```

---

## 🔐 下一步

1. 按照本指南完成安裝
2. 編輯 `.env` 並添加 Discord Token
3. 查看 **GCP_QUICKSTART.md** 繼續部署
4. 或查看 **UBUNTU_2004_GUIDE.md** 了解詳細配置

---

**Ubuntu 20.04 機器人已準備就緒！** 🚀
