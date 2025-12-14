# 🚀 Quick Start Guide

## 5 分鐘快速開始

### 步驟 1: 克隆倉庫

```bash
git clone https://github.com/caizongxun/crypto-discord-bot.git
cd crypto-discord-bot
```

### 步驟 2: 取得 Discord Bot Token

1. 進入 [Discord Developer Portal](https://discord.com/developers/applications)
2. 點擊 "New Application"
3. 選擇 "Bot" → "Add Bot"
4. 複製 Token
5. 在 "MESSAGE CONTENT INTENT" 打開
6. 在 "OAuth2" → "URL Generator" 選擇:
   - Scopes: `bot`
   - Permissions: `Send Messages`, `Read Message History`, `Use Slash Commands`
7. 生成的 URL 邀請機器人到你的伺服器

### 步驟 3: 設定環境變數

**Windows:**
```bash
copy .env.example .env
# 編輯 .env，添加你的 Discord Token
DISCORD_TOKEN=your_token_here
```

**Linux/macOS:**
```bash
cp .env.example .env
# 編輯 .env
nano .env
```

### 步驟 4: 運行機器人

**Windows:**
```bash
run.bat
```

**Linux/macOS:**
```bash
chmod +x run.sh
./run.sh
```

**或手動:**
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python bot.py
```

### 步驟 5: 測試機器人

在 Discord 伺服器中輸入:

```
.models      # 列出所有加載的模型
.predict     # 獲取所有預測
.signal      # 獲取交易信號
.stats       # 顯示統計信息
```

## 🎯 常見問題

### Q: 機器人不回應

A: 檢查:
1. `DISCORD_TOKEN` 是否正確設置
2. 機器人是否在伺服器中
3. 機器人是否有發送消息的權限
4. 確認沒有拼寫錯誤

### Q: 模型加載失敗

A: 這是正常的，某些模型可能有維度不匹配。機器人會自動跳過並繼續加載其他模型。

### Q: Binance API 被阻止

A: 機器人會自動嘗試其他交易所:
- Binance → Bybit → OKX → Kraken

如果都失敗，檢查你的網絡連接或使用 VPN。

### Q: 如何使用網頁儀表板

A: 同時運行:

```bash
# 終端 1: 機器人
python bot.py

# 終端 2: 儀表板
python dashboard.py
```

然後在瀏覽器中打開: `http://localhost:5000`

## 📊 命令參考

| 命令 | 說明 |
|------|------|
| `.models` | 列出所有加載的模型及詳細信息 |
| `.predict [SYMBOL]` | 顯示特定或全部預測 |
| `.signal [SYMBOL]` | 顯示特定或全部交易信號 |
| `.stats` | 顯示機器人統計信息 |
| `.reload` | 重新加載所有模型 |
| `.dashboard` | 獲取儀表板 URL |
| `.test [SYMBOL]` | 測試單個模型 |

## 🐳 Docker 部署

### 使用 Docker Compose (推薦)

```bash
# 編輯 .env
echo "DISCORD_TOKEN=your_token_here" > .env

# 啟動
docker-compose up -d

# 查看日誌
docker-compose logs -f crypto-bot

# 停止
docker-compose down
```

### 手動 Docker

```bash
# 構建
docker build -t crypto-bot .

# 運行
docker run -e DISCORD_TOKEN=your_token_here crypto-bot
```

## 🔒 安全建議

⚠️ **重要:**

1. **永遠不要** 在代碼中硬編碼 Discord Token
2. 將 `.env` 添加到 `.gitignore` (已默認設置)
3. 使用強密碼保護你的 Discord 帳戶
4. 定期輪換 Token (在 Developer Portal 重新生成)
5. 不要與他人分享你的 `.env` 文件

## 📈 性能提示

### 優化預測週期

編輯 `bot.py` 的 `@tasks.loop()` 裝飾器:

```python
@tasks.loop(minutes=60)  # 每 60 分鐘運行一次
async def prediction_loop():
    # ...
```

### 優化模型加載

編輯 `bot_predictor.py`:

```python
DEFAULT_LOOKBACK = 50  # 減少歷史數據量 (默認 100)
EXCHANGES = ['binance']  # 只使用首選交易所
```

## 📚 下一步

- 📖 詳見 [完整 README](README.md)
- 🐛 遇到問題? 查看 [TROUBLESHOOTING](TROUBLESHOOTING.md)
- 💡 想添加功能? 查看 [貢獻指南](CONTRIBUTING.md)

## 🤝 支持

需要幫助?

- 📝 [開設 Issue](https://github.com/caizongxun/crypto-discord-bot/issues)
- 💬 [討論](https://github.com/caizongxun/crypto-discord-bot/discussions)
- 📧 聯系: caizongxun@example.com

---

**祝你使用愉快!** 🎉
