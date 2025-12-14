# 📁 Project Structure & File Reference

## 完整項目架構

```
crypto-discord-bot/
│
├── 🤖 核心文件
│   ├── bot.py                          # Discord 機器人主程式 (14KB)
│   │   ├── @bot.event on_ready()       # 初始化 + 啟動預測循環
│   │   ├── @bot.command .models        # 列出所有加載的模型
│   │   ├── @bot.command .predict       # 預測 1-N 個幣種
│   │   ├── @bot.command .signal        # 交易信號 (LONG/SHORT)
│   │   ├── @bot.command .stats         # 統計信息
│   │   ├── @bot.command .reload        # 重新加載模型
│   │   ├── @bot.command .dashboard     # 儀表板 URL
│   │   ├── @bot.command .test          # 測試單個模型
│   │   ├── @tasks.loop(minutes=60)     # 自動預測循環
│   │   └── helper functions            # Embed 創建函數
│   │
│   └── bot_predictor.py                # 預測引擎 (18KB) ⭐ 核心
│       ├── CryptoLSTMModel             # 自適應 LSTM 架構
│       │   ├── __init__(input, hidden) # 自動檢測維度
│       │   └── forward(x)              # 前向傳播
│       │
│       ├── CryptoPredictor             # 主預測類
│       │   ├── initialize()            # 初始化 (下載模型)
│       │   ├── _get_hf_model_files()   # 從 HuggingFace 獲取
│       │   ├── _load_model()           # 加載單個模型
│       │   ├── _detect_model_config()  # 自動檢測架構 ⭐
│       │   ├── predict_single()        # 預測單個幣種
│       │   ├── _fetch_ohlcv()          # 從交易所獲取數據
│       │   ├── _prepare_features()     # 特徵歸一化
│       │   ├── _generate_predictions() # 生成 5 根 K 線預測
│       │   ├── _analyze_trend()        # 趨勢分析 + 信心度
│       │   ├── _calculate_entry_points()# 計算入場點
│       │   └── _calculate_support_resistance() # 支撐/阻力位
│       │
│       └── Constants:
│           ├── HF_REPO = "zongowo111/crypto_model"
│           ├── MODEL_PATTERN = "_model_v8.pth"
│           ├── DEVICE = torch.device('cpu')
│           ├── EXCHANGES = ['binance', 'bybit', 'okx', 'kraken']
│           └── DEFAULT_LOOKBACK = 100
│
├── 🌐 Web 儀表板
│   ├── dashboard.py                    # Flask 後端 (3.6KB)
│   │   ├── @app.route('/')             # 主頁面
│   │   ├── @app.route('/api/predictions') # 所有預測
│   │   ├── @app.route('/api/signals')  # 交易信號
│   │   └── @app.route('/api/statistics')# 統計數據
│   │
│   └── templates/
│       └── dashboard.html              # 前端 UI (16KB) 🎨
│           ├── HTML5 + CSS3 + JavaScript
│           ├── Responsive grid layout
│           ├── Real-time updates (30s)
│           ├── Filter tabs (ALL/LONG/SHORT)
│           └── Export to JSON
│
├── 📦 配置與依賴
│   ├── requirements.txt                # Python 依賴
│   │   ├── discord.py==2.4.0
│   │   ├── torch==2.0.1+cpu           # CPU 版本
│   │   ├── huggingface-hub==0.19.4
│   │   ├── ccxt==4.1.55               # 交易所 API
│   │   ├── Flask==3.0.0
│   │   ├── pandas==2.1.3
│   │   └── numpy==1.24.3
│   │
│   ├── .env.example                    # 環境變量模板
│   │   ├── DISCORD_TOKEN=...
│   │   ├── DASHBOARD_URL=...
│   │   └── DASHBOARD_PORT=5000
│   │
│   └── .gitignore
│       ├── .env
│       ├── venv/
│       ├── __pycache__/
│       ├── *.log
│       └── models/hf_cache/
│
├── 🚀 啟動腳本
│   ├── run.sh                          # Linux/macOS 啟動器 (3KB)
│   │   ├── 檢查 Python
│   │   ├── 創建虛擬環境
│   │   ├── 安裝依賴
│   │   ├── 驗證 .env
│   │   ├── 啟動 bot 和 dashboard
│   │   └── 信號處理 (Ctrl+C)
│   │
│   └── run.bat                         # Windows 啟動器 (2.6KB)
│       ├── 檢查 Python
│       ├── 創建虛擬環境
│       ├── 安裝依賴
│       ├── 驗證 .env
│       └── 啟動新窗口
│
├── 🐳 Docker 支持
│   ├── Dockerfile                      # 容器鏡像定義
│   │   ├── FROM python:3.11-slim
│   │   ├── COPY + 安裝依賴
│   │   ├── EXPOSE 5000
│   │   └── CMD ["python", "bot.py"]
│   │
│   └── docker-compose.yml              # 編排配置
│       ├── crypto-bot service
│       ├── dashboard service
│       ├── 共享網絡
│       └── 卷掛載
│
├── 📚 文檔
│   ├── README.md                       # 主文檔 (10KB) ⭐
│   │   ├── 功能列表
│   │   ├── 快速開始
│   │   ├── 架構圖
│   │   ├── 命令參考
│   │   ├── 故障排除
│   │   └── 性能指標
│   │
│   ├── QUICKSTART.md                   # 5 分鐘開始 (4KB)
│   │   ├── 逐步安裝
│   │   ├── Discord Token 獲取
│   │   ├── 常見問題
│   │   └── 命令快速參考
│   │
│   ├── ADVANCED.md                     # 高級配置 (10KB)
│   │   ├── 模型自訂
│   │   ├── 交易信號調整
│   │   ├── 性能優化
│   │   ├── Kubernetes 部署
│   │   └── 監控和告警
│   │
│   ├── TROUBLESHOOTING.md              # 故障排除指南
│   │   ├── 常見錯誤
│   │   ├── 解決方案
│   │   └── 日誌分析
│   │
│   ├── INSTALL_CPU_ONLY.md             # CPU 安裝指南
│   ├── REQUIREMENTS.md                 # 依賴說明
│   ├── TRADING_SIGNALS_GUIDE.md        # 交易信號說明
│   └── PROJECT_STRUCTURE.md            # 本文件
│
├── 📁 運行時目錄 (自動創建)
│   ├── venv/                           # Python 虛擬環境
│   │   ├── Scripts/ 或 bin/
│   │   ├── lib/
│   │   └── pyvenv.cfg
│   │
│   ├── models/
│   │   └── hf_cache/                  # HuggingFace 模型緩存
│   │       ├── ADA_model_v8.pth
│   │       ├── BTC_model_v8.pth
│   │       ├── ETH_model_v8.pth
│   │       └── ...
│   │
│   └── logs/ (可選)
│       ├── bot.log
│       └── dashboard.log
│
└── 📊 Git 配置
    ├── .github/
    │   └── workflows/              # CI/CD 流程 (可選)
    │       ├── test.yml
    │       └── deploy.yml
    │
    └── .gitignore
```

## 📊 文件大小與依賴關係

| 文件 | 大小 | 依賴 | 用途 |
|------|------|------|------|
| bot.py | 14 KB | discord.py | Discord 機器人 |
| bot_predictor.py | 18 KB | torch, ccxt | 核心預測引擎 |
| dashboard.py | 3.6 KB | Flask | Web 後端 |
| dashboard.html | 16 KB | JavaScript | Web UI |
| requirements.txt | - | pip | 所有依賴 |
| run.sh / run.bat | 3-2.6 KB | bash/cmd | 啟動腳本 |
| Dockerfile | 0.8 KB | docker | 容器化 |
| docker-compose.yml | 1.1 KB | docker-compose | 編排 |

## 🔄 數據流向

```
Discord 用戶
    │
    ├─ .models, .predict, .signal
    │
    ▼
  bot.py
    │
    ├─ 發送命令給 bot_predictor
    │
    ▼
bot_predictor.py
    │
    ├─ HuggingFace (下載模型)
    │
    ├─ CCXT API (Binance/Bybit/OKX/Kraken)
    │   └─ 1H OHLCV 數據
    │
    ├─ 特徵歸一化
    │
    ├─ LSTM 模型推理 (CPU)
    │
    ├─ 生成 5 根 K 線預測
    │
    └─ 計算交易信號
        ├─ 入場點
        ├─ 止損/止贏
        ├─ 支撐/阻力
        └─ 信心度
    │
    ▼
Discord Bot (嵌入式消息)
Web Dashboard (HTTP)
```

## 🎯 主要代碼邏輯流程

### 1. 模型初始化
```
CryptoPredictor.__init__()
    ↓
await predictor.initialize()
    ├─ _get_hf_model_files() [HuggingFace]
    ├─ for each model_file:
    │   └─ _load_model(symbol, file)
    │       ├─ hf_hub_download()
    │       ├─ torch.load()
    │       ├─ _detect_model_config() ⭐ 自適應
    │       ├─ CryptoLSTMModel()
    │       └─ model.load_state_dict()
    └─ Store in self.models[symbol]
```

### 2. 自動預測循環
```
@tasks.loop(minutes=60)
    ├─ for each symbol in models:
    │   └─ predict_single(symbol)
    │       ├─ _fetch_ohlcv() [with fallback]
    │       ├─ _prepare_features() [normalize]
    │       ├─ model.forward() [inference]
    │       ├─ _generate_predictions() [5 candles]
    │       ├─ _analyze_trend() [confidence]
    │       ├─ _calculate_entry_points()
    │       └─ return prediction dict
    │
    ├─ Cache in prediction_cache[symbol]
    │
    └─ Update dashboard + Discord
```

### 3. Discord 命令處理
```
bot.command .predict BTC
    ├─ Check if BTC in prediction_cache
    ├─ Create embed
    │   ├─ Current price
    │   ├─ Trend
    │   ├─ Predicted prices (5)
    │   ├─ Support/Resistance
    │   └─ Confidence
    └─ Send to Discord
```

## 🔑 關鍵技術

### 模型維度自動檢測
```python
# bot_predictor.py 的 _detect_model_config()

# 從 checkpoint 權重推導架構:
weight_ih = checkpoint['lstm.weight_ih_l0']
input_dim = weight_ih.shape[1]
hidden_dim = weight_ih.shape[0] // 4

# 結果:
# - ADA: (44, 128, 2, 1)   → input=44, hidden=128
# - BTC: (44, 256, 2, 1)   → input=44, hidden=256
# - UNI: (25, 128, 1, 1)   → 維度不同,加載失敗,自動跳過
```

### 交易所 Fallback
```python
# bot_predictor.py 的 _fetch_ohlcv()

EXCHANGES = ['binance', 'bybit', 'okx', 'kraken']

for exchange_name in EXCHANGES:
    try:
        exchange = ccxt[exchange_name]()
        ohlcv = await exchange.fetch_ohlcv(pair, '1h')
        return ohlcv  # 成功
    except Exception as e:
        continue  # 失敗,嘗試下一個
```

### 自動化預測
```python
# bot.py 的 @tasks.loop(minutes=60)

# 每當新的 1H K 線完成時 (每小時頂部) 執行
# ├─ 自動並行預測 20+ 幣種
# ├─ 計算交易信號
# ├─ 更新 Web 儀表板
# └─ 可選: 發送 Discord 通知
```

## 📈 部署流程圖

```
本地開發
    ↓
編輯 .env
    ↓
run.sh / run.bat / python bot.py
    ↓
┌─────────────────┬────────────────────┐
│   Discord Bot   │  Web Dashboard     │
│   :auto         │  :5000             │
└─────────────────┴────────────────────┘
    ↓
Production (Docker)
    ↓
docker-compose up -d
    ↓
┌─────────────────┬────────────────────┐
│   crypto-bot    │  dashboard         │
│   (container)   │  (container)       │
└─────────────────┴────────────────────┘
    ↓
Cloud (Kubernetes/AWS/GCP)
```

## 🎨 前端架構

```html
dashboard.html
├─ Header (標題 + 時間戳)
├─ 統計卡片
│   ├─ Total Symbols
│   ├─ LONG Signals 🟢
│   ├─ SHORT Signals 🔴
│   └─ Avg Confidence
├─ 控制按鈕
│   ├─ Refresh
│   └─ Export JSON
├─ 篩選標籤
│   ├─ All Signals
│   ├─ LONG 📈
│   └─ SHORT 📉
└─ 預測卡片網格
    ├─ 卡片 #1
    │   ├─ 符號 + 信號類型
    │   ├─ 當前價格
    │   ├─ 預測價格 (H+1 ~ H+5)
    │   ├─ 進場/止損/止贏
    │   └─ 信心度條形圖
    ├─ 卡片 #2
    └─ ...

JavaScript 更新
├─ fetch('/api/predictions') 每 30 秒
├─ 動態渲染卡片
├─ 篩選和排序
└─ 導出為 JSON
```

---

**最後更新**: 2025-12-14  
**總代碼行數**: ~2,500+ 行  
**支持的幣種**: 20+  
**模型精度**: 自動檢測  
**推理速度**: 10ms/幣種 (CPU)  
