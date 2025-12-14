# 🤖 Crypto Discord Bot - 文件和架構需求

## 📋 Bot 所需的文件

### **必需文件結構**

```
crypto-discord-bot/
├── bot.py                          ✅ 主要 Bot 程式 (Discord 集成 + 預測邏輯)
├── requirements.txt                ✅ Python 依賴列表
├── .env                            ✅ 環境變數 (token, channel id)
├── .env.example                    📄 .env 範本
├── .gitignore                      📄 Git 忽略規則
├── deploy_on_vm.sh                 📄 部署腳本 (可選)
├── README.md                       📄 說明文檔
│
├── models/
│   └── saved/
│       ├── BTC_model_v8.pth       ✅ 模型檔案 (GPU訓練)
│       ├── ETH_model_v8.pth       ✅ 模型檔案 (GPU訓練)
│       ├── SOL_model_v8.pth       ✅ 模型檔案 (GPU訓練)
│       ├── ... (其他 20 個模型)
│       └── PEPE_model_v8.pth      ✅ 模型檔案 (GPU訓練)
│
├── bot_predictor.py               ✅ 模型加載和預測邏輯
├── bias_corrections_v8.json       ✅ 模型偏差校正值
│
├── venv/                           📁 虛擬環境 (自動建立)
├── bot.log                         📝 Bot 日誌 (自動生成)
└── logs/                           📁 日誌目錄 (可選)
```

---

## ✅ 必需檔案詳解

### **核心執行檔**

| 檔案 | 來源 | 功能 | 大小 |
|------|------|------|------|
| `bot.py` | 本倉庫 | Discord Bot + 預測迴圈 | ~20 KB |
| `bot_predictor.py` | HuggingFace | 模型載入和推理 | ~5 KB |
| `requirements.txt` | 本倉庫 | Python 依賴 | 1 KB |

### **環境配置檔**

| 檔案 | 用途 | 必需 |
|------|------|------|
| `.env` | Discord Token, Channel ID, HF Token | ✅ 必需 |
| `models/saved/*.pth` | 已訓練的模型 | ✅ 必需 |
| `bias_corrections_v8.json` | 每個模型的偏差校正值 | ✅ 必需 |

### **可選檔案**

| 檔案 | 用途 | 說明 |
|------|------|------|
| `deploy_on_vm.sh` | 自動部署 | 簡化 VM 部署流程 |
| `README.md` | 說明文檔 | 使用說明 |
| `.gitignore` | Git 配置 | 防止上傳敏感檔案 |

---

## 🔄 GPU vs CPU 配置

### **訓練時 (你已完成) 🎓**

```python
# 訓練使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ✅ 使用 GPU 加速訓練
# 儲存的 .pth 檔案包含完整模型
```

### **推理時 (Discord Bot) 🤖**

**Bot 應該用 CPU 推理！** 原因：

1. ✅ **成本低** - 不需要 GPU VM
2. ✅ **延遲可接受** - 每小時預測一次
3. ✅ **電費省** - CPU 功耗低
4. ✅ **穩定性** - CPU 推理更穩定

---

## 🛠️ Bot 中的 GPU/CPU 配置

### **目前的 bot_predictor.py**

應該是這樣：

```python
import torch

class BotPredictor:
    def __init__(self):
        # ✅ 推理用 CPU
        self.device = torch.device('cpu')
        
        # 載入所有模型
        self.models = {}
        for symbol in ['BTC', 'ETH', 'SOL', ...]:
            model = torch.load(f'models/saved/{symbol}_model_v8.pth')
            model = model.to(self.device)  # 移到 CPU
            model.eval()  # 推理模式
            self.models[symbol] = model
    
    def predict(self, symbol):
        # 在 CPU 上推理
        with torch.no_grad():  # 不計算梯度 (推理時不需要)
            prediction = self.models[symbol](input_tensor)
        return prediction
```

---

## 📥 完整檔案下載清單

### **需要從 HuggingFace 自動下載**

```bash
# Bot 啟動時自動執行:
# 1. 下載所有 .pth 模型
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="caizongxun/crypto-price-predictor-v8",
    allow_patterns=["models/*.pth"],
    local_dir="."
)

# 2. 下載 bot_predictor.py (如果本地沒有)
# 3. 下載 bias_corrections_v8.json
```

### **手動下載 (可選)**

如果自動下載失敗，可以：

```bash
# 用專門的下載腳本
python download_from_hf.py
```

---

## 🚀 完整部署檔案清單

### **最小化部署 (僅需)**

```
✅ bot.py
✅ requirements.txt
✅ .env (自己建立)
✅ models/saved/*.pth (自動下載)
✅ bot_predictor.py (自動下載)
✅ bias_corrections_v8.json (自動下載)
```

### **完整部署 (推薦)**

```
✅ bot.py
✅ requirements.txt
✅ .env
✅ .env.example
✅ .gitignore
✅ README.md
✅ REQUIREMENTS.md (本檔案)
✅ models/saved/*.pth
✅ bot_predictor.py
✅ bias_corrections_v8.json
```

---

## 📦 Python 環境需求

### **requirements.txt 包含**

```
discord.py>=2.3.0              # Discord Bot
huggingface_hub>=0.16.0        # 模型下載
torch>=2.0.0                   # PyTorch (CPU 版本)
pandas>=1.5.0                  # 資料處理
numpy>=1.23.0                  # 數值計算
scikit-learn>=1.2.0            # 機器學習工具
python-dotenv>=1.0.0           # .env 讀取
ccxt>=2.0.0                    # 交易所 API (可選)
requests>=2.28.0               # HTTP 請求
```

### **GPU vs CPU 的 PyTorch**

**目前安裝的是 CPU 版本：**
```
torch>=2.0.0  # CPU 版本 (~200MB)
```

**如果要 GPU 版本，改為：**
```
torch==2.0.0+cu118  # CUDA 11.8 (GPU)
# 或
torch==2.0.0+cu121  # CUDA 12.1 (GPU)
# 檔案會更大 (~2GB)
```

---

## 🔧 模型加載特殊處理

### **GPU 訓練 → CPU 推理的轉換**

```python
# ✅ 標準做法
import torch

# 1. 載入模型 (自動偵測原始裝置)
model = torch.load('models/saved/BTC_model_v8.pth')

# 2. 移到 CPU
model = model.to('cpu')
model.eval()  # 設為推理模式

# 3. 推理
with torch.no_grad():
    output = model(input_data)
```

### **注意事項**

1. ✅ **無需特殊處理** - PyTorch 自動處理裝置轉換
2. ✅ **模型檔案相同** - GPU 訓練的模型可直接在 CPU 推理
3. ✅ **速度可接受** - CPU 推理延遲 1-5 秒 (對於每小時一次的預測足夠)
4. ✅ **記憶體充足** - 20 個模型 (~2GB) 在 4GB+ RAM 的 VM 上無問題

---

## 📊 性能預估

### **CPU 推理性能**

| 指標 | 值 | 說明 |
|------|-----|------|
| 單個預測時間 | 1-3 秒 | CPU 推理 |
| 20 個幣種 | 20-60 秒 | 串行推理 |
| 記憶體使用 | 2-3 GB | 所有模型載入 |
| CPU 使用率 | 50-80% | 推理期間 |

### **與 GPU 的比較**

| 項目 | CPU | GPU |
|------|-----|-----|
| 推理速度 | 1-3 秒 | 0.1-0.5 秒 |
| 成本 | 低 | 高 |
| 電費 | 低 | 高 |
| 設置複雜度 | 簡單 | 複雜 |
| 穩定性 | 高 | 中 |

**結論：** 對於每小時一次的預測，**CPU 完全足夠！** ✅

---

## 🎯 檔案來源總結

### **本倉庫提供**
```
✅ bot.py
✅ requirements.txt
✅ .env.example
✅ .gitignore
✅ deploy_on_vm.sh
✅ README.md
```

### **HuggingFace 自動下載**
```
✅ models/saved/*.pth (20 個模型)
✅ bot_predictor.py
✅ bias_corrections_v8.json
```

### **你需要提供**
```
✅ .env (填入你的 tokens)
```

---

## ✅ 部署檢查清單

```bash
# 1. 檢查本倉庫檔案
ls -la bot.py requirements.txt .env.example

# 2. 檢查虛擬環境
ls -la venv/bin/python

# 3. 檢查依賴
pip list | grep discord
pip list | grep torch

# 4. 檢查模型下載
ls -la models/saved/ | wc -l  # 應該有 20+ 檔案

# 5. 檢查預測器
ls -la bot_predictor.py

# 6. 檢查偏差校正
ls -la bias_corrections_v8.json

# 7. 驗證 .env
grep DISCORD_BOT_TOKEN .env
grep HUGGINGFACE_TOKEN .env

# 8. 測試導入
python -c "import torch; print(torch.__version__)"
python -c "import discord; print(discord.__version__)"
```

---

## 🚀 快速參考

**最少需要：**
- bot.py ✅
- requirements.txt ✅
- .env ✅
- models/saved/*.pth (自動下載) ✅

**推理配置：**
- 使用 CPU ✅
- 無需特殊處理 ✅
- GPU 訓練的模型直接可用 ✅

---

**上次更新：** 2025-12-14

**狀態：** ✅ 生產就緒
