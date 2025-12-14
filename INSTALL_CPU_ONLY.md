# 🚀 CPU-Only Installation Guide (VM Deployment)

## ⚠️ 重要提示

**本 Bot 僅使用 CPU 版本 PyTorch，無需安裝 CUDA！**

- ✅ **無需 GPU**
- ✅ **無需 CUDA**
- ✅ **無需 cuDNN**
- ✅ **無需複雜配置**

---

## 🎯 為什麼只用 CPU？

| 比較項 | CPU | GPU |
|-------|-----|-----|
| **部署成本** | 低 ✅ | 高 ❌ |
| **安裝複雜度** | 簡單 ✅ | 複雜 ❌ |
| **預測速度** | 1-3秒 ✅ | 0.1秒 ❌ (不必要) |
| **每小時預測** | 足夠 ✅ | 過度 ❌ |
| **記憶體** | 2-3 GB ✅ | 4-6 GB ❌ |
| **電費** | 低 ✅ | 高 ❌ |

**結論：** 對於每小時一次的預測，**CPU 完全足夠！** ✅

---

## 📥 完整 CPU-Only 安裝步驟

### **Step 1: SSH 進 VM**

```bash
ssh user@vm_ip
cd ~/crypto-discord-bot
```

### **Step 2: 檢查 Python 版本**

```bash
# 確認 Python 3.8+
python3 --version

# 應該看到: Python 3.8.x 或更新
```

### **Step 3: 建立虛擬環境**

```bash
# 建立虛擬環境 (隔離環境，不影響系統 Python)
python3 -m venv venv

# 激活虛擬環境
source venv/bin/activate

# 確認激活 (命令行前應該有 (venv))
# (venv) user@vm:~/crypto-discord-bot$
```

### **Step 4: 升級 pip**

```bash
# 升級 pip (重要！舊版 pip 可能無法正確安裝 CPU 版本)
pip install --upgrade pip setuptools wheel

# 驗證
pip --version
# 應該看到最新版本
```

### **Step 5: 安裝 CPU-Only PyTorch**

```bash
# 方式 A: 自動安裝 (推薦)
# pip 會自動選擇 CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或方式 B: 從 requirements.txt 安裝 (更簡單)
pip install -r requirements.txt

# 驗證安裝
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# 應該看到:
# PyTorch 2.x.x+cpu
# CUDA Available: False  ✅ (這是正確的)
```

### **Step 6: 驗證其他依賴**

```bash
# 驗證所有必要的 packages
python -c "
import torch
import discord
import pandas
import numpy
from huggingface_hub import snapshot_download
print('✅ All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'Discord.py version: {discord.__version__}')
"
```

### **Step 7: 配置 .env**

```bash
# 複製範本
cp .env.example .env

# 編輯 .env
nano .env

# 填入:
# DISCORD_BOT_TOKEN=your_token
# DISCORD_CHANNEL_ID=your_channel_id
# HUGGINGFACE_TOKEN=hf_xxx
# HUGGINGFACE_REPO_ID=caizongxun/crypto-price-predictor-v8
# PREDICTION_INTERVAL=3600
# CRYPTO_SYMBOLS=
```

### **Step 8: 啟動 Bot**

```bash
# 確保虛擬環境激活
source venv/bin/activate

# 啟動 Bot
python bot.py

# 應該看到:
# ============================================================
# 🤖 Crypto Discord Bot - Starting
# ============================================================
# ✓ Found .env at: ...
# ✓ Configuration loaded successfully
# ...
# 🔍 Auto-detecting available models...
# ✓ Detected 20 unique symbols
# ✓ Bot logged in as YourBotName#1234
# ✓ Connected to channel: your-channel-name
# Downloading models from HuggingFace...
# ✓ All systems ready, starting prediction loop
```

**成功！** Bot 正在運行中 ✅

---

## 🧪 驗證 CPU-Only 配置

### **檢查 PyTorch 配置**

```python
import torch

# 檢查 CUDA 是否可用 (應該是 False)
print(f"CUDA Available: {torch.cuda.is_available()}")  # False ✅

# 檢查使用的設備
device = torch.device('cpu')
print(f"Device: {device}")  # cpu ✅

# 檢查 PyTorch 版本
print(f"PyTorch Version: {torch.__version__}")  # Should be +cpu
```

### **運行時檢查**

```bash
# 在 Bot 運行時，檢查日誌
tail -f bot.log

# 應該看到:
# ✓ Bot predictor loaded successfully
# ✓ All systems ready
# Starting prediction cycle for 20 symbols...
```

---

## 📊 預期安裝時間和大小

| 階段 | 耗時 | 大小 |
|------|------|------|
| 虛擬環境建立 | 10秒 | 50 MB |
| PyTorch CPU | 2-5分鐘 | 200 MB |
| 其他依賴 | 1-2分鐘 | 100 MB |
| 模型下載 (首次) | 5-15分鐘 | 1.5-2 GB |
| **總計** | **10-25分鐘** | **~2 GB** |

---

## 🚨 常見問題

### ❌ "RuntimeError: Cuda is not available"

**原因:** 試圖使用 GPU 但沒有 CUDA

**解決方案:**
```python
# 確保使用 CPU
device = torch.device('cpu')
model = model.to(device)
```

### ❌ "ImportError: No module named 'torch'"

**原因:** 沒有激活虛擬環境或安裝失敗

**解決方案:**
```bash
# 檢查虛擬環境
source venv/bin/activate

# 重新安裝
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### ❌ "pip: command not found"

**原因:** 沒有激活虛擬環境

**解決方案:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## 💡 CPU 版本注意事項

### **優點**
- ✅ 安裝快速簡單
- ✅ 檔案大小小 (~200MB)
- ✅ 無需複雜配置
- ✅ 相容所有 Linux/Windows/Mac
- ✅ 記憶體使用少
- ✅ 電費低

### **性能**
- 單個預測: 1-3 秒
- 20 個幣種: 20-60 秒
- 每小時一次: ✅ 足夠
- 實時交易: ❌ 不適合 (但不是本 Bot 的用途)

### **最佳實踐**

```python
# ✅ 正確用法
import torch

device = torch.device('cpu')
model = torch.load('model.pth')
model = model.to(device)
model.eval()

with torch.no_grad():
    output = model(input_data)
```

---

## 🔄 後續維護

### **更新 Bot**

```bash
# 進入虛擬環境
source venv/bin/activate

# 更新代碼
git pull origin main

# 重啟 Bot
sudo systemctl restart crypto-discord-bot
```

### **更新依賴**

```bash
# 進入虛擬環境
source venv/bin/activate

# 升級所有包
pip install --upgrade -r requirements.txt
```

---

## ✅ 部署檢查清單

在啟動 Bot 前檢查：

- [ ] Python 3.8+ 已安裝
- [ ] 虛擬環境已建立 (`venv` 目錄存在)
- [ ] 虛擬環境已激活 (命令行有 `(venv)`)
- [ ] PyTorch CPU 版本已安裝 (`torch.__version__` 包含 `+cpu`)
- [ ] 所有依賴已安裝 (`pip list` 顯示所有包)
- [ ] `.env` 已配置 (包含所有必要的 tokens)
- [ ] `HUGGINGFACE_TOKEN` 有效
- [ ] `DISCORD_BOT_TOKEN` 有效
- [ ] `DISCORD_CHANNEL_ID` 正確
- [ ] 網路連接穩定

---

## 🎉 完成！

如果一切順利，你應該看到：

```
🤖 Crypto Discord Bot - Starting
============================================================
✓ Found .env at: /home/user/crypto-discord-bot/.env
✓ Configuration loaded successfully
🔍 Auto-detecting available models...
✓ Detected 20 unique symbols
✓ Bot logged in as YourBot#1234
✓ Connected to channel: your-channel
✓ All systems ready, starting prediction loop
```

**Bot 正在運行！** 🎉

---

**最後更新:** 2025-12-14

**狀態:** ✅ CPU-Only Ready
