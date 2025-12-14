# 🔧 Advanced Configuration Guide

## 📋 目錄

1. [模型自訂](#模型自訂)
2. [交易信號調整](#交易信號調整)
3. [性能優化](#性能優化)
4. [多交易所配置](#多交易所配置)
5. [監控和日誌](#監控和日誌)
6. [部署優化](#部署優化)

## 🤖 模型自訂

### 使用自訂模型

如果你有自己訓練的模型，可以直接放在 HuggingFace:

```python
# bot_predictor.py 中修改
HF_REPO = "your_username/your_model_repo"  # 你的 HuggingFace 倉庫
MODEL_PATTERN = "_model_v8.pth"  # 你的模型文件名模式
```

### 模型架構檢測

自動檢測支援的維度:

```python
# bot_predictor.py 中的 _detect_model_config() 方法

# 支援檢測:
- input_features (from lstm.weight_ih_l0)
- hidden_size (from lstm weights)
- num_layers (from lstm.weight_hh_l*)
- bidirectional (from lstm.weight_*_reverse)
- output_features (from regressor layers)
```

如果模型維度檢測失敗，手動指定:

```python
# 在 bot_predictor.py 中添加特定檢查
def _detect_model_config(self, checkpoint: Dict) -> Dict:
    config = {
        'input_features': 44,     # 手動設定
        'hidden_size': 128,
        'num_layers': 2,
        'output_features': 1,
        'bidirectional': False
    }
    return config
```

## 💹 交易信號調整

### 修改進出場邏輯

編輯 `bot_predictor.py` 中的 `_calculate_entry_points()`:

```python
def _calculate_entry_points(
    self,
    current_price: float,
    predicted_prices: List[float],
    trend: str
) -> Tuple[float, float, float]:
    
    if trend == 'UPTREND':
        # 自訂上升趨勢的進出場
        entry = min(predicted_prices) * 0.98  # 進場價格
        stop_loss = entry * 0.95  # 修改: 5% -> 3%
        take_profit = entry * 1.10  # 修改: 5% -> 10%
    
    elif trend == 'DOWNTREND':
        # 自訂下降趨勢的進出場
        entry = max(predicted_prices) * 1.02
        stop_loss = entry * 1.05
        take_profit = entry * 0.90
    
    return entry, stop_loss, take_profit
```

### 修改信心度計算

```python
def _analyze_trend(self, ohlcv_data, predicted_prices):
    # 增加信心度權重
    confidence = 0.7  # 基礎
    confidence += momentum * 0.5  # 從 0.3 增加到 0.5
    confidence += prediction_accuracy * 0.2  # 新增預測準確度
    
    return trend, min(0.99, confidence)
```

## ⚡ 性能優化

### 減少預測延遲

```python
# bot_predictor.py
DEFAULT_LOOKBACK = 50  # 從 100 減少到 50 (2倍加速)

# 或針對特定幣種
async def _prepare_features(self, ohlcv_data, lookback=50):
    # 減少歷史數據 = 更快的預測
    pass
```

### 並行預測

```python
# bot.py 中的 prediction_loop()

@tasks.loop(minutes=60)
async def prediction_loop():
    # 並行處理所有幣種
    tasks = [
        predictor.predict_single(symbol)
        for symbol in list(predictor.models.keys())
    ]
    
    results = await asyncio.gather(*tasks)
    
    # 並行 = 20個幣種同時預測而不是順序
```

### 禁用不需要的功能

```python
# bot.py

# 禁用 Web 儀表板
# (註釋掉 dashboard.py 的啟動)

# 禁用特定命令
# @bot.command(name='models')
# async def cmd_list_models(ctx):
#     pass  # 被禁用
```

## 🌍 多交易所配置

### 添加新交易所

```python
# bot_predictor.py

EXCHANGES = ['binance', 'bybit', 'okx', 'kraken', 'coinbase']

# 或按優先級排序
EXCHANGES = {
    'BTC': ['binance', 'coinbase'],  # BTC 使用這兩個
    'ALT': ['bybit', 'okx'],          # 其他幣種用這兩個
    'default': ['binance', 'kraken']  # 默認順序
}
```

### 自訂交易所設置

```python
async def _fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
    for exchange_name in self.exchange_fallback:
        try:
            exchange_config = {
                'rateLimit': 1000,  # 請求速率限制
                'enableRateLimit': True,
                'timeout': 30000,  # 超時 30 秒
            }
            
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class(exchange_config)
            
            # ...
        except Exception as e:
            continue
```

## 📊 監控和日誌

### 啟用詳細日誌

```python
# bot.py 的開始

import logging

logging.basicConfig(
    level=logging.DEBUG,  # 從 INFO 改為 DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),  # 保存到文件
        logging.StreamHandler()  # 也顯示在控制台
    ]
)
```

### 性能監控

```python
# 在 bot_predictor.py 中添加

import time

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def record(self, key: str, duration: float):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(duration)
    
    def report(self):
        for key, durations in self.metrics.items():
            avg = sum(durations) / len(durations)
            max_time = max(durations)
            print(f"{key}: avg={avg:.2f}ms, max={max_time:.2f}ms")

monitor = PerformanceMonitor()

# 使用
start = time.time()
result = await predictor.predict_single(symbol)
monitor.record(f"predict_{symbol}", (time.time() - start) * 1000)
```

### 錯誤追蹤

```python
# 集中錯誤處理

class ErrorHandler:
    def __init__(self):
        self.errors = {}
    
    def log_error(self, error_type: str, error: Exception):
        if error_type not in self.errors:
            self.errors[error_type] = []
        self.errors[error_type].append({
            'time': datetime.utcnow(),
            'message': str(error)
        })
    
    def get_summary(self):
        return {key: len(v) for key, v in self.errors.items()}
```

## 🚀 部署優化

### Docker 資源限制

```yaml
# docker-compose.yml

services:
  crypto-bot:
    # ...
    deploy:
      resources:
        limits:
          cpus: '1'  # 限制 1 CPU
          memory: 2G  # 限制 2GB 內存
        reservations:
          cpus: '0.5'
          memory: 1G
```

### Kubernetes 部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crypto-bot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: crypto-bot
  template:
    metadata:
      labels:
        app: crypto-bot
    spec:
      containers:
      - name: crypto-bot
        image: crypto-bot:latest
        env:
        - name: DISCORD_TOKEN
          valueFrom:
            secretKeyRef:
              name: discord-secret
              key: token
        resources:
          limits:
            cpu: 1
            memory: 2Gi
          requests:
            cpu: 500m
            memory: 1Gi
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import torch; print('OK')"
          initialDelaySeconds: 30
          periodSeconds: 10
```

### 零停機部署

```bash
#!/bin/bash

# 新建立副本
docker-compose up -d crypto-bot-v2

# 等待準備就緒
sleep 30

# 檢查健康狀態
if docker-compose exec -T crypto-bot-v2 python bot.py --health-check; then
    # 停止舊版本
    docker-compose down crypto-bot
    # 重命名新版本
    docker-compose rename crypto-bot-v2 crypto-bot
else
    # 回滾
    docker-compose down crypto-bot-v2
fi
```

## 🔐 安全性強化

### 密鑰管理

```python
# 使用環境變量而不是硬編碼
import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN must be set")
```

### API 速率限制

```python
from functools import wraps
import asyncio

def rate_limit(calls_per_second: int):
    min_interval = 1 / calls_per_second
    last_called = [0]
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            elapsed = asyncio.get_event_loop().time() - last_called[0]
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            last_called[0] = asyncio.get_event_loop().time()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(calls_per_second=10)
async def predict_single(self, symbol: str):
    # ...
    pass
```

## 📈 指標和告警

### Prometheus 集成

```python
from prometheus_client import Counter, Histogram, Gauge

# 定義指標
prediction_total = Counter(
    'predictions_total',
    'Total predictions made',
    ['symbol', 'trend']
)

prediction_duration = Histogram(
    'prediction_duration_seconds',
    'Prediction duration',
    ['symbol']
)

models_loaded = Gauge(
    'models_loaded',
    'Number of loaded models'
)

# 使用
with prediction_duration.labels(symbol=symbol).time():
    result = await predictor.predict_single(symbol)

prediction_total.labels(symbol=symbol, trend=result['trend']).inc()
models_loaded.set(len(predictor.models))
```

## 🔄 自動更新

### 檢查新模型

```python
@tasks.loop(hours=1)  # 每小時檢查一次
async def check_for_new_models():
    try:
        current_files = set(predictor.model_info.keys())
        new_files = await predictor._get_hf_model_files()
        new_symbols = {predictor._extract_symbol(f) for f in new_files}
        
        added = new_symbols - current_files
        if added:
            logger.info(f"Found new models: {added}")
            for symbol in added:
                await predictor._load_model(symbol, f"{symbol}{MODEL_PATTERN}")
    except Exception as e:
        logger.error(f"Error checking for new models: {e}")
```

---

**提示**: 所有這些優化都是可選的。開始時使用默認設置，然後根據需要進行調整。
