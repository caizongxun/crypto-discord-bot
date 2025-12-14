# 🚀 Crypto Discord Bot - Real-time Price Prediction

A fully automated Discord bot that:
- ✅ Automatically downloads LSTM models from HuggingFace
- ✅ Detects model architecture (adaptive dimensions)
- ✅ Fetches real-time 1H K-line data from Binance (with fallback)
- ✅ Predicts next 3-5 candles price movement
- ✅ Generates trading signals (LONG/SHORT) with entry/exit points
- ✅ Provides a beautiful web dashboard for visualization

## 📋 Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/caizongxun/crypto-discord-bot.git
cd crypto-discord-bot

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```
DISCORD_TOKEN=your_discord_bot_token_here
DASHBOARD_URL=http://localhost:5000
DASHBOARD_PORT=5000
FLASK_ENV=development
```

### 3. Run the Bot

```bash
# Terminal 1: Run Discord Bot
python bot.py

# Terminal 2: Run Web Dashboard (optional)
python dashboard.py
```

## 🎮 Discord Bot Commands

### Model Management

```
.models          # List all loaded models with detailed info
.reload          # Reload all models from HuggingFace
.test BTC        # Test a single model
```

### Predictions & Signals

```
.predict         # Show all predictions (or .predict BTC for specific)
.signal          # Show all trading signals (sorted by confidence)
.stats           # Display bot statistics
```

### Dashboard

```
.dashboard       # Get link to web dashboard
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────┐
│         HuggingFace Model Hub               │
│    zongowo111/crypto_model/                 │
│  (ADA_model_v8.pth, BTC_model_v8.pth, ...)│
└────────────────┬────────────────────────────┘
                 │
         ┌───────▼────────┐
         │ bot_predictor  │  🤖 Auto-detects:
         │                │  • Input dimensions
         │ CryptoPredictor│  • Hidden sizes
         │                │  • Model architecture
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌────────┐   ┌────────┐   ┌──────────┐
│ Binance│   │  Bybit │   │ Web API  │
│ (1H)   │   │  OKX   │   │ Prices  │
│ OHLCV  │   │ Kraken │   │ Features │
└────────┘   └────────┘   └──────────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
         ┌────────▼────────┐
         │  Predictions    │
         │ & Signals Gen   │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
 ┌──────────┐ ┌──────────┐  ┌──────────┐
 │ Discord  │ │  Web     │  │  Cache   │
 │   Bot    │ │Dashboard │  │ (JSON)   │
 └──────────┘ └──────────┘  └──────────┘
```

## 📊 Model Architecture Detection

The bot automatically detects model configurations:

```python
# From checkpoint weights, detects:
input_features  = checkpoint['lstm.weight_ih_l0'].shape[1]
hidden_size     = checkpoint['lstm.weight_ih_l0'].shape[0] // 4
num_layers      = max(layer_num from lstm.weight_hh_l*)
bidirectional   = 'lstm.weight_ih_l0_reverse' in checkpoint
output_features = checkpoint['regressor.*.weight'].shape[0]
```

**Example output:**

```
✓ BTC loaded successfully
  Input: 44 | Hidden: 128 | Output: 1
✓ ETH loaded successfully
  Input: 44 | Hidden: 256 | Output: 1
```

## 🎯 Trading Signal Generation

### Entry Point Calculation

For each prediction, the bot calculates:

1. **UPTREND**: Entry at lowest predicted price, SL 3%, TP 5%
2. **DOWNTREND**: Entry at highest predicted price, SL 3%, TP 5%
3. **Support/Resistance**: Last 50 candle highs and lows

### Confidence Scoring

```python
confidence = {
    'trend_agreement': 0.7,        # If historical + predicted agree
    'momentum_factor': 0.0-0.3,    # RSI-like momentum
    'final_score': 0.5-0.99        # Combined score
}
```

## 📈 Prediction Pipeline

```
1. Fetch 100 latest 1H candles (O, H, L, C, V)
2. Normalize using min-max scaling
3. Feed into LSTM model (batch size 1)
4. Get price prediction output
5. Generate 5 future prices (exponential influence)
6. Analyze trend (historical SMA + predicted direction)
7. Calculate entry/exit points
8. Generate confidence score
9. Return complete trading signal
```

## 🔄 Automatic Prediction Loop

The bot runs predictions every 60 minutes (new 1H candle):

```
08:57:29 - Starting cycle [20 symbols]
08:57:34 - BTC: UPTREND | Confidence: 87%
08:57:39 - ETH: UPTREND | Confidence: 72%
08:57:44 - SOL: NEUTRAL | Confidence: 58%
...
09:00:12 - ✓ Cycle complete (17 successful, 2 failed)
09:00:12 - Waiting for next 1H candle...
```

## 🌐 Web Dashboard Features

### Real-time Updates

- ✅ Live prediction cards (refreshes every 30s)
- ✅ Filter by signal type (ALL / LONG / SHORT)
- ✅ Click-to-copy trading levels
- ✅ Confidence progress bars
- ✅ Support/resistance display

### API Endpoints

```
GET /api/predictions          # All predictions
GET /api/predictions/<symbol> # Specific symbol
GET /api/signals              # Trading signals (sorted by confidence)
GET /api/statistics           # Summary statistics
```

### Export Data

```javascript
// Export all predictions as JSON
document.querySelector('button[onclick="exportData()"]').click()
```

## ⚙️ Configuration

### Model Cache

Models are automatically cached in `./models/hf_cache/`:

```
models/hf_cache/
├── ADA_model_v8.pth
├── BTC_model_v8.pth
├── ETH_model_v8.pth
└── ...
```

### Exchange Fallback Order

If Binance is blocked in your region:

```python
EXCHANGES = ['binance', 'bybit', 'okx', 'kraken']
```

Bot automatically tries next exchange on failure.

### LSTM Hyperparameters

Adjustable in `bot_predictor.py`:

```python
DEFAULT_LOOKBACK = 100          # Historical candles to use
PREDICTION_HORIZON = 5          # Candles to predict ahead
CONFIDENCE_THRESHOLD = 0.5      # Minimum confidence to display
```

## 📝 Example Discord Output

```
💰 BTC/USDT Prediction
📈 3-5 Candle Trend: UPTREND
Current Price: $45,234.50

H+1: $45,520.80
H+2: $45,840.20
H+3: $46,180.50
H+4: $46,540.30
H+5: $46,920.70

Support: $44,800.00
Resistance: $46,500.00
Confidence: 85%

---
🎯 Trading Signal: LONG
Entry: $45,200.00
Stop Loss: $43,844.00
 Take Profit: $47,460.00
```

## 🐛 Troubleshooting

### Models won't load

```
✗ UNI: Error(s) in loading state_dict
size mismatch for lstm.weight_ih_l0
```

**Solution**: Model was trained with different input dimensions.

```bash
# Check model details
python bot_predictor.py
```

### Binance API blocked (451 error)

**Bot automatically handles this** - tries:
1. Binance
2. Bybit
3. OKX
4. Kraken

If still failing, use VPN or check exchange status.

### Discord bot doesn't respond

1. Check `DISCORD_TOKEN` in `.env`
2. Verify bot has message permissions
3. Ensure bot is in server
4. Check logs:

```bash
grep -i "error" bot.log
```

## 📦 Project Structure

```
crypto-discord-bot/
├── bot.py                    # Main Discord bot
├── bot_predictor.py          # Prediction engine
├── dashboard.py              # Flask web server
├── templates/
│   └── dashboard.html        # Web UI
├── models/
│   └── hf_cache/             # Downloaded models
├── requirements.txt          # Python dependencies
├── .env.example              # Configuration template
├── README.md                 # This file
└── TROUBLESHOOTING.md        # Common issues
```

## 🚀 Deployment

### Local Testing

```bash
python bot.py        # Terminal 1
python dashboard.py  # Terminal 2 (optional)
```

### Docker (Coming Soon)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "bot.py"]
```

### Cloud Deployment

Supported platforms:
- ✅ Linux VPS (Ubuntu, Debian)
- ✅ Windows Server
- ✅ GitHub Actions
- ✅ Docker containers

See [INSTALL_CPU_ONLY.md](INSTALL_CPU_ONLY.md) for detailed setup.

## 📊 Performance Metrics

**Average Prediction Time**
- Model loading: ~50ms (first time), <1ms (cached)
- Data fetching: ~5s (exchange API)
- Prediction: ~10ms (CPU)
- Total per symbol: ~5.2s

**Throughput**
- 20 symbols: ~2 minutes per cycle
- 50 symbols: ~5 minutes per cycle

## 🔐 Security Notes

- ✅ No private keys stored in code
- ✅ API keys in `.env` (excluded from git)
- ✅ Discord token secured
- ⚠️ Dashboard accessible on LAN (add authentication for production)

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please:

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📞 Support

- 🐛 Report bugs: [Issues](https://github.com/caizongxun/crypto-discord-bot/issues)
- 💬 Discussions: [Discussions](https://github.com/caizongxun/crypto-discord-bot/discussions)
- 📧 Email: caizongxun@example.com

---

**⭐ If this project helps you, please give it a star!**
