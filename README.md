# 🤖 Crypto Discord Bot

A Discord bot that fetches crypto price predictions from HuggingFace models and sends real-time notifications to your Discord server.

## ✨ Features

- **🤖 Automatic Predictions**: Automatically fetches crypto price predictions at configured intervals
- **💾 Model Management**: Automatically downloads and manages models from HuggingFace
- **📊 Rich Embeds**: Beautiful Discord embeds with prediction data
- **⚙️ Flexible Configuration**: Easy .env configuration for tokens and settings
- **🔄 Async Operations**: Non-blocking async operations for smooth performance
- **📝 Detailed Logging**: Comprehensive logging to file and console
- **🎯 Manual Commands**: Commands to manually trigger predictions

## 📋 Requirements

- Python 3.8+
- Discord.py 2.3+
- HuggingFace Hub
- PyTorch
- Pandas & Scikit-learn

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/caizongxun/crypto-discord-bot.git
cd crypto-discord-bot
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure .env

```bash
cp .env.example .env
nano .env  # Edit with your tokens
```

**Required Variables:**

```env
# Discord Configuration
DISCORD_BOT_TOKEN=your_discord_bot_token_here
DISCORD_CHANNEL_ID=your_discord_channel_id_here

# HuggingFace Configuration
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
HUGGINGFACE_REPO_ID=caizongxun/crypto-price-predictor-v8

# Bot Configuration
PREDICTION_INTERVAL=3600  # Seconds (3600 = 1 hour)
CRYPTO_SYMBOLS=BTC,ETH,SOL,BNB,XRP  # Comma-separated
```

### 5. Run Bot

```bash
python bot.py
```

## 📖 Configuration Details

### DISCORD_BOT_TOKEN

Get from [Discord Developer Portal](https://discord.com/developers/applications)

1. Create New Application
2. Go to "Bot" section
3. Copy "TOKEN"
4. Reset token if leaked

### DISCORD_CHANNEL_ID

1. Enable Developer Mode in Discord
2. Right-click channel → Copy ID

### HUGGINGFACE_TOKEN

Get from [HuggingFace Settings](https://huggingface.co/settings/tokens)

1. Create new token with read access
2. Copy to .env

### PREDICTION_INTERVAL

Seconds between predictions (default: 3600 = 1 hour)

```
300 = 5 minutes
1800 = 30 minutes
3600 = 1 hour
86400 = 24 hours
```

### CRYPTO_SYMBOLS

Comma-separated list of cryptocurrencies:

```
BTC,ETH,SOL,BNB,XRP,ADA,DOT,LINK,MATIC,AVAX
```

## 🎮 Bot Commands

### !predict [SYMBOL]

Manually trigger prediction for a symbol

```
!predict BTC
!predict ETH
!predict SOL
```

### !status

Check bot status and configuration

```
!status
```

## 📊 Prediction Output

Each prediction includes:

- **Current Price**: Latest market price
- **Predicted Price**: Model's price prediction
- **Change**: Percentage change from current
- **Direction**: Up (📈) / Down (📉) / Neutral (➡️)
- **Confidence**: Model confidence (0-100%)

## 🔧 VM Deployment

### SSH into VM

```bash
ssh user@vm_ip
```

### Clone and Setup

```bash
cd ~
mkdir crypto-discord-bot
cd crypto-discord-bot
git clone https://github.com/caizongxun/crypto-discord-bot.git .
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Configure .env

```bash
nano .env
# Add your tokens
```

### Run in Background (systemd - Recommended)

```bash
sudo cat > /etc/systemd/system/crypto-discord-bot.service << 'EOF'
[Unit]
Description=Crypto Discord Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/crypto-discord-bot
ExecStart=/home/ubuntu/crypto-discord-bot/venv/bin/python bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl start crypto-discord-bot
sudo systemctl enable crypto-discord-bot

# View logs
sudo journalctl -u crypto-discord-bot -f
```

### Run in Background (screen)

```bash
screen -S crypto-bot
python bot.py
# Press Ctrl+A then D to detach

# Re-attach
screen -r crypto-bot
```

### Run in Background (nohup)

```bash
nohup python bot.py > bot.log 2>&1 &

# View logs
tail -f bot.log
```

## 🐛 Troubleshooting

### Bot not connecting to Discord

```bash
# Check token
grep DISCORD_BOT_TOKEN .env

# Verify in Discord Developer Portal that:
# 1. Token hasn't expired
# 2. Bot has required permissions
# 3. Bot is in the server
```

### Models not downloading

```bash
# Check HuggingFace token
grep HUGGINGFACE_TOKEN .env

# Test connection
python -c "from huggingface_hub import list_repo_files; print(list_repo_files('caizongxun/crypto-price-predictor-v8', repo_type='model'))"
```

### Memory/Disk issues

```bash
# Check disk space
df -h

# Check memory
free -h

# Model size
du -sh models/
```

### No predictions appearing

```bash
# Check logs
tail -50 bot.log

# Verify bot is running
ps aux | grep bot.py

# Check Discord channel
# Make sure bot has permission to send messages
```

## 📂 Project Structure

```
crypto-discord-bot/
├── bot.py                 # Main bot script
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── .env                  # Your configuration (git ignored)
├── bot.log              # Bot logs
├── models/
│   └── saved/           # Downloaded models
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

## 📝 Logs

Bot logs are saved to `bot.log` and displayed in console.

```bash
# View recent logs
tail -100 bot.log

# Follow logs in real-time
tail -f bot.log

# Search for errors
grep ERROR bot.log
```

## 🔐 Security

⚠️ **Important:**

- Never commit `.env` file to git
- Never share your bot token
- Use strong, unique tokens
- Rotate tokens regularly
- Keep discord.py updated

## 📊 Model Information

**Model Name:** Crypto Price Predictor V8

**Architecture:**
- Bidirectional LSTM
- 2 stacked layers
- 44 technical indicators
- Bias correction per symbol

**Performance:**
- Average MAPE: < 0.05%
- Direction Accuracy: ~65-75%

**Supported Symbols:**
BTC, ETH, SOL, BNB, XRP, ADA, DOT, LINK, MATIC, AVAX, FTM, NEAR, ATOM, ARB, OP, LTC, DOGE, UNI, SHIB, PEPE

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## 📞 Support

For issues and questions:

1. Check [Troubleshooting](#-troubleshooting) section
2. Review bot logs
3. Check Discord permissions
4. Open an issue on GitHub

---

**Last Updated:** 2025-12-14

**Status:** ✅ Production Ready
