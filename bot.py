#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Discord Bot

Downloads models from HuggingFace and sends crypto price predictions to Discord
Automatically detects all available models in models/ directory
Fetches real-time data from multiple exchanges and generates trading signals

Usage:
  python bot.py

Requirements:
  - .env file with Discord and HuggingFace tokens
  - Python 3.8+
  - discord.py
  - huggingface_hub
  - torch
  - pandas
  - scikit-learn
  - ccxt (for Binance data)
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
import discord
from discord.ext import commands, tasks
from datetime import datetime
import traceback
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelDetector:
    """自動偵測和提取模型幣種"""
    
    @staticmethod
    def detect_symbols_from_models():
        """
        從 models/ 目錄自動偵測所有可用模型
        提取模型檔名中的幣種 (例如: BTC_model_v8.pth -> BTC)
        """
        models_dir = Path('models')  # 改成 models（不是 models/saved）
        symbols = set()
        
        if not models_dir.exists():
            logger.warning(f"⚠️  Models directory not found: {models_dir}")
            return []
        
        # 掃描所有 .pth 檔案
        model_files = list(models_dir.glob('*.pth'))
        
        if not model_files:
            logger.warning(f"⚠️  No model files found in {models_dir}")
            return []
        
        logger.info(f"\n🔍 Found {len(model_files)} model files:")
        
        for model_file in model_files:
            filename = model_file.stem  # 不含副檔名
            
            # 嘗試從檔名中提取幣種
            # 支援的格式: BTC_model_v8, BTC_model, btc_model_v8, BTC_v8 等
            # 第1步: 嘗試正一的格式
            match = re.match(r'^([A-Za-z]+)', filename)
            
            if match:
                symbol = match.group(1).upper()
                
                # 第2步: 檢查是否是有效的幣種
                # 也接受整數作为幣種（不是BTC但是2345）
                if len(symbol) <= 6 and not symbol.isdigit():
                    symbols.add(symbol)
                    logger.info(f"  ✓ {filename} -> {symbol}")
                else:
                    logger.warning(f"  ⚠️  Invalid symbol extracted from {filename}: {symbol}")
            else:
                logger.warning(f"  ⚠️  Could not extract symbol from {filename}")
        
        sorted_symbols = sorted(list(symbols))
        logger.info(f"✓ Detected {len(sorted_symbols)} unique symbols: {', '.join(sorted_symbols)}")
        
        if len(sorted_symbols) == 0:
            logger.warning("⚠️  No valid symbols detected. Make sure model files are named like 'BTC_model_v8.pth'")
        
        return sorted_symbols


class Config:
    """Load and store configuration from .env"""
    
    @staticmethod
    def find_env_file():
        """
        自動搜尋 .env 檔案
        """
        search_paths = [
            Path.cwd() / ".env",
            Path(__file__).parent / ".env",
            Path(__file__).parent.parent / ".env",
            Path.home() / ".env",
        ]
        
        for env_path in search_paths:
            if env_path.exists():
                logger.info(f"✓ Found .env at: {env_path}")
                return str(env_path)
        
        logger.warning("⚠️  .env file not found in standard locations")
        return None
    
    @staticmethod
    def read_env_file(env_path):
        """
        強化版 .env 檔案讀取
        """
        env_dict = {}
        
        try:
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(env_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.info(f"✓ Successfully read .env with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error("✗ Could not read .env file with any encoding")
                return env_dict
            
            for line in content.split('\n'):
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    
                    env_dict[key] = value
            
            logger.info(f"✓ Parsed {len(env_dict)} variables from .env")
            return env_dict
        
        except Exception as e:
            logger.error(f"✗ Error reading .env file: {e}")
            return env_dict
    
    @staticmethod
    def load():
        """
        Load configuration from .env
        自動偵測模型幣種
        """
        env_file = Config.find_env_file()
        if env_file:
            logger.info(f"Loading environment from: {env_file}")
            env_dict = Config.read_env_file(env_file)
            
            for key, value in env_dict.items():
                os.environ[key] = value
            
            load_dotenv(env_file, override=True, encoding='utf-8')
        else:
            logger.warning("⚠️  No .env file found, trying system environment")
            load_dotenv()
        
        # Required Discord config
        discord_token = os.getenv('DISCORD_BOT_TOKEN')
        channel_id = os.getenv('DISCORD_CHANNEL_ID')
        
        if not discord_token:
            logger.error("✗ DISCORD_BOT_TOKEN not found in .env")
            raise ValueError("DISCORD_BOT_TOKEN is required")
        
        if not channel_id:
            logger.error("✗ DISCORD_CHANNEL_ID not found in .env")
            raise ValueError("DISCORD_CHANNEL_ID is required")
        
        # Optional HuggingFace config
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        hf_repo_id = os.getenv('HUGGINGFACE_REPO_ID', 'zongowo111/crypto_model')
        
        # Bot config
        prediction_interval = int(os.getenv('PREDICTION_INTERVAL', '3600'))
        
        # 自動偵測模型幣種
        logger.info("\n🔍 Auto-detecting available models...")
        auto_detected_symbols = ModelDetector.detect_symbols_from_models()
        
        # 如果有手動配置的幣種，就使用手動配置；否則使用自動偵測
        manual_symbols = os.getenv('CRYPTO_SYMBOLS')
        if manual_symbols and manual_symbols.strip() and manual_symbols != 'BTC,ETH,SOL,BNB,XRP':
            crypto_symbols = [s.strip().upper() for s in manual_symbols.split(',')]
            logger.info(f"✓ Using manually configured symbols ({len(crypto_symbols)}): {', '.join(crypto_symbols)}")
        elif auto_detected_symbols and len(auto_detected_symbols) > 0:
            crypto_symbols = auto_detected_symbols
            logger.info(f"✓ Using auto-detected symbols ({len(crypto_symbols)}): {', '.join(crypto_symbols)}")
        else:
            # 預設值 (如果沒有自動偵測到)
            logger.warning("⚠️  No models found, using default symbols")
            crypto_symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
            logger.info(f"🚧 Using {len(crypto_symbols)} default symbols: {', '.join(crypto_symbols)}")
        
        logger.info(f"\n✓ Configuration loaded successfully")
        logger.info(f"  Discord Channel: {channel_id}")
        logger.info(f"  HuggingFace Repo: {hf_repo_id}")
        logger.info(f"  Prediction Interval: {prediction_interval}s ({prediction_interval//3600}h)")
        logger.info(f"  Crypto Symbols ({len(crypto_symbols)}): {', '.join(sorted(crypto_symbols))}")
        
        return {
            'discord_token': discord_token,
            'channel_id': int(channel_id),
            'hf_token': hf_token,
            'hf_repo_id': hf_repo_id,
            'prediction_interval': prediction_interval,
            'crypto_symbols': crypto_symbols
        }


class ModelManager:
    """Manage model downloads and predictions"""
    
    def __init__(self, hf_token, hf_repo_id):
        self.hf_token = hf_token
        self.hf_repo_id = hf_repo_id
        self.bot_predictor = None
        self.ready = False
    
    async def initialize(self):
        """
        Initialize model manager and load models
        """
        try:
            logger.info("\n🚀 Initializing model manager...")
            
            # Check if models exist
            models_dir = Path('models')
            if models_dir.exists() and len(list(models_dir.glob('*.pth'))) > 0:
                model_count = len(list(models_dir.glob('*.pth')))
                logger.info(f"✓ Found {model_count} models locally")
            else:
                logger.warning(f"⚠️  No models found in {models_dir}")
            
            # Import and initialize predictor
            logger.info("Loading bot predictor...")
            try:
                from bot_predictor import BotPredictor
                self.bot_predictor = BotPredictor()
                self.ready = True
                logger.info("✓ Bot predictor loaded successfully")
            except ImportError:
                logger.error("✗ bot_predictor.py not found")
                logger.info("  Downloading from HuggingFace...")
                await self._download_bot_predictor()
                from bot_predictor import BotPredictor
                self.bot_predictor = BotPredictor()
                self.ready = True
                logger.info("✓ Bot predictor loaded successfully")
        
        except Exception as e:
            logger.error(f"✗ Failed to initialize model manager: {e}")
            logger.error(traceback.format_exc())
            self.ready = False
    
    async def _download_bot_predictor(self):
        """
        Download bot_predictor.py from HuggingFace
        """
        try:
            from huggingface_hub import hf_hub_download
            
            logger.info("Downloading bot_predictor.py...")
            
            hf_hub_download(
                repo_id=self.hf_repo_id,
                filename="bot_predictor.py",
                repo_type="model",
                local_dir=".",
                token=self.hf_token
            )
            
            logger.info("✓ bot_predictor.py downloaded successfully")
        
        except Exception as e:
            logger.error(f"✗ bot_predictor.py download failed: {e}")
            raise
    
    async def predict(self, symbol):
        """
        Get prediction for a symbol (with real-time data fetching)
        """
        if not self.ready or not self.bot_predictor:
            logger.warning(f"Model manager not ready, skipping prediction for {symbol}")
            return None
        
        try:
            # Call the async predict method
            prediction = await self.bot_predictor.predict(symbol, '1h')
            return prediction
        
        except Exception as e:
            logger.error(f"✗ Prediction failed for {symbol}: {e}")
            return None


class CryptoPredictorBot(commands.Cog):
    """Discord bot cog for crypto predictions"""
    
    def __init__(self, bot, config):
        self.bot = bot
        self.config = config
        self.model_manager = ModelManager(
            config['hf_token'],
            config['hf_repo_id']
        )
        self.channel = None
    
    @commands.Cog.listener()
    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f"\n🤖 Bot logged in as {self.bot.user}")
        
        # Get channel
        self.channel = self.bot.get_channel(self.config['channel_id'])
        if not self.channel:
            logger.error(f"✗ Channel {self.config['channel_id']} not found")
            return
        
        logger.info(f"✓ Connected to channel: {self.channel.name}")
        
        # Initialize model manager
        await self.model_manager.initialize()
        
        if self.model_manager.ready:
            logger.info("✓ All systems ready, starting prediction loop")
            self.prediction_loop.start()
        else:
            logger.error("✗ Model manager not ready, prediction loop not started")
    
    @tasks.loop(seconds=None)
    async def prediction_loop(self):
        """
        Main prediction loop - runs at configured interval
        Fetches real-time data and generates trading signals
        """
        try:
            if not self.channel or not self.model_manager.ready:
                return
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 Starting prediction cycle for {len(self.config['crypto_symbols'])} symbols...")
            logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info(f"{'='*80}")
            
            predictions = {}
            for i, symbol in enumerate(self.config['crypto_symbols'], 1):
                logger.info(f"\n[{i}/{len(self.config['crypto_symbols'])}] Processing {symbol}...")
                prediction = await self.model_manager.predict(symbol)
                if prediction:
                    predictions[symbol] = prediction
                await asyncio.sleep(2)  # Rate limiting (2 sec between requests)
            
            if predictions:
                await self._send_trading_signals(predictions)
                logger.info(f"\n✓ Prediction cycle completed for {len(predictions)} symbols")
            else:
                logger.warning("No successful predictions this cycle")
        
        except Exception as e:
            logger.error(f"✗ Prediction loop error: {e}")
            logger.error(traceback.format_exc())
    
    @prediction_loop.before_loop
    async def before_prediction_loop(self):
        """
        Wait until bot is ready before starting prediction loop
        """
        await self.bot.wait_until_ready()
        # Set the interval
        self.prediction_loop.change_interval(
            seconds=self.config['prediction_interval']
        )
    
    async def _send_trading_signals(self, predictions):
        """
        Send trading signals to Discord with detailed information
        """
        try:
            for symbol, signal in predictions.items():
                if not signal:
                    continue
                
                # Determine color based on signal type
                if 'STRONG_BUY' in signal.get('signal_type', ''):
                    color = discord.Color.green()
                elif 'BUY' in signal.get('signal_type', ''):
                    color = discord.Color.from_rgb(0, 200, 100)
                elif 'STRONG_SELL' in signal.get('signal_type', ''):
                    color = discord.Color.red()
                elif 'SELL' in signal.get('signal_type', ''):
                    color = discord.Color.from_rgb(200, 0, 0)
                else:
                    color = discord.Color.blue()
                
                embed = discord.Embed(
                    title=f"🎯 {symbol} Trading Signal",
                    description=f"**{signal.get('recommendation', 'HOLD')}**",
                    color=color
                )
                
                # Price Information
                embed.add_field(
                    name="💰 Price Information",
                    value=(
                        f"Current: `${signal.get('current_price', 0):.2f}`\n"
                        f"Predicted: `${signal.get('predicted_price', 0):.2f}`\n"
                        f"Change: `{signal.get('price_change_percent', 0):+.2f}%`"
                    ),
                    inline=False
                )
                
                # Trading Strategy
                embed.add_field(
                    name="📈 Trading Strategy",
                    value=(
                        f"Entry: `${signal.get('entry_point', 0):.2f}`\n"
                        f"🎯 High Target: `${signal.get('high_target', 0):.2f}`\n"
                        f"🛑 Low Target: `${signal.get('low_target', 0):.2f}`\n"
                        f"Stop Loss: `${signal.get('stop_loss', 0):.2f}`\n"
                        f"Take Profit: `${signal.get('take_profit', 0):.2f}`"
                    ),
                    inline=False
                )
                
                # Technical Analysis
                embed.add_field(
                    name="📊 Technical Analysis",
                    value=(
                        f"Trend: `{signal.get('trend', 'UNKNOWN')}`\n"
                        f"Support: `${signal.get('support', 0):.2f}`\n"
                        f"Resistance: `${signal.get('resistance', 0):.2f}`\n"
                        f"RSI: `{signal.get('rsi', 0):.2f}`\n"
                        f"ATR: `${signal.get('atr', 0):.4f}`"
                    ),
                    inline=False
                )
                
                # Confidence
                confidence = signal.get('confidence', 0)
                confidence_bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
                embed.add_field(
                    name="🎯 Confidence",
                    value=f"{confidence_bar} {confidence*100:.1f}%",
                    inline=False
                )
                
                embed.set_footer(
                    text=f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | Model V8"
                )
                
                await self.channel.send(embed=embed)
                await asyncio.sleep(1)  # Delay between messages
            
            logger.info(f"✓ Sent {len(predictions)} trading signals")
        
        except Exception as e:
            logger.error(f"✗ Failed to send trading signals: {e}")
            logger.error(traceback.format_exc())
    
    @commands.command(name='predict')
    async def predict_command(self, ctx, symbol: str = 'BTC'):
        """
        Manual prediction command
        Usage: !predict BTC
        """
        symbol = symbol.upper()
        
        async with ctx.typing():
            prediction = await self.model_manager.predict(symbol)
        
        if prediction:
            embed = discord.Embed(
                title=f"🎯 {symbol} Trading Signal",
                description=f"**{prediction.get('recommendation', 'HOLD')}**",
                color=discord.Color.green()
            )
            
            embed.add_field(name="Current Price", value=f"${prediction.get('current_price', 0):.2f}", inline=True)
            embed.add_field(name="Predicted Price", value=f"${prediction.get('predicted_price', 0):.2f}", inline=True)
            embed.add_field(name="Entry Point", value=f"${prediction.get('entry_point', 0):.2f}", inline=True)
            embed.add_field(name="Take Profit", value=f"${prediction.get('take_profit', 0):.2f}", inline=True)
            embed.add_field(name="Stop Loss", value=f"${prediction.get('stop_loss', 0):.2f}", inline=True)
            embed.add_field(name="Confidence", value=f"{prediction.get('confidence', 0)*100:.1f}%", inline=True)
            
            embed.set_footer(text="Model V8 | Crypto Price Predictor")
            
            await ctx.send(embed=embed)
        else:
            await ctx.send(f"❌ Failed to get prediction for {symbol}")
    
    @commands.command(name='status')
    async def status_command(self, ctx):
        """
        Check bot status and available symbols
        """
        status = "✅ Ready" if self.model_manager.ready else "❌ Not Ready"
        
        embed = discord.Embed(
            title="🤖 Bot Status",
            color=discord.Color.green() if self.model_manager.ready else discord.Color.red()
        )
        embed.add_field(name="Status", value=status, inline=False)
        embed.add_field(name="Model Manager", value="✅ Initialized" if self.model_manager.bot_predictor else "❌ Not initialized", inline=False)
        embed.add_field(name=f"Symbols ({len(self.config['crypto_symbols'])})", value=", ".join(sorted(self.config['crypto_symbols'])), inline=False)
        embed.add_field(name="Interval", value=f"{self.config['prediction_interval']}s ({self.config['prediction_interval']//3600}h)", inline=False)
        
        await ctx.send(embed=embed)
    
    @commands.command(name='models')
    async def models_command(self, ctx):
        """
        List all available models
        """
        models_dir = Path('models')
        
        if not models_dir.exists():
            await ctx.send("❌ Models directory not found")
            return
        
        model_files = sorted(list(models_dir.glob('*.pth')))
        
        if not model_files:
            await ctx.send("❌ No models found")
            return
        
        embed = discord.Embed(
            title="📦 Available Models",
            description=f"Total: {len(model_files)} models",
            color=discord.Color.blue()
        )
        
        # 分組顯示模型 (每個 embed field 最多 1024 字符)
        models_text = ""
        for i, model_file in enumerate(model_files, 1):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            line = f"{i}. `{model_file.name}` ({size_mb:.1f} MB)\n"
            
            if len(models_text) + len(line) > 1000:
                embed.add_field(name="\u200b", value=models_text, inline=False)
                models_text = line
            else:
                models_text += line
        
        if models_text:
            embed.add_field(name="\u200b", value=models_text, inline=False)
        
        await ctx.send(embed=embed)


async def main():
    """
    Main function to start the bot
    """
    try:
        # Load configuration
        logger.info("="*60)
        logger.info("🤖 Crypto Discord Bot v2.0 - Starting")
        logger.info("="*60)
        
        config = Config.load()
        
        # Create bot
        intents = discord.Intents.default()
        intents.message_content = True
        
        bot = commands.Bot(command_prefix='!', intents=intents)
        
        # Add cog
        await bot.add_cog(CryptoPredictorBot(bot, config))
        
        # Start bot
        logger.info(f"\n🚀 Connecting to Discord...")
        await bot.start(config['discord_token'])
    
    except Exception as e:
        logger.error(f"✗ Bot failed to start: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n😋 Bot stopped by user")
    except Exception as e:
        logger.error(f"✗ Fatal error: {e}")
        sys.exit(1)
