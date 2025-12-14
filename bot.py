#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Discord Bot

Downloads models from HuggingFace and sends crypto price predictions to Discord
Automatically detects all available models in models/saved/ directory

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
        從 models/saved/ 目錄自動偵測所有可用模型
        提取模型檔名中的幣種 (例如: BTC_model_v8.pth -> BTC)
        """
        models_dir = Path('models/saved')
        symbols = set()
        
        if not models_dir.exists():
            logger.warning(f"⚠️  Models directory not found: {models_dir}")
            return []
        
        # 掃描所有 .pth 檔案
        model_files = list(models_dir.glob('*.pth'))
        
        if not model_files:
            logger.warning(f"⚠️  No model files found in {models_dir}")
            return []
        
        logger.info(f"Found {len(model_files)} model files:")
        
        for model_file in model_files:
            filename = model_file.stem  # 不含副檔名
            
            # 嘗試從檔名中提取幣種
            # 支援的格式: BTC_model_v8, BTC_model, btc_model_v8 等
            match = re.match(r'^([A-Za-z]+)(?:_model)?(?:_v\d+)?$', filename)
            
            if match:
                symbol = match.group(1).upper()
                symbols.add(symbol)
                logger.info(f"  ✓ {filename} -> {symbol}")
            else:
                # 如果不符合預期格式，嘗試只提取字母部分
                letters_only = re.match(r'^([A-Za-z]+)', filename)
                if letters_only:
                    symbol = letters_only.group(1).upper()
                    symbols.add(symbol)
                    logger.info(f"  ✓ {filename} -> {symbol} (extracted)")
                else:
                    logger.warning(f"  ⚠️  Could not extract symbol from {filename}")
        
        sorted_symbols = sorted(list(symbols))
        logger.info(f"✓ Detected {len(sorted_symbols)} unique symbols: {', '.join(sorted_symbols)}")
        
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
        hf_repo_id = os.getenv('HUGGINGFACE_REPO_ID', 'caizongxun/crypto-price-predictor-v8')
        
        # Bot config
        prediction_interval = int(os.getenv('PREDICTION_INTERVAL', '3600'))
        
        # 自動偵測模型幣種
        logger.info("\n🔍 Auto-detecting available models...")
        auto_detected_symbols = ModelDetector.detect_symbols_from_models()
        
        # 如果有手動配置的幣種，就使用手動配置；否則使用自動偵測
        manual_symbols = os.getenv('CRYPTO_SYMBOLS')
        if manual_symbols and manual_symbols != 'BTC,ETH,SOL,BNB,XRP':
            crypto_symbols = [s.strip().upper() for s in manual_symbols.split(',')]
            logger.info(f"✓ Using manually configured symbols: {', '.join(crypto_symbols)}")
        elif auto_detected_symbols:
            crypto_symbols = auto_detected_symbols
            logger.info(f"✓ Using auto-detected symbols: {', '.join(crypto_symbols)}")
        else:
            # 預設值 (如果沒有自動偵測到)
            crypto_symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
            logger.warning(f"⚠️  No models found, using default symbols: {', '.join(crypto_symbols)}")
        
        logger.info(f"\n✓ Configuration loaded successfully")
        logger.info(f"  Discord Channel: {channel_id}")
        logger.info(f"  HuggingFace Repo: {hf_repo_id}")
        logger.info(f"  Prediction Interval: {prediction_interval}s")
        logger.info(f"  Crypto Symbols ({len(crypto_symbols)}): {', '.join(crypto_symbols)}")
        
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
        Initialize model manager and download models
        """
        try:
            logger.info("Initializing model manager...")
            
            # Check if models are already downloaded
            models_dir = Path('models/saved')
            if models_dir.exists() and len(list(models_dir.glob('*.pth'))) > 0:
                logger.info(f"✓ Found {len(list(models_dir.glob('*.pth')))} models locally")
            else:
                logger.info("Downloading models from HuggingFace...")
                await self._download_models()
            
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
    
    async def _download_models(self):
        """
        Download all models from HuggingFace
        """
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading models from {self.hf_repo_id}...")
            
            snapshot_download(
                repo_id=self.hf_repo_id,
                repo_type="model",
                allow_patterns=["models/*.pth"],
                local_dir=".",
                token=self.hf_token
            )
            
            logger.info("✓ Models downloaded successfully")
        
        except Exception as e:
            logger.error(f"✗ Model download failed: {e}")
            raise
    
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
        Get prediction for a symbol
        """
        if not self.ready or not self.bot_predictor:
            logger.warning(f"Model manager not ready, skipping prediction for {symbol}")
            return None
        
        try:
            prediction = self.bot_predictor.predict(symbol)
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
        logger.info(f"Bot logged in as {self.bot.user}")
        
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
        """
        try:
            if not self.channel or not self.model_manager.ready:
                return
            
            logger.info(f"Starting prediction cycle for {len(self.config['crypto_symbols'])} symbols...")
            
            predictions = {}
            for symbol in self.config['crypto_symbols']:
                prediction = await self.model_manager.predict(symbol)
                if prediction:
                    predictions[symbol] = prediction
                await asyncio.sleep(1)  # Rate limiting
            
            if predictions:
                await self._send_predictions(predictions)
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
    
    async def _send_predictions(self, predictions):
        """
        Send prediction embed to Discord
        """
        try:
            embed = discord.Embed(
                title="📊 Crypto Price Predictions",
                description=f"Predictions at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                color=discord.Color.blue()
            )
            
            for symbol, prediction in predictions.items():
                if prediction:
                    current = prediction.get('current_price', 0)
                    predicted = prediction.get('corrected_price', 0)
                    direction = prediction.get('direction', '?')
                    confidence = prediction.get('confidence', 0)
                    
                    change = ((predicted - current) / current * 100) if current > 0 else 0
                    change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    
                    field_value = (
                        f"Current: `${current:.2f}`\n"
                        f"Predicted: `${predicted:.2f}`\n"
                        f"Change: `{change:+.2f}%` {change_emoji}\n"
                        f"Direction: `{direction}`\n"
                        f"Confidence: `{confidence*100:.1f}%`"
                    )
                    
                    embed.add_field(
                        name=f"{symbol}",
                        value=field_value,
                        inline=True
                    )
            
            embed.set_footer(text="Model V8 | Crypto Price Predictor")
            
            await self.channel.send(embed=embed)
            logger.info(f"✓ Sent predictions for {len(predictions)} symbols")
        
        except Exception as e:
            logger.error(f"✗ Failed to send predictions: {e}")
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
                title=f"📊 {symbol} Price Prediction",
                color=discord.Color.green()
            )
            
            current = prediction.get('current_price', 0)
            predicted = prediction.get('corrected_price', 0)
            direction = prediction.get('direction', '?')
            confidence = prediction.get('confidence', 0)
            
            change = ((predicted - current) / current * 100) if current > 0 else 0
            change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            
            embed.add_field(name="Current Price", value=f"${current:.2f}", inline=True)
            embed.add_field(name="Predicted Price", value=f"${predicted:.2f}", inline=True)
            embed.add_field(name="Change", value=f"{change:+.2f}% {change_emoji}", inline=True)
            embed.add_field(name="Direction", value=f"`{direction}`", inline=True)
            embed.add_field(name="Confidence", value=f"{confidence*100:.1f}%", inline=True)
            embed.add_field(name="Time", value=f"<t:{int(datetime.now().timestamp())}:R>", inline=True)
            
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
        embed.add_field(name=f"Symbols ({len(self.config['crypto_symbols'])})", value=", ".join(self.config['crypto_symbols']), inline=False)
        embed.add_field(name="Interval", value=f"{self.config['prediction_interval']}s ({self.config['prediction_interval']//3600}h)", inline=False)
        
        await ctx.send(embed=embed)
    
    @commands.command(name='models')
    async def models_command(self, ctx):
        """
        List all available models
        """
        models_dir = Path('models/saved')
        
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
        logger.info("🤖 Crypto Discord Bot - Starting")
        logger.info("="*60)
        
        config = Config.load()
        
        # Create bot
        intents = discord.Intents.default()
        intents.message_content = True
        
        bot = commands.Bot(command_prefix='!', intents=intents)
        
        # Add cog
        await bot.add_cog(CryptoPredictorBot(bot, config))
        
        # Start bot
        logger.info(f"Connecting to Discord...")
        await bot.start(config['discord_token'])
    
    except Exception as e:
        logger.error(f"✗ Bot failed to start: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"✗ Fatal error: {e}")
        sys.exit(1)
