#!/usr/bin/env python3
"""
Crypto Discord Bot - Real-time Price Prediction
Automatically loads models from HuggingFace and provides trading signals
"""

import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
import discord
from discord.ext import commands, tasks

from bot_predictor import CryptoPredictor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Discord Bot
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.guild_messages = True

bot = commands.Bot(command_prefix='.', intents=intents)

# Global predictor instance
predictor: CryptoPredictor = None

# Store prediction results for dashboard
prediction_cache = {}


@bot.event
async def on_ready():
    """Bot is ready to receive commands"""
    logger.info(f'✓ Bot logged in as {bot.user}')
    logger.info(f'✓ Bot ID: {bot.user.id}')
    
    # Start prediction loop
    if not prediction_loop.is_running():
        prediction_loop.start()
        logger.info('✓ Prediction loop started')


@bot.command(name='models')
async def cmd_list_models(ctx):
    """
    列出所有加載的模型及其詳細信息
    Usage: .models
    """
    if not predictor or not predictor.models:
        await ctx.send('❌ No models loaded')
        return
    
    embed = discord.Embed(
        title='📊 Loaded Cryptocurrency Models',
        description=f'Total: {len(predictor.models)} models',
        color=discord.Color.blue(),
        timestamp=datetime.utcnow()
    )
    
    # Group models by status
    loaded = []
    failed = []
    
    for symbol, info in predictor.model_info.items():
        if info['status'] == 'loaded':
            loaded.append(f"✓ **{symbol}**\n  Input: {info['input_features']} | Hidden: {info['hidden_size']} | Output: {info['output_features']}")
        else:
            failed.append(f"✗ {symbol} - {info.get('error', 'Unknown error')}")
    
    if loaded:
        embed.add_field(
            name='✅ Successfully Loaded',
            value='\n'.join(loaded[:10]),  # Max 10 per field
            inline=False
        )
        if len(loaded) > 10:
            embed.add_field(
                name='... and more',
                value=f'Total loaded: {len(loaded)}',
                inline=False
            )
    
    if failed:
        embed.add_field(
            name='❌ Failed to Load',
            value='\n'.join(failed[:5]),
            inline=False
        )
    
    await ctx.send(embed=embed)


@bot.command(name='predict')
async def cmd_predict(ctx, symbol: str = None):
    """
    獲得特定幣種的實時預測
    Usage: .predict BTC  或  .predict (預測所有)
    """
    if not predictor:
        await ctx.send('❌ Predictor not initialized')
        return
    
    if symbol:
        symbol = symbol.upper()
        if symbol not in prediction_cache:
            await ctx.send(f'❌ No prediction for {symbol}. Try `.models` to see available')
            return
        
        result = prediction_cache[symbol]
        embed = _create_prediction_embed(symbol, result)
        await ctx.send(embed=embed)
    else:
        # Show all predictions
        if not prediction_cache:
            await ctx.send('⏳ No predictions available yet. Please wait for the first cycle...')
            return
        
        # Send up to 5 embeds (Discord rate limiting)
        symbols = list(prediction_cache.keys())[:5]
        for sym in symbols:
            result = prediction_cache[sym]
            embed = _create_prediction_embed(sym, result)
            await ctx.send(embed=embed)
            await asyncio.sleep(0.5)  # Rate limiting


@bot.command(name='signal')
async def cmd_signal(ctx, symbol: str = None):
    """
    獲得交易信號 (LONG/SHORT + 入場點)
    Usage: .signal BTC  或  .signal (所有信號)
    """
    if not predictor:
        await ctx.send('❌ Predictor not initialized')
        return
    
    if symbol:
        symbol = symbol.upper()
        if symbol not in prediction_cache:
            await ctx.send(f'❌ No signal for {symbol}')
            return
        
        result = prediction_cache[symbol]
        embed = _create_signal_embed(symbol, result)
        await ctx.send(embed=embed)
    else:
        # Show signals with highest confidence
        if not prediction_cache:
            await ctx.send('⏳ No signals available yet...')
            return
        
        # Sort by confidence and send top 5
        sorted_signals = sorted(
            prediction_cache.items(),
            key=lambda x: x[1].get('confidence_score', 0),
            reverse=True
        )[:5]
        
        for sym, result in sorted_signals:
            embed = _create_signal_embed(sym, result)
            await ctx.send(embed=embed)
            await asyncio.sleep(0.5)


@bot.command(name='stats')
async def cmd_stats(ctx):
    """
    顯示機器人統計信息
    Usage: .stats
    """
    if not predictor:
        await ctx.send('❌ Predictor not initialized')
        return
    
    embed = discord.Embed(
        title='📈 Bot Statistics',
        color=discord.Color.green(),
        timestamp=datetime.utcnow()
    )
    
    total_models = len(predictor.model_info)
    loaded_count = sum(1 for info in predictor.model_info.values() if info['status'] == 'loaded')
    failed_count = total_models - loaded_count
    
    embed.add_field(
        name='Models',
        value=f'Loaded: {loaded_count}/{total_models}\nFailed: {failed_count}',
        inline=True
    )
    
    embed.add_field(
        name='Predictions',
        value=f'Cached: {len(prediction_cache)}\nLast update: {predictor.last_update or "Never"}',
        inline=True
    )
    
    embed.add_field(
        name='API Status',
        value=f'Exchange: {predictor.exchange_fallback[0] if predictor.exchange_fallback else "Checking..."}',
        inline=False
    )
    
    await ctx.send(embed=embed)


@bot.command(name='dashboard')
async def cmd_dashboard(ctx):
    """
    顯示網頁儀表板的 URL
    Usage: .dashboard
    """
    dashboard_url = os.getenv('DASHBOARD_URL', 'http://localhost:5000')
    
    embed = discord.Embed(
        title='📊 Prediction Dashboard',
        description=f'[Open Dashboard]({dashboard_url})',
        color=discord.Color.purple(),
        timestamp=datetime.utcnow()
    )
    
    embed.add_field(
        name='Features',
        value='✓ Real-time predictions\n✓ All cryptocurrencies\n✓ Trading signals\n✓ Technical analysis',
        inline=False
    )
    
    await ctx.send(embed=embed)


@bot.command(name='reload')
async def cmd_reload(ctx):
    """
    重新加載所有模型
    Usage: .reload
    """
    async with ctx.typing():
        try:
            global predictor
            logger.info('Reloading models...')
            predictor = CryptoPredictor()
            await predictor.initialize()
            
            embed = discord.Embed(
                title='✓ Models Reloaded',
                description=f'Successfully loaded {len(predictor.models)} models',
                color=discord.Color.green()
            )
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f'Error reloading models: {e}')
            await ctx.send(f'❌ Error reloading: {str(e)}')


@bot.command(name='test')
async def cmd_test(ctx):
    """
    測試單個幣種的完整流程
    Usage: .test BTC
    """
    if not predictor:
        await ctx.send('❌ Predictor not initialized')
        return
    
    async with ctx.typing():
        try:
            # Test with BTC or first available model
            test_symbol = 'BTC'
            if test_symbol not in predictor.models:
                test_symbol = list(predictor.models.keys())[0] if predictor.models else None
            
            if not test_symbol:
                await ctx.send('❌ No models available for testing')
                return
            
            result = await predictor.predict_single(test_symbol)
            
            embed = discord.Embed(
                title=f'✓ Test Prediction: {test_symbol}',
                color=discord.Color.green()
            )
            
            if result:
                embed.add_field('Current Price', f"${result.get('current_price', 'N/A')}", inline=True)
                embed.add_field('Trend', result.get('trend', 'N/A'), inline=True)
                embed.add_field('Confidence', f"{result.get('confidence_score', 0):.2%}", inline=True)
            else:
                embed.add_field('Status', 'Prediction failed', inline=False)
            
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f'Test command error: {e}')
            await ctx.send(f'❌ Error: {str(e)}')


@tasks.loop(minutes=60)  # Run every hour (new 1H candle)
async def prediction_loop():
    """
    自動預測循環 - 每當新的 1H K線出現時執行
    """
    if not predictor:
        return
    
    try:
        logger.info('\n' + '='*80)
        logger.info(f'🔄 Starting prediction cycle - {datetime.utcnow().isoformat()}')
        logger.info('='*80)
        
        # Predict all loaded models
        for symbol in list(predictor.models.keys()):
            try:
                result = await predictor.predict_single(symbol)
                if result:
                    prediction_cache[symbol] = result
                    logger.info(f'✓ {symbol}: {result.get("trend", "N/A")} | Confidence: {result.get("confidence_score", 0):.2%}')
                else:
                    logger.warning(f'⚠️  {symbol}: No prediction result')
            except Exception as e:
                logger.error(f'✗ {symbol}: {str(e)}')
        
        # Update prediction update time
        predictor.last_update = datetime.utcnow().isoformat()
        
        # Optionally send summary to a specific channel
        # await _send_summary_to_channel(bot, prediction_cache)
        
    except Exception as e:
        logger.error(f'Prediction loop error: {e}')


def _create_prediction_embed(symbol: str, result: dict) -> discord.Embed:
    """
    Create Discord embed for prediction result
    """
    embed = discord.Embed(
        title=f'📊 {symbol}/USDT Prediction',
        color=discord.Color.blue(),
        timestamp=datetime.utcnow()
    )
    
    # Price and trend
    current_price = result.get('current_price', 'N/A')
    trend = result.get('trend', 'UNKNOWN')
    
    trend_emoji = '📈' if 'UP' in trend else '📉' if 'DOWN' in trend else '➡️'
    
    embed.add_field(
        name='Current Price',
        value=f'${current_price}',
        inline=True
    )
    
    embed.add_field(
        name='3-5 Candle Trend',
        value=f'{trend_emoji} {trend}',
        inline=True
    )
    
    # Predicted prices
    pred_prices = result.get('predicted_prices', [])
    if pred_prices:
        embed.add_field(
            name='Predicted Prices (Next 5H)',
            value='\n'.join([f'H+{i+1}: ${p:.2f}' for i, p in enumerate(pred_prices[:5])]),
            inline=False
        )
    
    # Technical indicators
    embed.add_field(
        name='Support/Resistance',
        value=f'Support: ${result.get("support", "N/A")}\nResistance: ${result.get("resistance", "N/A")}',
        inline=True
    )
    
    embed.add_field(
        name='Confidence',
        value=f"{result.get('confidence_score', 0):.2%}",
        inline=True
    )
    
    return embed


def _create_signal_embed(symbol: str, result: dict) -> discord.Embed:
    """
    Create Discord embed for trading signal
    """
    signal_type = result.get('signal_type', 'NEUTRAL')
    entry_price = result.get('entry_price')
    stop_loss = result.get('stop_loss')
    take_profit = result.get('take_profit')
    
    color = discord.Color.green() if 'LONG' in signal_type else discord.Color.red() if 'SHORT' in signal_type else discord.Color.yellow()
    
    embed = discord.Embed(
        title=f'🎯 Trading Signal: {symbol}',
        color=color,
        timestamp=datetime.utcnow()
    )
    
    embed.add_field(
        name='Signal Type',
        value=signal_type,
        inline=True
    )
    
    embed.add_field(
        name='Confidence',
        value=f"{result.get('confidence_score', 0):.2%}",
        inline=True
    )
    
    if entry_price:
        embed.add_field(
            name='Entry Price',
            value=f'${entry_price:.2f}',
            inline=True
        )
    
    if stop_loss:
        embed.add_field(
            name='Stop Loss',
            value=f'${stop_loss:.2f}',
            inline=True
        )
    
    if take_profit:
        embed.add_field(
            name='Take Profit',
            value=f'${take_profit:.2f}',
            inline=True
        )
    
    trend = result.get('trend', '')
    if trend:
        embed.add_field(
            name='Market Trend',
            value=trend,
            inline=False
        )
    
    return embed


async def main():
    """
    主程序入口
    """
    global predictor
    
    try:
        # Initialize predictor
        logger.info('Initializing CryptoPredictor...')
        predictor = CryptoPredictor()
        await predictor.initialize()
        logger.info(f'✓ Predictor initialized with {len(predictor.models)} models')
        
        # Start Discord bot
        token = os.getenv('DISCORD_TOKEN')
        if not token:
            raise ValueError('DISCORD_TOKEN not found in .env')
        
        logger.info('Starting Discord bot...')
        await bot.start(token)
        
    except Exception as e:
        logger.error(f'Fatal error: {e}')
        raise


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('\nBot stopped by user')
    except Exception as e:
        logger.error(f'Bot crashed: {e}')
