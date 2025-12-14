#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Bot Predictor with Real-Time Data Fetching & Trading Signal Generation

Features:
1. Real-time data fetching from multiple exchanges (Binance, Bybit, OKX)
2. Fallback exchanges for geo-restrictions
3. Multi-timeframe technical analysis
4. Support/Resistance identification
5. Trading signal generation with entry/exit points
6. Risk management (stop-loss, take-profit)
7. Confidence scoring
"""

import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import warnings
import ccxt
from datetime import datetime, timedelta
import asyncio
import re

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class CryptoLSTMModel(torch.nn.Module):
    """LSTM model with variable hidden_size support"""
    
    def __init__(self, input_size=44, hidden_size=32, num_layers=2, output_size=1):
        super(CryptoLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # Bidirectional LSTM
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Bidirectional LSTM output size = hidden_size * 2
        lstm_output_size = hidden_size * 2
        
        # Regressor with flexible input size
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(lstm_output_size, hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size) or (batch, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, input_size) -> (batch, 1, input_size)
        
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Take last time step
        output = self.regressor(last_output)
        
        return output


class DataFetcher:
    """Real-time data fetching from multiple exchanges with fallback"""
    
    def __init__(self):
        # Initialize multiple exchanges
        self.exchanges = self._init_exchanges()
        self.cache = {}  # Cache to avoid excessive API calls
        self.cache_time = {}  # Cache timestamp
        self.cache_duration = 60  # Cache for 60 seconds
    
    def _init_exchanges(self) -> Dict:
        """
        Initialize multiple exchanges with fallback support
        """
        exchanges = {}
        
        # Primary: Binance
        try:
            exchanges['binance'] = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 1000,
            })
            logger.info("✓ Binance initialized")
        except Exception as e:
            logger.warning(f"⚠️  Binance initialization failed: {e}")
        
        # Fallback 1: Bybit (no geo-restriction)
        try:
            exchanges['bybit'] = ccxt.bybit({
                'enableRateLimit': True,
                'rateLimit': 1000,
            })
            logger.info("✓ Bybit initialized")
        except Exception as e:
            logger.warning(f"⚠️  Bybit initialization failed: {e}")
        
        # Fallback 2: OKX (no geo-restriction)
        try:
            exchanges['okx'] = ccxt.okx({
                'enableRateLimit': True,
                'rateLimit': 1000,
            })
            logger.info("✓ OKX initialized")
        except Exception as e:
            logger.warning(f"⚠️  OKX initialization failed: {e}")
        
        # Fallback 3: Kraken
        try:
            exchanges['kraken'] = ccxt.kraken({
                'enableRateLimit': True,
                'rateLimit': 3000,
            })
            logger.info("✓ Kraken initialized")
        except Exception as e:
            logger.warning(f"⚠️  Kraken initialization failed: {e}")
        
        return exchanges
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from multiple exchanges with fallback
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (1m, 5m, 1h, 4h, 1d)
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.cache:
            time_diff = datetime.now() - self.cache_time[cache_key]
            if time_diff.total_seconds() < self.cache_duration:
                logger.debug(f"Using cached data for {symbol}")
                return self.cache[cache_key]
        
        # Try exchanges in order
        exchange_order = ['binance', 'bybit', 'okx', 'kraken']
        
        for exchange_name in exchange_order:
            if exchange_name not in self.exchanges:
                continue
            
            try:
                exchange = self.exchanges[exchange_name]
                logger.info(f"Fetching {symbol} from {exchange_name}...")
                
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df.astype(float)
                
                # Cache the data
                self.cache[cache_key] = df
                self.cache_time[cache_key] = datetime.now()
                
                logger.info(f"✓ Fetched {len(df)} candles for {symbol} from {exchange_name}")
                return df
            
            except Exception as e:
                logger.warning(f"⚠️  {exchange_name} failed for {symbol}: {str(e)[:100]}")
                continue
        
        logger.error(f"✗ Failed to fetch {symbol} from all exchanges")
        return None
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        for exchange_name in ['binance', 'bybit', 'okx', 'kraken']:
            if exchange_name not in self.exchanges:
                continue
            
            try:
                exchange = self.exchanges[exchange_name]
                ticker = exchange.fetch_ticker(symbol)
                return float(ticker['last'])
            except:
                continue
        
        return None


class TechnicalAnalyzer:
    """Technical analysis for trading signals"""
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """
        計算支撐位和阻力位
        """
        try:
            recent = df.tail(window)
            support = recent['low'].min()
            resistance = recent['high'].max()
            return float(support), float(resistance)
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return 0, float('inf')
    
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> Dict:
        """計算移動平均線"""
        try:
            return {
                'sma20': float(df['close'].rolling(20).mean().iloc[-1]),
                'sma50': float(df['close'].rolling(50).mean().iloc[-1]),
                'ema12': float(df['close'].ewm(span=12).mean().iloc[-1]),
                'ema26': float(df['close'].ewm(span=26).mean().iloc[-1]),
            }
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {}
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
        """計算RSI"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> Tuple[float, float, float]:
        """計算MACD"""
        try:
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            return float(macd.iloc[-1]), float(signal.iloc[-1]), float(histogram.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0, 0, 0
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """計算ATR"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            return float(atr.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0
    
    @staticmethod
    def identify_trend(df: pd.DataFrame) -> str:
        """識別趨勢方向"""
        try:
            sma20 = df['close'].rolling(20).mean()
            sma50 = df['close'].rolling(50).mean()
            current_price = df['close'].iloc[-1]
            
            if current_price > sma20.iloc[-1] > sma50.iloc[-1]:
                return 'UPTREND'
            elif current_price < sma20.iloc[-1] < sma50.iloc[-1]:
                return 'DOWNTREND'
            else:
                return 'SIDEWAYS'
        except Exception as e:
            logger.error(f"Error identifying trend: {e}")
            return 'SIDEWAYS'


class BotPredictor:
    """
    Advanced Bot Predictor with Real-Time Data & Trading Signals
    """
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.models = {}  # Dictionary of loaded models
        self.data_fetcher = DataFetcher()
        self.analyzer = TechnicalAnalyzer()
        self.bias_corrections = {}  # Bias corrections for each symbol
        
        logger.info("🤖 BotPredictor initialized (CPU mode)")
        
        # Load all models
        self._load_all_models()
        
        # Load bias corrections
        self._load_bias_corrections()
    
    def _detect_model_architecture(self, checkpoint: Dict) -> Tuple[int, int]:
        """
        Detect the correct input_size and hidden_size from checkpoint weights.
        
        For bidirectional LSTM:
        lstm.weight_ih_l0 shape = [hidden_size * 4 * 2, input_size]
                               = [hidden_size * 8, input_size]
        
        Args:
            checkpoint: Model state dict
        
        Returns:
            Tuple of (input_size, hidden_size)
        """
        try:
            if 'lstm.weight_ih_l0' in checkpoint:
                weight_shape = checkpoint['lstm.weight_ih_l0'].shape
                first_dim = weight_shape[0]  # hidden_size * 8 for bidirectional
                input_size = weight_shape[1]  # input_size
                
                # For bidirectional LSTM: hidden_size = first_dim / 8
                # (4 gates per LSTM + bidirectional = 4 * 2 = 8)
                hidden_size = first_dim // 8
                
                logger.debug(f"  Detected: lstm.weight_ih_l0 shape {weight_shape} -> input_size={input_size}, hidden_size={hidden_size}")
                return input_size, hidden_size
            
            # Default fallback
            logger.warning("  Could not find lstm.weight_ih_l0, using defaults")
            return 44, 32
        except Exception as e:
            logger.warning(f"Could not detect architecture: {e}, using defaults")
            return 44, 32
    
    def _load_all_models(self):
        """Load all available models from models/"""
        try:
            models_dir = Path('models')
            if not models_dir.exists():
                logger.warning(f"Models directory not found: {models_dir}")
                return
            
            model_files = sorted(list(models_dir.glob('*.pth')))
            logger.info(f"Found {len(model_files)} model files")
            
            for model_file in model_files:
                try:
                    # Extract symbol from filename
                    stem = model_file.stem  # e.g., ADA_model_v8
                    match = re.match(r'^([A-Za-z]+)', stem)
                    if match:
                        symbol = match.group(1).upper()
                        logger.info(f"Loading {symbol} from {model_file.name}...")
                        
                        # Load checkpoint
                        checkpoint = torch.load(model_file, map_location=self.device)
                        
                        # Detect architecture from checkpoint
                        input_size, hidden_size = self._detect_model_architecture(checkpoint)
                        
                        # Create model with detected architecture
                        model = CryptoLSTMModel(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=2,
                            output_size=1
                        )
                        
                        try:
                            model.load_state_dict(checkpoint)
                            model.eval()
                            self.models[symbol] = model
                            logger.info(f"✓ {symbol} loaded (hidden_size={hidden_size})")
                        except RuntimeError as load_error:
                            logger.warning(f"⚠️  Failed to load {symbol}: {str(load_error)[:100]}")
                            logger.warning(f"   Skipping {symbol}")
                    else:
                        logger.warning(f"Could not extract symbol from {model_file.name}")
                
                except Exception as e:
                    logger.error(f"✗ Failed to load {model_file.name}: {str(e)[:150]}")
            
            logger.info(f"\n✓ Total loaded: {len(self.models)} models")
            if self.models:
                logger.info(f"  Available symbols: {', '.join(sorted(self.models.keys()))}")
        
        except Exception as e:
            logger.error(f"Failed to load models directory: {e}")
    
    def _load_bias_corrections(self):
        """Load bias corrections for each model"""
        try:
            bias_file = Path('bias_corrections_v8.json')
            if bias_file.exists():
                with open(bias_file, 'r') as f:
                    self.bias_corrections = json.load(f)
                logger.info(f"✓ Loaded bias corrections for {len(self.bias_corrections)} symbols")
            else:
                logger.debug(f"Bias corrections file not found: {bias_file}")
        
        except Exception as e:
            logger.error(f"Failed to load bias corrections: {e}")
    
    def _apply_bias_correction(self, symbol: str, prediction: float) -> float:
        """Apply bias correction to model prediction"""
        if symbol in self.bias_corrections:
            bias = self.bias_corrections[symbol].get('bias', 0)
            scale = self.bias_corrections[symbol].get('scale', 1.0)
            return prediction * scale + bias
        return prediction
    
    def _build_feature_vector(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build 44-dimensional feature vector from OHLCV data and technical indicators.
        
        Returns:
            numpy array of shape (44,)
        """
        try:
            features = []
            
            # 1. Basic OHLCV (5 features)
            current = df.iloc[-1]
            features.extend([
                float(current['open']),
                float(current['high']),
                float(current['low']),
                float(current['close']),
                float(current['volume'])
            ])
            
            # 2. Price changes and ratios (10 features)
            close_prices = df['close'].values
            for period in [1, 5, 10, 20, 50]:
                if len(df) >= period:
                    pct_change = (close_prices[-1] - close_prices[-period]) / close_prices[-period]
                    features.append(float(pct_change))
            
            # 3. Moving averages (12 features)
            try:
                features.append(float(df['close'].rolling(5).mean().iloc[-1]))
                features.append(float(df['close'].rolling(10).mean().iloc[-1]))
                features.append(float(df['close'].rolling(20).mean().iloc[-1]))
                features.append(float(df['close'].rolling(50).mean().iloc[-1]))
                features.append(float(df['close'].ewm(span=5).mean().iloc[-1]))
                features.append(float(df['close'].ewm(span=12).mean().iloc[-1]))
                features.append(float(df['close'].ewm(span=26).mean().iloc[-1]))
                
                # Volatility from rolling windows
                features.append(float(df['close'].rolling(20).std().iloc[-1]))
                features.append(float(df['close'].rolling(50).std().iloc[-1]))
                
                # High-Low range
                features.append(float((df['high'].rolling(20).max() - df['low'].rolling(20).min()).iloc[-1]))
                features.append(float(df['high'].iloc[-1] - df['low'].iloc[-1]))
                features.append(float((df['close'] - df['open']).abs().mean()))
            except:
                features.extend([0.0] * 12)
            
            # 4. Momentum indicators (12 features)
            try:
                # RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                features.append(float(rsi.iloc[-1]))
                
                # MACD
                exp1 = df['close'].ewm(span=12).mean()
                exp2 = df['close'].ewm(span=26).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9).mean()
                histogram = macd - signal
                features.append(float(macd.iloc[-1]))
                features.append(float(signal.iloc[-1]))
                features.append(float(histogram.iloc[-1]))
                
                # ATR
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                features.append(float(atr.iloc[-1]))
                
                # Bollinger Bands
                sma = df['close'].rolling(20).mean()
                std = df['close'].rolling(20).std()
                bb_upper = sma + (std * 2)
                bb_lower = sma - (std * 2)
                bb_position = (df['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                features.append(float(bb_position))
                
                # Stochastic
                lowest_low = df['low'].rolling(14).min()
                highest_high = df['high'].rolling(14).max()
                k_percent = 100 * ((df['close'].iloc[-1] - lowest_low.iloc[-1]) / (highest_high.iloc[-1] - lowest_low.iloc[-1]))
                features.append(float(k_percent))
                
                # Volume indicators
                volume_sma = df['volume'].rolling(20).mean()
                volume_ratio = df['volume'].iloc[-1] / (volume_sma.iloc[-1] + 1e-8)
                features.append(float(volume_ratio))
                
                # OBV
                obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
                features.append(float(obv.iloc[-1]))
            except:
                features.extend([0.0] * 12)
            
            # 5. Volatility measures (5 features)
            try:
                # Daily returns volatility
                returns = df['close'].pct_change()
                features.append(float(returns.std()))
                features.append(float(returns.mean()))
                
                # Range metrics
                features.append(float((df['high'] - df['low']).mean()))
                features.append(float((df['high'] - df['low']).std()))
                
                # Price position in range
                min_price = df['close'].rolling(50).min().iloc[-1]
                max_price = df['close'].rolling(50).max().iloc[-1]
                price_position = (df['close'].iloc[-1] - min_price) / (max_price - min_price)
                features.append(float(price_position))
            except:
                features.extend([0.0] * 5)
            
            # Ensure we have exactly 44 features
            features = features[:44]
            while len(features) < 44:
                features.append(0.0)
            
            return np.array(features, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error building feature vector: {e}")
            return np.zeros(44, dtype=np.float32)
    
    async def predict(self, symbol: str, timeframe: str = '1h') -> Optional[Dict]:
        """
        Generate trading signal for a symbol
        """
        try:
            # Check if model exists
            if symbol not in self.models:
                logger.warning(f"No model found for {symbol}")
                return None
            
            trading_pair = f"{symbol}/USDT"
            
            # 1️⃣ Fetch real-time data
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Analyzing {symbol}...")
            logger.info(f"{'='*60}")
            
            df = self.data_fetcher.fetch_ohlcv(trading_pair, timeframe, limit=100)
            if df is None or df.empty:
                logger.error(f"Failed to fetch data for {symbol}")
                return None
            
            # 2️⃣ Get current price
            current_price = df['close'].iloc[-1]
            logger.info(f"Current Price: ${current_price:.2f}")
            
            # 3️⃣ Technical Analysis
            logger.info(f"\n📊 Technical Analysis:")
            support, resistance = self.analyzer.calculate_support_resistance(df)
            logger.info(f"  Support: ${support:.2f}")
            logger.info(f"  Resistance: ${resistance:.2f}")
            
            trend = self.analyzer.identify_trend(df)
            logger.info(f"  Trend: {trend}")
            
            mas = self.analyzer.calculate_moving_averages(df)
            logger.info(f"  SMA20: ${mas.get('sma20', 0):.2f}")
            logger.info(f"  SMA50: ${mas.get('sma50', 0):.2f}")
            
            rsi = self.analyzer.calculate_rsi(df)
            macd, signal, histogram = self.analyzer.calculate_macd(df)
            atr = self.analyzer.calculate_atr(df)
            
            logger.info(f"\n📈 Indicators:")
            logger.info(f"  RSI(14): {rsi:.2f} {'(Overbought)' if rsi > 70 else '(Oversold)' if rsi < 30 else '(Neutral)'}")
            logger.info(f"  MACD: {macd:.6f}")
            logger.info(f"  ATR(14): ${atr:.2f}")
            
            # 4️⃣ Model Prediction
            logger.info(f"\n🤖 Model Prediction:")
            features = self._build_feature_vector(df)
            
            # Reshape to (1, 1, 44) for LSTM input
            X = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            
            model = self.models[symbol]
            with torch.no_grad():
                predicted_price = model(X).item()
            
            corrected_price = self._apply_bias_correction(symbol, predicted_price)
            logger.info(f"  Raw Prediction: ${predicted_price:.2f}")
            logger.info(f"  Corrected Prediction: ${corrected_price:.2f}")
            
            # 5️⃣ Calculate Price Movement
            price_change = ((corrected_price - current_price) / current_price) * 100
            direction = "📈 UP" if price_change > 0 else "📉 DOWN"
            logger.info(f"  Expected Change: {price_change:+.2f}% {direction}")
            
            # 6️⃣ Identify High/Low Points and Generate Signal
            logger.info(f"\n🏄 Trading Strategy:")
            
            if trend == 'UPTREND':
                if current_price < support:
                    entry_point = support
                    high_point = resistance
                    low_point = support * 0.98
                    signal_type = "BUY_STRONG"
                    recommendation = "STRONG BUY at support"
                elif current_price < mas['sma20']:
                    entry_point = current_price * 0.99
                    high_point = resistance
                    low_point = support
                    signal_type = "BUY"
                    recommendation = "BUY near SMA20"
                else:
                    entry_point = current_price
                    high_point = resistance
                    low_point = support
                    signal_type = "HOLD"
                    recommendation = "HOLD in uptrend"
            
            elif trend == 'DOWNTREND':
                if current_price > resistance:
                    entry_point = resistance
                    low_point = support
                    high_point = resistance * 1.02
                    signal_type = "SELL_STRONG"
                    recommendation = "STRONG SELL at resistance"
                elif current_price > mas['sma20']:
                    entry_point = current_price * 1.01
                    low_point = support
                    high_point = resistance
                    signal_type = "SELL"
                    recommendation = "SELL near SMA20"
                else:
                    entry_point = current_price
                    low_point = support
                    high_point = resistance
                    signal_type = "HOLD"
                    recommendation = "HOLD in downtrend"
            
            else:
                entry_point = (support + resistance) / 2
                high_point = resistance
                low_point = support
                signal_type = "RANGE_TRADE"
                recommendation = "RANGE TRADE between support and resistance"
            
            # 7️⃣ Calculate Confidence
            confidence_factors = []
            if trend == 'UPTREND' and price_change > 0:
                confidence_factors.append(0.8)
            elif trend == 'DOWNTREND' and price_change < 0:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            if rsi < 30 and trend == 'UPTREND':
                confidence_factors.append(0.8)
            elif rsi > 70 and trend == 'DOWNTREND':
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            if histogram > 0 and trend == 'UPTREND':
                confidence_factors.append(0.8)
            elif histogram < 0 and trend == 'DOWNTREND':
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            confidence = np.mean(confidence_factors)
            
            logger.info(f"  Entry: ${entry_point:.2f}")
            logger.info(f"  High Target: ${high_point:.2f}")
            logger.info(f"  Low Target: ${low_point:.2f}")
            logger.info(f"  Stop Loss: ${low_point * 0.98:.2f}")
            logger.info(f"  Take Profit: ${high_point * 1.02:.2f}")
            logger.info(f"  Confidence: {confidence*100:.1f}%")
            logger.info(f"  Recommendation: {recommendation}")
            
            # 8️⃣ Build result
            result = {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(corrected_price),
                'price_change_percent': float(price_change),
                'trend': trend,
                'signal_type': signal_type,
                'recommendation': recommendation,
                'entry_point': float(entry_point),
                'high_target': float(high_point),
                'low_target': float(low_point),
                'stop_loss': float(low_point * 0.98),
                'take_profit': float(high_point * 1.02),
                'support': float(support),
                'resistance': float(resistance),
                'rsi': float(rsi),
                'macd': float(macd),
                'atr': float(atr),
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat(),
            }
            
            logger.info(f"{'='*60}\n")
            return result
        
        except Exception as e:
            logger.error(f"✗ Prediction failed for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


# For backward compatibility
class Predictor(BotPredictor):
    """Legacy name for BotPredictor"""
    pass


if __name__ == '__main__':
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the predictor
    async def test():
        predictor = BotPredictor()
        test_symbols = ['BTC', 'ETH', 'SOL']
        
        for symbol in test_symbols:
            result = await predictor.predict(symbol, '1h')
            if result:
                print(f"\n✓ {symbol}: {result['recommendation']}")
    
    asyncio.run(test())
