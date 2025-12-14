#!/usr/bin/env python3
"""
CryptoPredictor - Loads models from HuggingFace and provides predictions
Supports automatic dimension detection and adaptive model loading
"""

import os
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    import ccxt
    import ccxt.async_support as ccxt_async
except ImportError:
    print('Warning: ccxt not installed. Install with: pip install ccxt')

from huggingface_hub import hf_hub_download, list_repo_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
HF_REPO = "zongowo111/crypto_model"
MODEL_PATTERN = "_model_v8.pth"
DEVICE = torch.device('cpu')  # CPU only as specified
DEFAULT_LOOKBACK = 100  # Default lookback window

# Fallback exchanges when Binance is blocked
EXCHANGES = ['binance', 'bybit', 'okx', 'kraken']


class CryptoLSTMModel(torch.nn.Module):
    """
    Adaptive LSTM model that can handle variable input dimensions
    Automatically infers hidden size from model weight shapes
    """
    
    def __init__(
        self,
        input_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_features: int = 1,
        bidirectional: bool = False
    ):
        super().__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_features = output_features
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = torch.nn.LSTM(
            input_features,
            hidden_size,
            num_layers,
            bidirectional=bidirectional,
            dropout=0.2,
            batch_first=True
        )
        
        # Linear layers for output
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(lstm_output_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, output_features)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        predictions = self.regressor(last_out)
        return predictions


class CryptoPredictor:
    """
    Main predictor class that manages model loading and prediction
    """
    
    def __init__(self, cache_dir: str = './models/hf_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_info: Dict[str, Dict] = {}
        self.exchange_fallback: List[str] = EXCHANGES.copy()
        self.last_update = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f'CryptoPredictor initialized with cache: {self.cache_dir}')
    
    async def initialize(self):
        """
        Initialize predictor: download and load all models
        """
        try:
            logger.info(f'\ud83d\udd04 Downloading model list from {HF_REPO}...')
            
            # Get list of model files from HuggingFace
            model_files = await self._get_hf_model_files()
            
            if not model_files:
                logger.warning('⚠️  No model files found on HuggingFace')
                return
            
            logger.info(f'\u2713 Found {len(model_files)} model files')
            
            # Download and load each model
            for model_file in model_files:
                symbol = self._extract_symbol(model_file)
                if not symbol:
                    continue
                
                await self._load_model(symbol, model_file)
            
            logger.info(f'\u2713 Total loaded: {len(self.models)} models')
            
        except Exception as e:
            logger.error(f'Error initializing predictor: {e}')
            raise
    
    async def _get_hf_model_files(self) -> List[str]:
        """
        Get list of model files from HuggingFace repository
        """
        try:
            files = list_repo_files(repo_id=HF_REPO, repo_type='model')
            return [f for f in files if MODEL_PATTERN in f]
        except Exception as e:
            logger.error(f'Error fetching HuggingFace files: {e}')
            return []
    
    def _extract_symbol(self, filename: str) -> Optional[str]:
        """
        Extract cryptocurrency symbol from model filename
        Format: {SYMBOL}_model_v8.pth -> {SYMBOL}
        """
        try:
            if MODEL_PATTERN not in filename:
                return None
            
            symbol = filename.replace(MODEL_PATTERN, '').upper()
            return symbol if symbol and symbol.isalpha() else None
        except Exception as e:
            logger.error(f'Error extracting symbol from {filename}: {e}')
            return None
    
    async def _load_model(self, symbol: str, model_file: str) -> bool:
        """
        Download and load a model from HuggingFace
        Automatically detects model dimensions
        """
        try:
            logger.info(f'Loading {symbol} from {model_file}...')
            
            # Download model file
            local_path = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                hf_hub_download,
                HF_REPO,
                model_file,
                str(self.cache_dir)
            )
            
            # Load checkpoint
            checkpoint = torch.load(local_path, map_location=DEVICE)
            
            # Detect model architecture from checkpoint
            model_config = self._detect_model_config(checkpoint)
            
            # Create model with detected config
            model = CryptoLSTMModel(
                input_features=model_config['input_features'],
                hidden_size=model_config['hidden_size'],
                num_layers=model_config['num_layers'],
                output_features=model_config['output_features'],
                bidirectional=model_config['bidirectional']
            ).to(DEVICE)
            
            # Load weights
            model.load_state_dict(checkpoint)
            model.eval()
            
            # Store model
            self.models[symbol] = model
            self.model_info[symbol] = {
                'status': 'loaded',
                'input_features': model_config['input_features'],
                'hidden_size': model_config['hidden_size'],
                'num_layers': model_config['num_layers'],
                'output_features': model_config['output_features'],
                'bidirectional': model_config['bidirectional'],
                'file': model_file
            }
            
            logger.info(f'\u2713 {symbol} loaded successfully')
            logger.info(f'  Input: {model_config["input_features"]} | Hidden: {model_config["hidden_size"]} | Output: {model_config["output_features"]}')
            
            return True
            
        except Exception as e:
            logger.error(f'✗ Failed to load {symbol}: {str(e)}')
            self.model_info[symbol] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def _detect_model_config(self, checkpoint: Dict) -> Dict:
        """
        Automatically detect model configuration from checkpoint weights
        """
        config = {
            'input_features': 1,
            'hidden_size': 128,
            'num_layers': 2,
            'output_features': 1,
            'bidirectional': False
        }
        
        try:
            # Detect input features from LSTM weight_ih_l0
            # Shape: [4 * hidden_size, input_features]
            if 'lstm.weight_ih_l0' in checkpoint:
                weight_ih = checkpoint['lstm.weight_ih_l0']
                config['input_features'] = weight_ih.shape[1]
                config['hidden_size'] = weight_ih.shape[0] // 4
            
            # Detect num_layers
            num_layers = 1
            for key in checkpoint.keys():
                if 'lstm.weight_hh_l' in key:
                    layer_num = int(key.split('_l')[1])
                    num_layers = max(num_layers, layer_num + 1)
            config['num_layers'] = num_layers
            
            # Detect bidirectional
            for key in checkpoint.keys():
                if 'lstm.weight_ih_l0_reverse' in key:
                    config['bidirectional'] = True
                    break
            
            # Detect output features from last linear layer
            for key in ['regressor.6.weight', 'regressor.4.weight', 'regressor.2.weight']:
                if key in checkpoint:
                    config['output_features'] = checkpoint[key].shape[0]
                    break
            
            logger.info(f'  Detected config: {config}')
            
        except Exception as e:
            logger.warning(f'Error detecting model config, using defaults: {e}')
        
        return config
    
    async def predict_single(self, symbol: str) -> Optional[Dict]:
        """
        Make a prediction for a single cryptocurrency
        
        Returns:
            Dict with: trend, confidence_score, predicted_prices, entry_price, stop_loss, take_profit
        """
        try:
            if symbol not in self.models:
                logger.warning(f'Model for {symbol} not loaded')
                return None
            
            # Fetch real-time OHLCV data (1H)
            ohlcv_data = await self._fetch_ohlcv(symbol, '1h', 100)
            if ohlcv_data is None or len(ohlcv_data) < 10:
                logger.warning(f'Insufficient data for {symbol}')
                return None
            
            # Prepare features
            features = self._prepare_features(ohlcv_data)
            if features is None:
                return None
            
            # Get current price
            current_price = ohlcv_data[-1]['close']
            
            # Make prediction
            model = self.models[symbol]
            with torch.no_grad():
                X = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)  # Add batch dimension
                prediction = model(X).cpu().numpy()[0][0]
            
            # Predict next 3-5 candles
            predicted_prices = self._generate_predictions(current_price, prediction, count=5)
            
            # Generate trading signals
            trend, confidence = self._analyze_trend(ohlcv_data, predicted_prices)
            entry_price, stop_loss, take_profit = self._calculate_entry_points(
                current_price,
                predicted_prices,
                trend
            )
            
            # Calculate support/resistance
            support, resistance = self._calculate_support_resistance(ohlcv_data)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'trend': trend,
                'confidence_score': confidence,
                'predicted_prices': predicted_prices,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'support': support,
                'resistance': resistance,
                'signal_type': 'LONG' if 'UP' in trend else 'SHORT' if 'DOWN' in trend else 'NEUTRAL',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f'Error predicting {symbol}: {e}')
            return None
    
    async def _fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100
    ) -> Optional[List[Dict]]:
        """
        Fetch OHLCV data from multiple exchanges with fallback
        """
        pair = f'{symbol}/USDT'
        
        for exchange_name in self.exchange_fallback:
            try:
                logger.info(f'Fetching {pair} from {exchange_name}...')
                
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class()
                
                # Fetch OHLCV data
                ohlcv = await exchange.fetch_ohlcv(pair, timeframe, limit=limit)
                await exchange.close()
                
                # Convert to dictionary format
                result = []
                for candle in ohlcv:
                    result.append({
                        'timestamp': candle[0],
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })
                
                logger.info(f'✓ Fetched {len(result)} candles for {pair} from {exchange_name}')
                return result
                
            except Exception as e:
                logger.warning(f'⚠️  {exchange_name} failed for {pair}: {str(e)[:100]}')
                continue
        
        logger.error(f'✗ Failed to fetch {pair} from all exchanges')
        return None
    
    def _prepare_features(self, ohlcv_data: List[Dict], lookback: int = 100) -> Optional[np.ndarray]:
        """
        Prepare input features for model
        Returns normalized OHLCV data with optional technical indicators
        """
        try:
            data = np.array([
                [
                    c['open'],
                    c['high'],
                    c['low'],
                    c['close'],
                    c['volume']
                ]
                for c in ohlcv_data[-lookback:]
            ])
            
            # Normalize using min-max scaling
            for i in range(data.shape[1]):
                min_val = np.min(data[:, i])
                max_val = np.max(data[:, i])
                if max_val > min_val:
                    data[:, i] = (data[:, i] - min_val) / (max_val - min_val)
            
            return data
            
        except Exception as e:
            logger.error(f'Error preparing features: {e}')
            return None
    
    def _generate_predictions(
        self,
        current_price: float,
        model_output: float,
        count: int = 5
    ) -> List[float]:
        """
        Generate predicted prices for next N candles
        """
        predictions = []
        price = current_price
        
        # Model output is typically normalized price change
        for i in range(count):
            change = model_output * (i + 1) * 0.01  # Exponential influence
            new_price = price * (1 + change)
            predictions.append(new_price)
            price = new_price
        
        return predictions
    
    def _analyze_trend(self, ohlcv_data: List[Dict], predicted_prices: List[float]) -> Tuple[str, float]:
        """
        Analyze trend based on historical and predicted data
        Returns (trend_string, confidence_score)
        """
        # Historical trend
        close_prices = [c['close'] for c in ohlcv_data[-20:]]
        current = close_prices[-1]
        
        # Calculate SMA20
        sma20 = np.mean(close_prices)
        
        # Predicted trend
        avg_predicted = np.mean(predicted_prices)
        
        # Calculate trend confidence
        historical_direction = 1 if current > sma20 else -1
        predicted_direction = 1 if avg_predicted > current else -1
        
        # Confidence increases if both agree
        confidence = 0.7 if historical_direction == predicted_direction else 0.5
        
        # Calculate RSI-like momentum
        gains = sum(1 for i in range(1, len(close_prices)) if close_prices[i] > close_prices[i-1])
        momentum = gains / len(close_prices)
        confidence += momentum * 0.3
        confidence = min(0.99, confidence)
        
        if predicted_direction > 0 and momentum > 0.5:
            trend = 'UPTREND'
        elif predicted_direction < 0 and momentum < 0.5:
            trend = 'DOWNTREND'
        else:
            trend = 'NEUTRAL'
        
        return trend, confidence
    
    def _calculate_entry_points(
        self,
        current_price: float,
        predicted_prices: List[float],
        trend: str
    ) -> Tuple[float, float, float]:
        """
        Calculate entry price, stop loss, and take profit levels
        """
        if trend == 'NEUTRAL':
            return current_price, current_price * 0.98, current_price * 1.02
        
        # Find high and low in predicted prices
        high = max(predicted_prices)
        low = min(predicted_prices)
        
        if trend == 'UPTREND':
            # Enter at the lowest point in predictions
            entry = low
            stop_loss = entry * 0.97  # 3% stop
            take_profit = entry * 1.05  # 5% target
        else:  # DOWNTREND
            # Enter at the highest point in predictions
            entry = high
            stop_loss = entry * 1.03  # 3% stop
            take_profit = entry * 0.95  # 5% target
        
        return entry, stop_loss, take_profit
    
    def _calculate_support_resistance(self, ohlcv_data: List[Dict], lookback: int = 50) -> Tuple[float, float]:
        """
        Calculate support and resistance levels
        """
        data = ohlcv_data[-lookback:]
        lows = [c['low'] for c in data]
        highs = [c['high'] for c in data]
        
        # Simple support/resistance: lowest low and highest high
        support = np.min(lows)
        resistance = np.max(highs)
        
        return support, resistance


if __name__ == '__main__':
    # Test the predictor
    async def test():
        predictor = CryptoPredictor()
        await predictor.initialize()
        
        if predictor.models:
            symbol = list(predictor.models.keys())[0]
            result = await predictor.predict_single(symbol)
            if result:
                print(f'\n{symbol} Prediction:')
                print(f'  Current: ${result["current_price"]:.2f}')
                print(f'  Trend: {result["trend"]}')
                print(f'  Confidence: {result["confidence_score"]:.2%}')
    
    import asyncio
    asyncio.run(test())
