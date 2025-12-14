#!/usr/bin/env python3
"""
Web Dashboard for Cryptocurrency Predictions
Flask-based visualization of all predictions
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import logging

load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global prediction cache (will be updated by bot)
prediction_cache = {}


@app.route('/')
def index():
    """
    Main dashboard page
    """
    return render_template('dashboard.html')


@app.route('/api/predictions')
def get_predictions():
    """
    API endpoint: Get all predictions
    """
    return jsonify({
        'timestamp': datetime.utcnow().isoformat(),
        'predictions': prediction_cache,
        'total_symbols': len(prediction_cache)
    })


@app.route('/api/predictions/<symbol>')
def get_prediction(symbol: str):
    """
    API endpoint: Get specific symbol prediction
    """
    symbol = symbol.upper()
    
    if symbol not in prediction_cache:
        return jsonify({'error': f'No prediction for {symbol}'}), 404
    
    return jsonify(prediction_cache[symbol])


@app.route('/api/signals')
def get_signals():
    """
    API endpoint: Get all trading signals sorted by confidence
    """
    signals = []
    for symbol, pred in prediction_cache.items():
        signals.append({
            'symbol': symbol,
            'signal_type': pred.get('signal_type', 'NEUTRAL'),
            'entry_price': pred.get('entry_price'),
            'stop_loss': pred.get('stop_loss'),
            'take_profit': pred.get('take_profit'),
            'confidence': pred.get('confidence_score', 0),
            'current_price': pred.get('current_price'),
            'trend': pred.get('trend')
        })
    
    # Sort by confidence
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    return jsonify({
        'timestamp': datetime.utcnow().isoformat(),
        'signals': signals,
        'long_signals': [s for s in signals if s['signal_type'] == 'LONG'],
        'short_signals': [s for s in signals if s['signal_type'] == 'SHORT']
    })


@app.route('/api/statistics')
def get_statistics():
    """
    API endpoint: Get overall statistics
    """
    if not prediction_cache:
        return jsonify({
            'total_symbols': 0,
            'long_signals': 0,
            'short_signals': 0,
            'neutral_signals': 0,
            'avg_confidence': 0
        })
    
    long_count = sum(1 for p in prediction_cache.values() if p.get('signal_type') == 'LONG')
    short_count = sum(1 for p in prediction_cache.values() if p.get('signal_type') == 'SHORT')
    neutral_count = len(prediction_cache) - long_count - short_count
    avg_confidence = sum(p.get('confidence_score', 0) for p in prediction_cache.values()) / len(prediction_cache) if prediction_cache else 0
    
    return jsonify({
        'total_symbols': len(prediction_cache),
        'long_signals': long_count,
        'short_signals': short_count,
        'neutral_signals': neutral_count,
        'avg_confidence': round(avg_confidence, 4)
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500


if __name__ == '__main__':
    port = int(os.getenv('DASHBOARD_PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    
    logger.info(f'Starting dashboard on port {port}')
    app.run(host='0.0.0.0', port=port, debug=debug)
