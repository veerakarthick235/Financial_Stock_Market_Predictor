from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_market, get_historical_data, StockPredictor
from config import Config
import os
import traceback

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'name': 'AI Financial Market Prediction API',
        'version': '2.0',
        'status': 'active',
        'demo_mode': Config.DEMO_MODE,
        'endpoints': {
            '/': 'GET - API information',
            '/predict': 'POST - Predict stock prices',
            '/historical': 'POST - Get historical data',
            '/train': 'POST - Train new model',
            '/models': 'GET - List available models',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'demo_mode': Config.DEMO_MODE,
        'models_dir': Config.MODELS_DIR,
        'data_dir': Config.DATA_DIR
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict stock price endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'symbol' not in data:
            return jsonify({
                'success': False,
                'error': 'Symbol is required'
            }), 400
        
        symbol = data['symbol'].strip().upper()
        retrain = data.get('retrain', False)
        
        print(f"\n{'='*60}")
        print(f"API Request: /predict")
        print(f"Symbol: {symbol}")
        print(f"Retrain: {retrain}")
        print(f"{'='*60}\n")
        
        result = predict_market(symbol, retrain=retrain)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        error_msg = str(e)
        print(f"Error in /predict: {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/historical', methods=['POST'])
def historical():
    """Get historical stock data endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'symbol' not in data:
            return jsonify({
                'success': False,
                'error': 'Symbol is required'
            }), 400
        
        symbol = data['symbol'].strip().upper()
        period = data.get('period', '1y')
        
        print(f"\n{'='*60}")
        print(f"API Request: /historical")
        print(f"Symbol: {symbol}, Period: {period}")
        print(f"{'='*60}\n")
        
        result = get_historical_data(symbol, period)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        error_msg = str(e)
        print(f"Error in /historical: {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/train', methods=['POST'])
def train():
    """Train a new model endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'symbol' not in data:
            return jsonify({
                'success': False,
                'error': 'Symbol is required'
            }), 400
        
        symbol = data['symbol'].strip().upper()
        
        print(f"\n{'='*60}")
        print(f"API Request: /train")
        print(f"Symbol: {symbol}")
        print(f"{'='*60}\n")
        
        predictor = StockPredictor(symbol)
        stock_data = predictor.get_data()
        metrics = predictor.train_model(stock_data)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'message': 'Model trained successfully',
            'metrics': metrics
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error in /train: {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List all available trained models"""
    try:
        models = []
        
        if os.path.exists(Config.MODELS_DIR):
            for file in os.listdir(Config.MODELS_DIR):
                if file.endswith('_model.h5'):
                    symbol = file.replace('_model.h5', '')
                    
                    # Get model file size
                    file_path = os.path.join(Config.MODELS_DIR, file)
                    file_size = os.path.getsize(file_path)
                    
                    models.append({
                        'symbol': symbol,
                        'size': f"{file_size / (1024*1024):.2f} MB"
                    })
        
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error in /models: {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting AI Stock Prediction Server")
    print("="*60)
    print(f"Models directory: {Config.MODELS_DIR}")
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Demo mode: {'ENABLED ⚠' if Config.DEMO_MODE else 'DISABLED'}")
    print(f"Server: http://{Config.HOST}:{Config.PORT}")
    print("="*60)
    print("\nPress CTRL+C to quit\n")
    
    app.run(
        debug=Config.DEBUG,
        host=Config.HOST,
        port=Config.PORT,
        use_reloader=False
    )