import numpy as np
import pandas as pd
import os
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from config import Config
from utils import (calculate_technical_indicators, calculate_sentiment_score, 
                   get_stock_info, calculate_confidence_score, validate_symbol)
import warnings
import time
warnings.filterwarnings('ignore')


class StockPredictor:
    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.model_path = os.path.join(Config.MODELS_DIR, f"{self.symbol}_model.h5")
        self.scaler_path = os.path.join(Config.MODELS_DIR, f"{self.symbol}_scaler.npy")
        self.scaler_max_path = os.path.join(Config.MODELS_DIR, f"{self.symbol}_scaler_max.npy")
        self.model = None
        self.scaler = None
        
    def get_data(self, start_date=None, end_date=None):
        """Fetch historical stock data with fallback to mock data"""
        try:
            print(f"Fetching data for {self.symbol}...")
            
            if start_date is None:
                start_date = Config.START_DATE
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            data = None
            
            # Try real data if not in demo mode
            if not Config.DEMO_MODE:
                try:
                    print("Attempting to fetch real data from Yahoo Finance...")
                    ticker = yf.Ticker(self.symbol)
                    data = ticker.history(period="2y", auto_adjust=True)
                    
                    if not data.empty:
                        print(f"✓ Successfully downloaded {len(data)} days of real data")
                    else:
                        print("⚠ No real data returned, switching to mock data")
                        
                except Exception as e:
                    print(f"⚠ Failed to fetch real data: {str(e)}")
            
            # Use mock data if real data failed or in demo mode
            if data is None or data.empty:
                mode_text = "DEMO MODE - " if Config.DEMO_MODE else ""
                print(f"⚠ {mode_text}Using mock data for {self.symbol}")
                
                from mock_data import generate_mock_stock_data
                data = generate_mock_stock_data(self.symbol)
                print(f"✓ Generated {len(data)} days of mock data")
            
            # Handle MultiIndex columns (sometimes from yfinance)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove any rows with NaN in Close price
            data = data.dropna(subset=['Close'])
            
            if len(data) < Config.SEQUENCE_LENGTH:
                raise ValueError(
                    f"Insufficient data: got {len(data)} days, "
                    f"need at least {Config.SEQUENCE_LENGTH}"
                )
            
            print(f"✓ Data validated: {len(data)} days from {data.index[0].date()} to {data.index[-1].date()}")
            return data
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise
    
    def preprocess_data(self, data, fit_scaler=True):
        """Prepare data for LSTM model"""
        try:
            close_prices = data['Close'].values.reshape(-1, 1)
            
            if fit_scaler:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = self.scaler.fit_transform(close_prices)
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted yet")
                scaled_data = self.scaler.transform(close_prices)
            
            X, y = [], []
            for i in range(Config.SEQUENCE_LENGTH, len(scaled_data)):
                X.append(scaled_data[i-Config.SEQUENCE_LENGTH:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            return X, y
            
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            raise
    
    def build_model(self):
        """Build advanced LSTM model"""
        try:
            model = Sequential([
                Bidirectional(LSTM(100, return_sequences=True, 
                                  input_shape=(Config.SEQUENCE_LENGTH, 1))),
                Dropout(0.2),
                Bidirectional(LSTM(100, return_sequences=True)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
            
            return model
            
        except Exception as e:
            print(f"Error building model: {str(e)}")
            raise
    
    def train_model(self, data):
        """Train the LSTM model"""
        try:
            print(f"Training model for {self.symbol}...")
            
            X, y = self.preprocess_data(data, fit_scaler=True)
            
            if len(X) < 100:
                raise ValueError(f"Insufficient data for training. Need at least 100 samples, got {len(X)}")
            
            # Split data into training and validation sets
            split_idx = int(len(X) * Config.TRAIN_SPLIT)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
            self.model = self.build_model()
            
            # Callbacks for better training
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=Config.EPOCHS,
                batch_size=Config.BATCH_SIZE,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            
            # Save model and scaler
            self.model.save(self.model_path)
            np.save(self.scaler_path, self.scaler.data_min_)
            np.save(self.scaler_max_path, self.scaler.data_max_)
            
            print(f"✓ Model saved to {self.model_path}")
            
            # Calculate training metrics
            train_loss = float(history.history['loss'][-1])
            val_loss = float(history.history['val_loss'][-1])
            
            return {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epochs_trained': len(history.history['loss'])
            }
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise
    
    def load_existing_model(self):
        """Load saved model and scaler"""
        try:
            if not os.path.exists(self.model_path):
                return False
            
            print(f"Loading saved model for {self.symbol}...")
            self.model = load_model(self.model_path)
            
            # Load scaler parameters
            min_val = np.load(self.scaler_path)
            max_val = np.load(self.scaler_max_path)
            
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.data_min_ = min_val
            self.scaler.data_max_ = max_val
            self.scaler.scale_ = 1.0 / (self.scaler.data_max_ - self.scaler.data_min_)
            self.scaler.min_ = -self.scaler.data_min_ * self.scaler.scale_
            
            print("✓ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict_next_price(self, data):
        """Predict next day's closing price"""
        try:
            if len(data) < Config.SEQUENCE_LENGTH:
                raise ValueError(f"Need at least {Config.SEQUENCE_LENGTH} days of data")
            
            recent_data = data['Close'][-Config.SEQUENCE_LENGTH:].values.reshape(-1, 1)
            scaled_recent = self.scaler.transform(recent_data)
            X_pred = np.reshape(scaled_recent, (1, Config.SEQUENCE_LENGTH, 1))
            
            pred_scaled = self.model.predict(X_pred, verbose=0)
            predicted_price = self.scaler.inverse_transform(pred_scaled)[0][0]
            
            return float(predicted_price)
            
        except Exception as e:
            print(f"Error predicting price: {str(e)}")
            raise
    
    def predict_multiple_days(self, data, days=7):
        """Predict prices for multiple days ahead"""
        try:
            predictions = []
            current_sequence = data['Close'][-Config.SEQUENCE_LENGTH:].values.reshape(-1, 1)
            
            for day in range(days):
                scaled_sequence = self.scaler.transform(current_sequence)
                X_pred = np.reshape(scaled_sequence, (1, Config.SEQUENCE_LENGTH, 1))
                
                pred_scaled = self.model.predict(X_pred, verbose=0)
                predicted_price = self.scaler.inverse_transform(pred_scaled)[0][0]
                
                predictions.append(float(predicted_price))
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[1:], [[predicted_price]], axis=0)
            
            return predictions
            
        except Exception as e:
            print(f"Error in multi-day prediction: {str(e)}")
            raise
    
    def get_historical_performance(self, data, days=30):
        """Get historical prices and predictions for performance analysis"""
        try:
            if len(data) < Config.SEQUENCE_LENGTH + days:
                days = max(1, len(data) - Config.SEQUENCE_LENGTH)
            
            actual_prices = []
            predicted_prices = []
            
            for i in range(-days, 0):
                if i - Config.SEQUENCE_LENGTH < -len(data):
                    continue
                    
                sequence = data['Close'][i-Config.SEQUENCE_LENGTH:i].values.reshape(-1, 1)
                scaled_sequence = self.scaler.transform(sequence)
                X_pred = np.reshape(scaled_sequence, (1, Config.SEQUENCE_LENGTH, 1))
                
                pred_scaled = self.model.predict(X_pred, verbose=0)
                predicted_price = self.scaler.inverse_transform(pred_scaled)[0][0]
                
                actual_prices.append(float(data['Close'].iloc[i]))
                predicted_prices.append(float(predicted_price))
            
            return actual_prices, predicted_prices
            
        except Exception as e:
            print(f"Error in historical performance: {str(e)}")
            return [], []


def predict_market(symbol: str, retrain=False):
    """Main prediction function - Entry point for predictions"""
    try:
        # Validate symbol
        if not validate_symbol(symbol):
            return {
                'success': False,
                'error': 'Invalid symbol format'
            }
        
        print(f"\n{'='*60}")
        print(f"Starting prediction for {symbol}")
        if Config.DEMO_MODE:
            print("⚠ DEMO MODE: Using mock data")
        print(f"{'='*60}\n")
        
        # Create predictor
        predictor = StockPredictor(symbol)
        
        # Get data
        data = predictor.get_data()
        print(f"Data shape: {data.shape}")
        
        # Add technical indicators
        try:
            data = calculate_technical_indicators(data)
            print("✓ Technical indicators calculated")
        except Exception as e:
            print(f"⚠ Warning: Could not calculate all technical indicators: {str(e)}")
        
        # Load or train model
        train_metrics = None
        if retrain or not predictor.load_existing_model():
            print("Training new model...")
            train_metrics = predictor.train_model(data)
            print(f"✓ Training complete: {train_metrics}")
        
        # Make predictions
        print("Making predictions...")
        next_price = predictor.predict_next_price(data)
        print(f"✓ Next day prediction: ${next_price:.2f}")
        
        # Get multi-day predictions
        week_predictions = predictor.predict_multiple_days(data, days=7)
        print(f"✓ Week predictions generated: {len(week_predictions)} days")
        
        # Calculate confidence
        actual, predicted = predictor.get_historical_performance(data, days=30)
        if len(actual) > 0 and len(predicted) > 0:
            confidence = calculate_confidence_score(np.array(actual), np.array(predicted))
        else:
            confidence = 50.0
        print(f"✓ Confidence score: {confidence}")
        
        # Get sentiment
        try:
            sentiment = calculate_sentiment_score(data)
        except:
            sentiment = 50.0
        print(f"✓ Sentiment score: {sentiment}")
        
        # Get stock info
        stock_info = get_stock_info(symbol)
        
        # Current price and statistics
        current_price = float(data['Close'].iloc[-1])
        price_change = ((next_price - current_price) / current_price) * 100
        
        # Get technical indicators safely
        technical_indicators = {}
        try:
            technical_indicators['rsi'] = float(data['RSI'].iloc[-1]) if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]) else None
        except:
            technical_indicators['rsi'] = None
            
        try:
            technical_indicators['macd'] = float(data['MACD'].iloc[-1]) if 'MACD' in data.columns and not pd.isna(data['MACD'].iloc[-1]) else None
        except:
            technical_indicators['macd'] = None
            
        try:
            technical_indicators['sma_20'] = float(data['SMA_20'].iloc[-1]) if 'SMA_20' in data.columns and not pd.isna(data['SMA_20'].iloc[-1]) else None
        except:
            technical_indicators['sma_20'] = None
        
        # Get historical data safely
        try:
            hist_dates = data.index[-30:].strftime('%Y-%m-%d').tolist()
            hist_prices = [float(p) for p in data['Close'][-30:].tolist()]
        except:
            hist_dates = []
            hist_prices = []
        
        # Build result
        result = {
            'success': True,
            'symbol': symbol.upper(),
            'stock_info': stock_info,
            'current_price': current_price,
            'predicted_price': next_price,
            'price_change_percent': round(price_change, 2),
            'week_predictions': week_predictions,
            'confidence_score': confidence,
            'sentiment_score': sentiment,
            'train_metrics': train_metrics,
            'technical_indicators': technical_indicators,
            'historical_data': {
                'dates': hist_dates,
                'prices': hist_prices
            },
            'demo_mode': Config.DEMO_MODE
        }
        
        print(f"\n{'='*60}")
        print(f"✓ Prediction successful for {symbol}")
        print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n{'='*60}")
        print(f"✗ ERROR in predict_market: {error_msg}")
        print(f"{'='*60}\n")
        
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': error_msg
        }


def get_historical_data(symbol: str, period='1y'):
    """Get historical data for charts"""
    try:
        if not validate_symbol(symbol):
            return {
                'success': False,
                'error': 'Invalid symbol format'
            }
        
        predictor = StockPredictor(symbol)
        data = predictor.get_data()
        
        # Determine period
        if period == '1m':
            data = data[-30:]
        elif period == '3m':
            data = data[-90:]
        elif period == '6m':
            data = data[-180:]
        elif period == '1y':
            data = data[-365:]
        # else keep all data
        
        data = calculate_technical_indicators(data)
        
        # Safely convert to lists
        def safe_list(series):
            try:
                return [float(x) if not pd.isna(x) else None for x in series.tolist()]
            except:
                return []
        
        return {
            'success': True,
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'open': safe_list(data['Open']),
            'high': safe_list(data['High']),
            'low': safe_list(data['Low']),
            'close': safe_list(data['Close']),
            'volume': safe_list(data['Volume']),
            'sma_20': safe_list(data['SMA_20']) if 'SMA_20' in data.columns else [],
            'ema_20': safe_list(data['EMA_20']) if 'EMA_20' in data.columns else [],
            'rsi': safe_list(data['RSI']) if 'RSI' in data.columns else [],
            'demo_mode': Config.DEMO_MODE
        }
        
    except Exception as e:
        print(f"Error in get_historical_data: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e)
        }