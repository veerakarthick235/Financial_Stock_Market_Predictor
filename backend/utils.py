import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def calculate_technical_indicators(df):
    """Calculate technical indicators for the stock data"""
    try:
        # Make a copy to avoid warnings
        df = df.copy()
        
        # Simple Moving Average
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # ATR (Average True Range) for volatility
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        return df
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return df


def calculate_sentiment_score(df):
    """Calculate market sentiment based on technical indicators"""
    try:
        sentiment_score = 50  # Start neutral
        
        # RSI sentiment (0-100 scale)
        if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
            last_rsi = df['RSI'].iloc[-1]
            if last_rsi > 70:
                sentiment_score -= 15  # Overbought - bearish
            elif last_rsi > 60:
                sentiment_score += 5   # Slightly bullish
            elif last_rsi < 30:
                sentiment_score += 15  # Oversold - bullish
            elif last_rsi < 40:
                sentiment_score -= 5   # Slightly bearish
        
        # MACD sentiment
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            if not pd.isna(df['MACD'].iloc[-1]) and not pd.isna(df['MACD_Signal'].iloc[-1]):
                macd_diff = df['MACD'].iloc[-1] - df['MACD_Signal'].iloc[-1]
                if macd_diff > 0:
                    sentiment_score += 10  # Bullish
                else:
                    sentiment_score -= 10  # Bearish
        
        # Price vs Moving Averages
        if 'SMA_20' in df.columns and not pd.isna(df['SMA_20'].iloc[-1]):
            if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1]:
                sentiment_score += 8  # Price above MA - bullish
            else:
                sentiment_score -= 8  # Price below MA - bearish
        
        # Price momentum (20-day change)
        if len(df) >= 20:
            price_change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / 
                               df['Close'].iloc[-20]) * 100
            # Cap the impact
            sentiment_score += min(max(price_change_pct * 2, -15), 15)
        
        # Volume trend
        if 'Volume_SMA' in df.columns and not pd.isna(df['Volume_SMA'].iloc[-1]):
            if df['Volume'].iloc[-1] > df['Volume_SMA'].iloc[-1] * 1.2:
                # High volume - amplify existing trend
                if sentiment_score > 50:
                    sentiment_score += 5
                else:
                    sentiment_score -= 5
        
        # Ensure score is between 0 and 100
        return max(0, min(100, sentiment_score))
        
    except Exception as e:
        print(f"Error calculating sentiment: {e}")
        return 50  # Return neutral on error


def get_stock_info(symbol):
    """Get detailed stock information with fallback to mock data"""
    try:
        from config import Config
        
        # Try real data if not in demo mode
        if not Config.DEMO_MODE:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                
                # Check if we got valid data
                if info and ('longName' in info or 'shortName' in info):
                    return {
                        'name': info.get('longName', info.get('shortName', symbol)),
                        'sector': info.get('sector', 'N/A'),
                        'industry': info.get('industry', 'N/A'),
                        'marketCap': format_market_cap(info.get('marketCap')),
                        'currency': info.get('currency', 'USD'),
                        'exchange': info.get('exchange', 'N/A')
                    }
            except Exception as e:
                print(f"Could not get real stock info: {e}")
        
        # Fallback to mock data
        from mock_data import get_mock_stock_info
        return get_mock_stock_info(symbol)
        
    except Exception as e:
        print(f"Error in get_stock_info: {e}")
        return {
            'name': symbol,
            'sector': 'N/A',
            'industry': 'N/A',
            'marketCap': 'N/A',
            'currency': 'USD',
            'exchange': 'N/A'
        }


def format_market_cap(market_cap):
    """Format market cap into readable string"""
    try:
        if market_cap is None or market_cap == 'N/A':
            return 'N/A'
        
        market_cap = float(market_cap)
        
        if market_cap >= 1_000_000_000_000:  # Trillion
            return f"${market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:  # Billion
            return f"${market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:  # Million
            return f"${market_cap / 1_000_000:.2f}M"
        else:
            return f"${market_cap:,.0f}"
    except:
        return 'N/A'


def calculate_confidence_score(actual_prices, predicted_prices):
    """Calculate prediction confidence based on historical accuracy"""
    try:
        if len(actual_prices) < 2 or len(predicted_prices) < 2:
            return 50.0
        
        # Remove any NaN or infinite values
        valid_idx = ~(np.isnan(actual_prices) | np.isnan(predicted_prices) | 
                     np.isinf(actual_prices) | np.isinf(predicted_prices))
        
        actual_prices = actual_prices[valid_idx]
        predicted_prices = predicted_prices[valid_idx]
        
        if len(actual_prices) < 2:
            return 50.0
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        actual_prices_safe = np.where(actual_prices == 0, 0.0001, actual_prices)
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices_safe)) * 100
        
        # Convert to confidence score (inverse relationship)
        # MAPE of 0% = 100% confidence, MAPE of 10%+ = 0% confidence
        confidence = max(0, min(100, 100 - (mape * 10)))
        
        return round(float(confidence), 2)
        
    except Exception as e:
        print(f"Error calculating confidence: {e}")
        return 50.0


def validate_symbol(symbol):
    """Validate if symbol is in correct format"""
    if not symbol:
        return False
    
    # Remove whitespace
    symbol = symbol.strip().upper()
    
    # Basic validation
    if len(symbol) < 1 or len(symbol) > 10:
        return False
    
    # Allow alphanumeric and dash (for crypto like BTC-USD)
    if not all(c.isalnum() or c == '-' for c in symbol):
        return False
    
    
    return True