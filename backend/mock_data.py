import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_stock_data(symbol, days=2000):
    """Generate realistic mock stock data for testing/demo"""
    
    # Different starting prices and characteristics for different stocks
    stock_configs = {
        'AAPL': {'base_price': 150, 'volatility': 0.02, 'trend': 0.0003},
        'MSFT': {'base_price': 300, 'volatility': 0.018, 'trend': 0.0004},
        'GOOGL': {'base_price': 140, 'volatility': 0.022, 'trend': 0.0003},
        'TSLA': {'base_price': 200, 'volatility': 0.04, 'trend': 0.0002},
        'AMZN': {'base_price': 130, 'volatility': 0.025, 'trend': 0.0003},
        'BTC-USD': {'base_price': 40000, 'volatility': 0.05, 'trend': 0.0005},
        'ETH-USD': {'base_price': 2500, 'volatility': 0.06, 'trend': 0.0004},
        'NVDA': {'base_price': 400, 'volatility': 0.035, 'trend': 0.0006},
        'META': {'base_price': 300, 'volatility': 0.028, 'trend': 0.0003},
        'NFLX': {'base_price': 400, 'volatility': 0.032, 'trend': 0.0002},
    }
    
    # Get config or use default
    config = stock_configs.get(symbol, {'base_price': 100, 'volatility': 0.02, 'trend': 0.0003})
    base_price = config['base_price']
    volatility = config['volatility']
    trend_factor = config['trend']
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Set random seed for consistent data per symbol
    np.random.seed(hash(symbol) % (2**32 - 1))
    
    # Generate price movements with trend
    trend = np.linspace(0, 1, len(dates)) * trend_factor
    random_walk = np.random.normal(0, volatility, len(dates))
    
    # Combine trend and random walk
    returns = trend + random_walk
    
    # Calculate cumulative returns to get prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add some cyclical patterns (simulate market cycles)
    cycle = np.sin(np.linspace(0, 8 * np.pi, len(dates))) * base_price * 0.1
    prices = prices + cycle
    
    # Generate OHLC data
    data = pd.DataFrame(index=dates)
    
    # Close prices
    data['Close'] = prices
    
    # Open prices (close of previous day with small gap)
    data['Open'] = data['Close'].shift(1) * (1 + np.random.uniform(-0.005, 0.005, len(dates)))
    data['Open'].iloc[0] = data['Close'].iloc[0] * 0.99
    
    # High prices (max of open/close + some variation)
    data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.015, len(dates)))
    
    # Low prices (min of open/close - some variation)
    data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.015, len(dates)))
    
    # Volume (with some correlation to price movement)
    price_changes = np.abs(data['Close'].pct_change().fillna(0))
    base_volume = np.random.randint(50000000, 150000000)
    data['Volume'] = (base_volume * (1 + price_changes * 10) * 
                     np.random.uniform(0.5, 1.5, len(dates))).astype(int)
    
    # Ensure data integrity
    data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
    data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
    
    # Remove any NaN values
    data = data.fillna(method='bfill')
    
    return data


def get_mock_stock_info(symbol):
    """Generate mock stock information"""
    
    stock_info_db = {
        'AAPL': {
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': '2,800,000,000,000',
            'currency': 'USD',
            'exchange': 'NASDAQ'
        },
        'MSFT': {
            'name': 'Microsoft Corporation',
            'sector': 'Technology',
            'industry': 'Software—Infrastructure',
            'marketCap': '2,400,000,000,000',
            'currency': 'USD',
            'exchange': 'NASDAQ'
        },
        'GOOGL': {
            'name': 'Alphabet Inc.',
            'sector': 'Communication Services',
            'industry': 'Internet Content & Information',
            'marketCap': '1,700,000,000,000',
            'currency': 'USD',
            'exchange': 'NASDAQ'
        },
        'TSLA': {
            'name': 'Tesla, Inc.',
            'sector': 'Consumer Cyclical',
            'industry': 'Auto Manufacturers',
            'marketCap': '800,000,000,000',
            'currency': 'USD',
            'exchange': 'NASDAQ'
        },
        'AMZN': {
            'name': 'Amazon.com, Inc.',
            'sector': 'Consumer Cyclical',
            'industry': 'Internet Retail',
            'marketCap': '1,500,000,000,000',
            'currency': 'USD',
            'exchange': 'NASDAQ'
        },
        'BTC-USD': {
            'name': 'Bitcoin USD',
            'sector': 'Cryptocurrency',
            'industry': 'Digital Currency',
            'marketCap': '800,000,000,000',
            'currency': 'USD',
            'exchange': 'CCC'
        },
        'ETH-USD': {
            'name': 'Ethereum USD',
            'sector': 'Cryptocurrency',
            'industry': 'Digital Currency',
            'marketCap': '300,000,000,000',
            'currency': 'USD',
            'exchange': 'CCC'
        },
        'NVDA': {
            'name': 'NVIDIA Corporation',
            'sector': 'Technology',
            'industry': 'Semiconductors',
            'marketCap': '1,200,000,000,000',
            'currency': 'USD',
            'exchange': 'NASDAQ'
        },
        'META': {
            'name': 'Meta Platforms, Inc.',
            'sector': 'Communication Services',
            'industry': 'Internet Content & Information',
            'marketCap': '900,000,000,000',
            'currency': 'USD',
            'exchange': 'NASDAQ'
        },
        'NFLX': {
            'name': 'Netflix, Inc.',
            'sector': 'Communication Services',
            'industry': 'Entertainment',
            'marketCap': '200,000,000,000',
            'currency': 'USD',
            'exchange': 'NASDAQ'
        }
    }
    
    # Return stock info or default
    return stock_info_db.get(symbol, {
        'name': f'{symbol} Corporation',
        'sector': 'Technology',
        'industry': 'Software',
        'marketCap': '100,000,000,000',
        'currency': 'USD',
        'exchange': 'NYSE'
    })