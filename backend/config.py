import os
from datetime import datetime, timedelta

class Config:
    """Application configuration"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Model parameters
    SEQUENCE_LENGTH = 60
    TRAIN_SPLIT = 0.8
    EPOCHS = 5  # Reduced for faster training
    BATCH_SIZE = 32
    
    # Data parameters - Use 2 years of data
    START_DATE = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    
    # Demo mode - Set to True to use mock data (when Yahoo Finance is blocked)
    # Set to False to try real data from Yahoo Finance
    DEMO_MODE = True  # ← CHANGE THIS TO False TO TRY REAL DATA
    
    # API settings
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000