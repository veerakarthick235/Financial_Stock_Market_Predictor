# 🚀 AI-Powered Financial Market Prediction System

A cutting-edge stock market prediction application using LSTM neural networks with a futuristic AI-themed interface.

## ✨ Features

- **Real-time Stock Prediction**: AI-powered next-day price forecasts
- **7-Day Forecasting**: Week-ahead price predictions
- **Technical Analysis**: RSI, MACD, SMA, EMA indicators
- **Market Sentiment Analysis**: AI-calculated market sentiment scores
- **Confidence Scoring**: Prediction reliability metrics
- **Historical Charts**: Interactive price history visualization
- **Custom Model Training**: Train AI models for specific stocks
- **Portfolio Tracking**: Monitor your investments
- **Futuristic UI**: Cyberpunk-inspired design with animations

## 🛠️ Installation

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Setup

Simply open `frontend/index.html` in a modern web browser, or use a local server:

```bash
cd frontend
python -m http.server 8000
```

Then visit `http://localhost:8000`

## 🎮 Usage

1. **Predict**: Enter a stock symbol (e.g., AAPL, TSLA) and click "Predict"
2. **Analyze**: View detailed technical analysis with charts
3. **Train AI**: Train custom models for specific stocks
4. **Portfolio**: Track your investment portfolio

## 🔧 Configuration

Edit `backend/config.py` to customize:
- Model parameters (epochs, batch size, etc.)
- Data range
- API settings

## 📊 API Endpoints

- `POST /predict` - Predict stock prices
- `POST /historical` - Get historical data
- `POST /train` - Train new model
- `GET /models` - List available models

## 🎨 Tech Stack

**Backend:**
- Flask
- TensorFlow/Keras
- NumPy, Pandas
- yFinance
- scikit-learn

**Frontend:**
- HTML5, CSS3, JavaScript
- Chart.js
- Custom animations

## 📝 License

MIT License

## ⚠️ Disclaimer

This tool is for educational purposes only. Do not use for actual trading without proper research and risk management.