import yfinance as yf
import pandas as pd

print("="*60)
print("Testing yfinance connectivity...")
print(f"yfinance version: {yf.__version__}")
print("="*60)

# Test 1: Simple ticker
print("\n[Test 1] Fetching AAPL data using Ticker...")
try:
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="5d")
    if len(hist) > 0:
        print(f"✓ SUCCESS: Got {len(hist)} days of AAPL data")
        print(hist.tail())
    else:
        print("✗ FAILED: No data returned")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 2: Download method
print("\n[Test 2] Downloading MSFT data...")
try:
    data = yf.download("MSFT", period="5d", progress=False)
    if len(data) > 0:
        print(f"✓ SUCCESS: Downloaded {len(data)} days of MSFT data")
        print(data.tail())
    else:
        print("✗ FAILED: No data returned")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 3: BTC-USD
print("\n[Test 3] Downloading BTC-USD data...")
try:
    data = yf.download("BTC-USD", period="5d", progress=False)
    if len(data) > 0:
        print(f"✓ SUCCESS: Downloaded {len(data)} days of BTC-USD data")
    else:
        print("✗ FAILED: No data returned")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "="*60)
print("Note: If all tests fail, the app will use MOCK DATA")
print("Set DEMO_MODE = True in config.py to use mock data")
print("="*60)