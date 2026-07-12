import yfinance as yf
import pandas as pd

def get_technical_indicators(ticker, period="6mo"):
    """Calculate RSI, MACD, EMA50, EMA200, and P/E Ratio for a stock"""
    try:
        # Ensure ticker has .NS suffix for Indian stocks
        if not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
        
        data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty:
            return {"error": f"No data found for {ticker}"}
        
        print(data.columns)
        print(type(data["Close"]))

        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        
        # ← NEW: Calculate 50 EMA and 200 EMA
        ema_50 = data['Close'].ewm(span=50).mean()
        ema_200 = data['Close'].ewm(span=200).mean()
        
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 0
        current_ema_50 = round(float(ema_50.iloc[-1]), 2) if not pd.isna(ema_50.iloc[-1]) else 0
        current_ema_200 = round(float(ema_200.iloc[-1]), 2) if not pd.isna(ema_200.iloc[-1]) else 0
        
        # ← NEW: Get P/E Ratio
        try:
            stock_info = yf.Ticker(ticker).info
            pe_ratio = round(stock_info.get('trailingPE', 0), 2)
        except:
            pe_ratio = 0
        
        return {
            "current_rsi": round(current_rsi, 2),
            "rsi_status": "Overbought (>70)" if current_rsi > 70 else "Oversold (<30)" if current_rsi < 30 else "Neutral",
            "current_price": round(float(data['Close'].iloc[-1]), 2),
            "candlestick_data": prepare_candlestick_data(data),
            "macd": round(float(macd.iloc[-1]), 2),
            "macd_signal": round(float(signal.iloc[-1]), 2),
            "ema_50": current_ema_50,  # ← NEW
            "ema_200": current_ema_200,  # ← NEW
            "pe_ratio": pe_ratio  # ← NEW
        }
    except Exception as e:
        return {"error": f"Failed to fetch data: {str(e)}"}

def prepare_candlestick_data(data):
    """Format OHLCV data for charting"""
    try:
        result = []
        for index, row in data.iterrows():
            result.append({
                "date": str(index.date()),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2)
            })
        return result
    except Exception as e:
        return []