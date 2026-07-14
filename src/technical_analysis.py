import yfinance as yf
import pandas as pd
import traceback

def get_technical_indicators(ticker, period="6mo"):
    """Calculate RSI, MACD, EMA50, EMA200, and P/E Ratio for a stock"""
    try:
        # ← NEW: Map common stock names to correct tickers
        stock_mapping = {
    # Nifty 50 Stocks
        'TCS': 'TCS',
        'TATA CONSULTANCY': 'TCS',
        'INFOSYS': 'INFOSYS',
        'INFY': 'INFOSYS',
        'HDFC': 'HDFCBANK',
        'HDFCBANK': 'HDFCBANK',
        'HDFC BANK': 'HDFCBANK',
        'ICICI': 'ICICIBANK',
        'ICICIBANK': 'ICICIBANK',
        'ICICI BANK': 'ICICIBANK',
        'AXISBANK': 'AXISBANK',
        'AXIS': 'AXISBANK',
        'AXIS BANK': 'AXISBANK',
        'SBIN': 'SBIN',
        'STATE BANK': 'SBIN',
        'STATE BANK OF INDIA': 'SBIN',
        'SBI': 'SBIN',
        'RELIANCE': 'RELIANCE',
        'RIL': 'RELIANCE',
        'ITC': 'ITC',
        'LT': 'LT',
        'LTNT': 'LT',
        'LARSEN': 'LT',
        'MARUTI': 'MARUTI',
        'MARUTI SUZUKI': 'MARUTI',
        'WIPRO': 'WIPRO',
        'JSWSTEEL': 'JSWSTEEL',
        'JSW STEEL': 'JSWSTEEL',
        'NTPC': 'NTPC',
        'POWER GRID': 'POWERGRID',
        'POWERGRID': 'POWERGRID',
        'COAL INDIA': 'COALINDIA',
        'COALINDIA': 'COALINDIA',
        'SUNPHARMA': 'SUNPHARMA',
        'SUN PHARMA': 'SUNPHARMA',
        'ASIANPAINT': 'ASIANPAINT',
        'ASIAN PAINTS': 'ASIANPAINT',
        'BAJAJ FINSERV': 'BAJAJFINSV',
        'BAJAJFINSV': 'BAJAJFINSV',
        'BAJAJ AUTO': 'BAJAJ-AUTO',
        'BHARATI': 'BHARTIARTL',
        'BHARATIARTL': 'BHARTIARTL',
        'BHARTI AIRTEL': 'BHARTIARTL',
        'AIRTEL': 'BHARTIARTL',
        'BEL': 'BEL',
        'BHARAT': 'BEL',
        'BHARATELECTRONICS': 'BEL',
        'BHARAT ELECTRONICS': 'BEL',
        'BPCL': 'BPCL',
        'BHARAT PETROLEUM': 'BPCL',
        'BRITANNIA': 'BRITANNIA',
        'CIPLA': 'CIPLA',
        'DMART': 'DMART',
        'DMart': 'DMART',
        'AVENUE SUPERMARTS': 'DMART',
        'EICHERMOT': 'EICHERMOT',
        'EICHER MOTORS': 'EICHERMOT',
        'GAIL': 'GAIL',
        'GAIL INDIA': 'GAIL',
        'GRASIM': 'GRASIM',
        'HINDALCO': 'HINDALCO',
        'HINDUNILVR': 'HINDUNILVR',
        'HINDUSTAN UNILEVER': 'HINDUNILVR',
        'HUL': 'HINDUNILVR',
        'HCLTECH': 'HCLTECH',
        'HCL TECH': 'HCLTECH',
        'HCL TECHNOLOGIES': 'HCLTECH',
        'HDFCLIFE': 'HDFCLIFE',
        'HDFC LIFE': 'HDFCLIFE',
        'IBANK': 'ICICIBANK',
        'INDIASIL': 'SHREECEM',
        'INDIGO': 'INDIGO',
        'INTERGLOBE': 'INDIGO',
        'IOC': 'IOC',
        'IOCL': 'IOC',
        'INDIAN OIL': 'IOC',
        'JSWSTEEL': 'JSWSTEEL',
        'KOTAKBANK': 'KOTAKBANK',
        'KOTAK BANK': 'KOTAKBANK',
        'KOTAK MAHINDRA': 'KOTAKBANK',
        'LTIM': 'LTIM',
        'LTI MINDTREE': 'LTIM',
        'MINDTREE': 'LTIM',
        'M&M': 'MM',
        'MAHINDRA': 'MM',
        'MAHINDRA & MAHINDRA': 'MM',
        'NESTLEIND': 'NESTLEIND',
        'NESTLE': 'NESTLEIND',
        'ONGC': 'ONGC',
        'OIL & NATURAL GAS': 'ONGC',
        'SBILIFE': 'SBILIFE',
        'SBI LIFE': 'SBILIFE',
        'SHREECEM': 'SHREECEM',
        'SHREE CEMENT': 'SHREECEM',
        'TECHM': 'TECHM',
        'TECH MAHINDRA': 'TECHM',
        'TITAN': 'TITAN',
        'TITAN COMPANY': 'TITAN',
        'ULTRACEMCO': 'ULTRACEMCO',
        'ULTRATECH CEMENT': 'ULTRACEMCO',
        'ADANIGREEN': 'ADANIGREEN',
        'ADANI GREEN': 'ADANIGREEN',
        'ADANIPOWERS': 'ADANIPOWERS',
        'ADANI POWER': 'ADANIPOWERS',
        'ADANIPORTS': 'ADANIPORTS',
        'ADANI PORTS': 'ADANIPORTS',
        'BE': 'BEL',
        'BEL LTD': 'BEL',
        'BHARAT ELECTRONICS LTD': 'BEL',

        'BAJAJ': 'BAJAJFINSV',
        'BAJAJFINSERV': 'BAJAJFINSV',
        'BAJAJ FINSERV LTD': 'BAJAJFINSV',

        'BAJAJFINANCE': 'BAJFINANCE',
        'BAJAJ FINANCE': 'BAJFINANCE',

        'LIC': 'LICI',
        'LIFE INSURANCE': 'LICI'
        }

        # <<< CHANGED >>>
        # <<< CHANGED >>>
        ticker = ticker.strip().upper()
        ticker = " ".join(ticker.split())

# First try exact match
        mapped_ticker = stock_mapping.get(ticker)

# If exact match not found, try partial matching
        if mapped_ticker is None:
            for key, value in stock_mapping.items():
                if ticker in key or key in ticker:
                    mapped_ticker = value
                    print(f"Partial Match Found: '{ticker}' -> '{mapped_ticker}'")
                    break

# If still not found, use whatever user entered
        ticker = mapped_ticker if mapped_ticker else ticker
        # Ensure ticker has .NS suffix for Indian stocks
        if not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
        
        # <<< CHANGED >>>
        print(f"Downloading ticker: {ticker}")

        try:
            data = yf.download(
                ticker,
                period=period,
                auto_adjust=False,
                progress=False
            )

    # Sometimes yfinance doesn't raise an exception, it just returns an empty DataFrame
            if data.empty:
                raise ValueError("Empty data returned")

        except Exception as download_error:
            print(f"Download Error: {download_error}")

    # Try again without the .NS suffix
            ticker_no_ns = ticker.replace(".NS", "")
            print(f"Trying fallback ticker: {ticker_no_ns}")

            try:
                data = yf.download(
                ticker_no_ns,
                period=period,
                auto_adjust=False,
                progress=False
            )

                if data.empty:
                
                    print("========== DEBUG ==========")
                    print(type(data))
                    print(data.columns)
                    print(data.head())
                    print("===========================")
                    return {"error": f"No data found for {ticker}"}

            except Exception as fallback_error:
                print("========== FALLBACK ERROR ==========")
                traceback.print_exc()
                print("====================================")
                return {"error": f"Failed downloading data for {ticker}"}
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # <<< CHANGED >>>
        required_cols = ["Open", "High", "Low", "Close"]

        for col in required_cols:
            if col not in data.columns:
                return {"error": f"{col} column missing for {ticker}"}

        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=required_cols)

        if data.empty:
            return {"error": f"No valid OHLC data for {ticker}"}
        # <<< CHANGED >>>
        close = data["Close"]

        print(data.columns)
        print(type(data["Close"]))

        # RSI (Relative Strength Index)
        try:
            # <<< CHANGED >>>
            #close = pd.to_numeric(data["Close"], errors="coerce")
            #close = close.dropna()
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 0
        except Exception as rsi_error:
            print(f"RSI calculation error: {str(rsi_error)}")
            current_rsi = 0
        
        # MACD
        try:
            # <<< CHANGED >>>
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_value = round(float(macd.iloc[-1]), 2)
            macd_signal = round(float(signal.iloc[-1]), 2)
        except Exception as macd_error:
            print(f"MACD calculation error: {str(macd_error)}")
            macd_value = 0
            macd_signal = 0
        
        # Calculate 50 EMA and 200 EMA
        try:
            ema_50 = close.ewm(span=50).mean()
            ema_200 = close.ewm(span=200).mean()
            current_ema_50 = round(float(ema_50.iloc[-1]), 2) if len(ema_50) > 0 and not pd.isna(ema_50.iloc[-1]) else 0
            current_ema_200 = round(float(ema_200.iloc[-1]), 2) if len(ema_200) > 0 and not pd.isna(ema_200.iloc[-1]) else 0
        except Exception as ema_error:
            print(f"EMA calculation error: {str(ema_error)}")
            current_ema_50 = 0
            current_ema_200 = 0
        
        # Get P/E Ratio
        pe_ratio = 0
        try:
            stock_info = yf.Ticker(ticker).info
            pe_ratio = round(stock_info.get('trailingPE', 0), 2) if stock_info.get('trailingPE') else 0
        except Exception as pe_error:
            print(f"P/E ratio fetch error: {str(pe_error)}")
            pe_ratio = 0
        
        return {
            "current_rsi": round(current_rsi, 2),
            "rsi_status": "Overbought (>70)" if current_rsi > 70 else "Oversold (<30)" if current_rsi < 30 else "Neutral",
            "current_price": round(float(close.iloc[-1]), 2),
            "candlestick_data": prepare_candlestick_data(data),
            "macd": macd_value,
            "macd_signal": macd_signal,
            "ema_50": current_ema_50,
            "ema_200": current_ema_200,
            "pe_ratio": pe_ratio
        }
    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

def prepare_candlestick_data(data):
    """Format OHLC data for Plotly"""

    result = []

    try:
        for index, row in data.iterrows():

            if pd.isna(row["Open"]) or pd.isna(row["High"]) \
            or pd.isna(row["Low"]) or pd.isna(row["Close"]):
                continue

            result.append({

                "date": index.strftime("%Y-%m-%d"),

                "open": round(float(row["Open"]),2),

                "high": round(float(row["High"]),2),

                "low": round(float(row["Low"]),2),

                "close": round(float(row["Close"]),2)

            })

        return result

    except Exception as e:

        print(f"Candlestick Error : {e}")

    return []