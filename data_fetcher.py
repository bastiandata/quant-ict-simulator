"""
data_fetcher.py (V2) — Descarga de datos OHLCV multi-timeframe desde Binance.
"""
import ccxt
import pandas as pd
import yfinance as yf

# Mapeo de timeframes Binance -> yfinance
YF_TIMEFRAMES = {
    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '60m', '4h': '60m', '1d': '1d', '1w': '1wk'
}

def fetch_ohlcv(symbol='ETH/USDT', timeframe='1h', limit=2000):
    """Descarga velas desde Binance Spot o Yahoo Finance (si es acción)."""
    if '/' in symbol:
        # Crypto via Binance
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    else:
        # Stocks via yfinance (e.g., 'AMZN')
        yf_tf = YF_TIMEFRAMES.get(timeframe, '1d')
        # Si piden 4h para stocks, yfinance solo da 60m o 1d, así que se usa 1d o 60m.
        if timeframe == '4h': yf_tf = '60m'
        
        # Calcular periodo según el timeframe y limit para no descargar de más
        if yf_tf in ['1m', '5m', '15m', '30m', '60m']:
            period = '60d' # Limite max para intraday en yf
        else:
            period = '5y'
            
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=yf_tf)
        if df.empty:
            raise ValueError(f"No se encontraron datos en Yahoo Finance para {symbol} con tf={yf_tf}")
        
        df = df.reset_index()
        # El nombre de la columna de fecha puede variar (Date o Datetime)
        date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df = df.rename(columns={
            date_col: 'timestamp', 'Open': 'open', 'High': 'high', 
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
        
        # Eliminar huso horario para compatibilidad
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        return df.tail(limit)


def fetch_multi_timeframe(symbol='ETH/USDT', timeframes=None, limit=500):
    """
    Descarga datos en múltiples temporalidades para confirmación multi-TF.
    Retorna dict {timeframe: DataFrame}.
    """
    if timeframes is None:
        timeframes = ['1h', '4h']

    exchange = ccxt.binance()
    data = {}
    for tf in timeframes:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            data[tf] = df
        except Exception as e:
            print(f"   ⚠️  Error descargando {tf}: {e}")
    return data
