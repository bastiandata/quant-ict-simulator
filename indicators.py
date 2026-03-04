"""
indicators.py (V2) — Indicadores técnicos mejorados.
RSI, MACD, EMA, Bollinger, Fibonacci, ATR, VWAP, EWMA Volatility.
"""
import numpy as np
import pandas as pd


# ────────────────────────────────────────────
#  RSI
# ────────────────────────────────────────────
def calc_rsi(df, period=14):
    """RSI con método Wilder."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


# ────────────────────────────────────────────
#  MACD
# ────────────────────────────────────────────
def calc_macd(df, fast=12, slow=26, signal_period=9):
    """MACD line, signal line y histograma."""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    return df


# ────────────────────────────────────────────
#  EMA
# ────────────────────────────────────────────
def calc_ema(df, periods=None):
    """EMAs múltiples."""
    if periods is None:
        periods = [9, 21, 50, 200]
    for p in periods:
        df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
    return df


# ────────────────────────────────────────────
#  Bollinger Bands
# ────────────────────────────────────────────
def calc_bollinger(df, period=20, std_dev=2):
    """Bandas de Bollinger."""
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
    return df


# ────────────────────────────────────────────
#  ATR (Average True Range)
# ────────────────────────────────────────────
def calc_atr(df, period=14):
    """ATR para dimensionar SL/TP dinámicamente."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.ewm(span=period, adjust=False).mean()
    return df


# ────────────────────────────────────────────
#  VWAP (Volume Weighted Average Price)
# ────────────────────────────────────────────
def calc_vwap(df):
    """VWAP acumulado (reset diario simplificado: usa toda la serie)."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_tp_vol = (typical_price * df['volume']).cumsum()
    cumulative_vol = df['volume'].cumsum()
    df['vwap'] = cumulative_tp_vol / cumulative_vol
    return df


# ────────────────────────────────────────────
#  EWMA Volatility (reemplaza σ estática)
# ────────────────────────────────────────────
def calc_ewma_volatility(df, span=30):
    """
    Volatilidad exponencialmente ponderada.
    Da más peso a la volatilidad reciente (captura clustering).
    """
    returns = df['close'].pct_change()
    df['ewma_vol'] = returns.ewm(span=span, adjust=False).std()
    return df


# ────────────────────────────────────────────
#  FIBONACCI
# ────────────────────────────────────────────
FIBONACCI_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]

def calc_fibonacci_levels(swing_high, swing_low):
    """Niveles Fib entre swing high y swing low."""
    diff = swing_high - swing_low
    levels = {}
    for ratio in FIBONACCI_RATIOS:
        levels[ratio] = swing_high - (diff * ratio)
    return levels


def find_multiple_fibonacci_tps(df, entry_price, fib_levels, direction, lookback=150, max_tps=3):
    """
    Encuentra múltiples TPs de Fibonacci en la dirección del trade.
    Retorna una lista de TPs viables ordenados desde el más cercano al más lejano.
    """
    recent = df.tail(lookback)
    tolerance = entry_price * 0.003
    
    viable_tps = []

    for ratio, level in fib_levels.items():
        if direction == 'LONG' and level <= entry_price:
            continue
        if direction == 'SHORT' and level >= entry_price:
            continue

        touches = ((recent['high'] >= level - tolerance) &
                   (recent['low'] <= level + tolerance)).sum()
        distance = abs(level - entry_price) / entry_price
        
        score = (touches / distance) if distance > 0 else 0
        
        viable_tps.append({
            'level': level,
            'ratio': ratio,
            'score': score,
            'distance': distance
        })

    # Ordenar por distancia (del más cercano al más lejano)
    viable_tps.sort(key=lambda x: x['distance'])
    
    return viable_tps[:max_tps]


# ────────────────────────────────────────────
#  SEÑALES DE INDICADORES
# ────────────────────────────────────────────
def generate_indicator_signals(df, direction):
    """
    Genera señales de todos los indicadores.
    Retorna lista de (nombre, señal, detalle).
    señal: 'bullish', 'bearish', 'neutral'
    """
    signals = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # RSI
    if 'rsi' in df.columns:
        rsi = last['rsi']
        if rsi < 30:
            signals.append(('RSI', 'bullish', f'{rsi:.1f} — Sobreventa'))
        elif rsi > 70:
            signals.append(('RSI', 'bearish', f'{rsi:.1f} — Sobrecompra'))
        elif rsi < 45:
            signals.append(('RSI', 'bullish', f'{rsi:.1f} — Zona baja'))
        elif rsi > 55:
            signals.append(('RSI', 'bearish', f'{rsi:.1f} — Zona alta'))
        else:
            signals.append(('RSI', 'neutral', f'{rsi:.1f} — Neutral'))

    # MACD
    if 'macd_line' in df.columns:
        macd_hist = last['macd_hist']
        prev_hist = prev['macd_hist']
        if last['macd_line'] > last['macd_signal']:
            if macd_hist > prev_hist:
                signals.append(('MACD', 'bullish', 'Cruce alcista + momentum creciente'))
            else:
                signals.append(('MACD', 'bullish', 'Por encima de señal'))
        elif last['macd_line'] < last['macd_signal']:
            if macd_hist < prev_hist:
                signals.append(('MACD', 'bearish', 'Cruce bajista + momentum decreciente'))
            else:
                signals.append(('MACD', 'bearish', 'Por debajo de señal'))
        else:
            signals.append(('MACD', 'neutral', 'Sin señal clara'))

    # EMA Stack
    if all(f'ema_{p}' in df.columns for p in [9, 21, 50]):
        e9, e21, e50 = last['ema_9'], last['ema_21'], last['ema_50']
        if e9 > e21 > e50:
            signals.append(('EMA Stack', 'bullish', 'EMA9 > EMA21 > EMA50'))
        elif e9 < e21 < e50:
            signals.append(('EMA Stack', 'bearish', 'EMA9 < EMA21 < EMA50'))
        else:
            signals.append(('EMA Stack', 'neutral', 'Sin alineación clara'))

    # Bollinger
    if 'bb_lower' in df.columns:
        close = last['close']
        if close <= last['bb_lower']:
            signals.append(('Bollinger', 'bullish', 'Precio en banda inferior'))
        elif close >= last['bb_upper']:
            signals.append(('Bollinger', 'bearish', 'Precio en banda superior'))
        elif close < last['bb_middle']:
            signals.append(('Bollinger', 'bullish', 'Precio bajo media BB'))
        else:
            signals.append(('Bollinger', 'bearish', 'Precio sobre media BB'))

    # VWAP
    if 'vwap' in df.columns:
        close = last['close']
        vwap = last['vwap']
        if close > vwap * 1.005:
            signals.append(('VWAP', 'bullish', f'Precio sobre VWAP (${vwap:.2f})'))
        elif close < vwap * 0.995:
            signals.append(('VWAP', 'bearish', f'Precio bajo VWAP (${vwap:.2f})'))
        else:
            signals.append(('VWAP', 'neutral', f'Precio ≈ VWAP (${vwap:.2f})'))

    return signals
