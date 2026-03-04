"""
ict_analysis.py (V2) — Análisis ICT mejorado.
Swing H/L, FVG, Order Blocks, Liquidity, Liquidation,
+ BOS/CHoCH (Break of Structure / Change of Character)
+ ICT Reversal Signal Detection
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


# ────────────────────────────────────────────
#  SWING HIGHS / LOWS
# ────────────────────────────────────────────
def identify_swings(df, window=10):
    """Detecta Swing Highs y Swing Lows."""
    high_idx = argrelextrema(df['high'].values, np.greater_equal, order=window)[0]
    low_idx  = argrelextrema(df['low'].values,  np.less_equal,    order=window)[0]

    df['swing_high'] = np.nan
    df['swing_low']  = np.nan
    df.loc[df.index[high_idx], 'swing_high'] = df['high'].iloc[high_idx].values
    df.loc[df.index[low_idx],  'swing_low']  = df['low'].iloc[low_idx].values

    df['last_swing_high'] = df['swing_high'].ffill()
    df['last_swing_low']  = df['swing_low'].ffill()
    return df


def get_swing_sequence(df, lookback=100):
    """
    Extrae la secuencia de swings recientes como lista ordenada por tiempo.
    Cada item: {'type': 'high'|'low', 'price': float, 'index': int}
    """
    recent = df.tail(lookback)
    swings = []

    for idx in recent.index:
        if not pd.isna(recent.loc[idx, 'swing_high']):
            swings.append({'type': 'high', 'price': recent.loc[idx, 'swing_high'], 'index': idx})
        if not pd.isna(recent.loc[idx, 'swing_low']):
            swings.append({'type': 'low', 'price': recent.loc[idx, 'swing_low'], 'index': idx})

    swings.sort(key=lambda x: x['index'])
    return swings


# ────────────────────────────────────────────
#  BOS / CHoCH (Break of Structure / Change of Character)
# ────────────────────────────────────────────
def detect_bos_choch(df, swings):
    """
    Detecta Break of Structure (BOS) y Change of Character (CHoCH).

    TENDENCIA ALCISTA (Higher Highs + Higher Lows):
      - BOS Bullish: precio rompe un swing high anterior → continuación alcista
      - CHoCH Bearish: precio rompe un swing low anterior → posible cambio a bajista

    TENDENCIA BAJISTA (Lower Highs + Lower Lows):
      - BOS Bearish: precio rompe un swing low anterior → continuación bajista
      - CHoCH Bullish: precio rompe un swing high anterior → posible cambio a alcista

    Retorna:
      - current_trend: 'bullish', 'bearish', 'neutral'
      - last_signal: dict con tipo, dirección y timestamp del último BOS/CHoCH
      - signals: lista de todos los BOS/CHoCH detectados
    """
    if len(swings) < 4:
        return 'neutral', None, []

    signals = []

    # Extraer secuencia de highs y lows separadamente
    highs = [s for s in swings if s['type'] == 'high']
    lows  = [s for s in swings if s['type'] == 'low']

    if len(highs) < 2 or len(lows) < 2:
        return 'neutral', None, []

    # Determinar estructura: ¿HH+HL (alcista) o LH+LL (bajista)?
    # Analizar los últimos pares de swings
    recent_highs = highs[-4:] if len(highs) >= 4 else highs
    recent_lows  = lows[-4:] if len(lows) >= 4 else lows

    # Contar Higher Highs / Lower Lows
    hh_count = sum(1 for i in range(1, len(recent_highs))
                   if recent_highs[i]['price'] > recent_highs[i-1]['price'])
    ll_count = sum(1 for i in range(1, len(recent_lows))
                   if recent_lows[i]['price'] < recent_lows[i-1]['price'])
    hl_count = sum(1 for i in range(1, len(recent_lows))
                   if recent_lows[i]['price'] > recent_lows[i-1]['price'])
    lh_count = sum(1 for i in range(1, len(recent_highs))
                   if recent_highs[i]['price'] < recent_highs[i-1]['price'])

    # Determinar tendencia dominante
    bull_score = hh_count + hl_count
    bear_score = ll_count + lh_count

    if bull_score > bear_score:
        structure_trend = 'bullish'
    elif bear_score > bull_score:
        structure_trend = 'bearish'
    else:
        structure_trend = 'neutral'

    # Detectar BOS y CHoCH en las últimas velas
    last_high = highs[-1]
    prev_high = highs[-2] if len(highs) >= 2 else None
    last_low  = lows[-1]
    prev_low  = lows[-2] if len(lows) >= 2 else None

    # Verificar si el precio actual rompe niveles
    current_price = df['close'].iloc[-1]

    # BOS / CHoCH basado en la estructura
    if prev_high and prev_low:
        if structure_trend == 'bearish':
            # En tendencia bajista, si el precio rompe un high → CHoCH Bullish (cambio de tendencia)
            if current_price > last_high['price']:
                sig = {
                    'type': 'CHoCH', 'direction': 'bullish',
                    'level': last_high['price'], 'index': last_high['index'],
                    'detail': f"Precio (${current_price:.2f}) rompió Swing High (${last_high['price']:.2f}) en tendencia bajista"
                }
                signals.append(sig)
            # Si rompe un low → BOS Bearish (continuación)
            if current_price < last_low['price']:
                sig = {
                    'type': 'BOS', 'direction': 'bearish',
                    'level': last_low['price'], 'index': last_low['index'],
                    'detail': f"Precio rompió Swing Low (${last_low['price']:.2f}) — continuación bajista"
                }
                signals.append(sig)

        elif structure_trend == 'bullish':
            # En tendencia alcista, si precio rompe un low → CHoCH Bearish
            if current_price < last_low['price']:
                sig = {
                    'type': 'CHoCH', 'direction': 'bearish',
                    'level': last_low['price'], 'index': last_low['index'],
                    'detail': f"Precio (${current_price:.2f}) rompió Swing Low (${last_low['price']:.2f}) en tendencia alcista"
                }
                signals.append(sig)
            # Si rompe un high → BOS Bullish (continuación)
            if current_price > last_high['price']:
                sig = {
                    'type': 'BOS', 'direction': 'bullish',
                    'level': last_high['price'], 'index': last_high['index'],
                    'detail': f"Precio rompió Swing High (${last_high['price']:.2f}) — continuación alcista"
                }
                signals.append(sig)

        else:
            # Neutral: evaluar rupturas como posibles inicios de tendencia
            if current_price > last_high['price']:
                signals.append({
                    'type': 'BOS', 'direction': 'bullish',
                    'level': last_high['price'], 'index': last_high['index'],
                    'detail': f"Ruptura de Swing High → posible inicio alcista"
                })
            if current_price < last_low['price']:
                signals.append({
                    'type': 'BOS', 'direction': 'bearish',
                    'level': last_low['price'], 'index': last_low['index'],
                    'detail': f"Ruptura de Swing Low → posible inicio bajista"
                })

    # Analizar relación de los 2 últimos pares para determinar tendencia actual
    if len(highs) >= 2 and len(lows) >= 2:
        last_hh = highs[-1]['price'] > highs[-2]['price']
        last_hl = lows[-1]['price'] > lows[-2]['price']
        last_lh = highs[-1]['price'] < highs[-2]['price']
        last_ll = lows[-1]['price'] < lows[-2]['price']

        if last_hh and last_hl:
            current_trend = 'bullish'
        elif last_lh and last_ll:
            current_trend = 'bearish'
        elif last_hh and last_ll:
            # Expansión: indecisión, pero con sesgo del último movimiento
            current_trend = 'neutral'
        elif last_lh and last_hl:
            # Contracción: indecisión
            current_trend = 'neutral'
        else:
            current_trend = structure_trend
    else:
        current_trend = structure_trend

    # Si hay un CHoCH, la tendencia cambia
    choch_signals = [s for s in signals if s['type'] == 'CHoCH']
    if choch_signals:
        last_choch = choch_signals[-1]
        current_trend = last_choch['direction']

    last_signal = signals[-1] if signals else None
    return current_trend, last_signal, signals


# ────────────────────────────────────────────
#  ICT REVERSAL SIGNAL DETECTION
# ────────────────────────────────────────────
def detect_ict_reversal_signal(entry_price, direction, bullish_fvgs, bearish_fvgs,
                                bullish_obs, bearish_obs, rsi_value, bos_choch_trend):
    """
    Detecta si las condiciones ICT señalan una REVERSIÓN.
    
    Un reversal signal LONG fuerte se da cuando:
    - Precio en zona de Bullish FVG + Bullish OB (zona de demanda)
    - RSI oversold (< 35)
    - Estructura muestra CHoCH bullish o tendencia cambiando
    
    Retorna: reversal_strength (-1.0 a +1.0)
      +1.0 = señal de reversión alcista muy fuerte
      -1.0 = señal de reversión bajista muy fuerte
       0.0 = sin señal de reversión
    """
    tolerance = entry_price * 0.005
    strength = 0.0

    if direction == 'LONG':
        # ¿Precio en zona de demanda (Bullish FVG)?
        in_fvg = False
        if not bullish_fvgs.empty:
            for _, fvg in bullish_fvgs.iterrows():
                if fvg['fvg_bottom'] - tolerance <= entry_price <= fvg['fvg_top'] + tolerance:
                    in_fvg = True
                    break

        # ¿Precio en Bullish OB?
        in_ob = False
        if not bullish_obs.empty:
            for _, ob in bullish_obs.iterrows():
                if ob['ob_low'] - tolerance <= entry_price <= ob['ob_high'] + tolerance:
                    in_ob = True
                    break

        # Scoring de reversión
        if in_fvg:
            strength += 0.25
        if in_ob:
            strength += 0.25
        if rsi_value is not None and rsi_value < 35:
            strength += 0.25
        elif rsi_value is not None and rsi_value < 45:
            strength += 0.10
        if bos_choch_trend == 'bullish':
            strength += 0.25
        elif bos_choch_trend == 'neutral':
            strength += 0.05

    else:  # SHORT
        in_fvg = False
        if not bearish_fvgs.empty:
            for _, fvg in bearish_fvgs.iterrows():
                if fvg['fvg_bottom'] - tolerance <= entry_price <= fvg['fvg_top'] + tolerance:
                    in_fvg = True
                    break

        in_ob = False
        if not bearish_obs.empty:
            for _, ob in bearish_obs.iterrows():
                if ob['ob_low'] - tolerance <= entry_price <= ob['ob_high'] + tolerance:
                    in_ob = True
                    break

        if in_fvg:
            strength -= 0.25
        if in_ob:
            strength -= 0.25
        if rsi_value is not None and rsi_value > 65:
            strength -= 0.25
        elif rsi_value is not None and rsi_value > 55:
            strength -= 0.10
        if bos_choch_trend == 'bearish':
            strength -= 0.25
        elif bos_choch_trend == 'neutral':
            strength -= 0.05

    return max(-1.0, min(1.0, strength))


# ────────────────────────────────────────────
#  FAIR VALUE GAPS (FVG)
# ────────────────────────────────────────────
def identify_fvg(df):
    """Bullish FVG: low[i+1] > high[i-1] | Bearish FVG: high[i+1] < low[i-1]"""
    bullish, bearish = [], []
    for i in range(1, len(df) - 1):
        prev_high = df['high'].iloc[i - 1]
        prev_low  = df['low'].iloc[i - 1]
        next_high = df['high'].iloc[i + 1]
        next_low  = df['low'].iloc[i + 1]
        ts        = df['timestamp'].iloc[i]

        if next_low > prev_high:
            bullish.append({'timestamp': ts, 'fvg_top': next_low, 'fvg_bottom': prev_high, 'type': 'bullish'})
        if next_high < prev_low:
            bearish.append({'timestamp': ts, 'fvg_top': prev_low, 'fvg_bottom': next_high, 'type': 'bearish'})

    return pd.DataFrame(bullish), pd.DataFrame(bearish)


# ────────────────────────────────────────────
#  ORDER BLOCKS (OB)
# ────────────────────────────────────────────
def identify_order_blocks(df, consecutive_candles=3):
    """Bullish OB: última vela bajista antes de rally. Bearish OB: última alcista antes de caída."""
    bullish, bearish = [], []
    for i in range(len(df) - consecutive_candles):
        if df['close'].iloc[i] < df['open'].iloc[i]:
            if all(df['close'].iloc[i+j] > df['open'].iloc[i+j] for j in range(1, consecutive_candles+1)):
                bullish.append({
                    'timestamp': df['timestamp'].iloc[i],
                    'ob_high': df['high'].iloc[i], 'ob_low': df['low'].iloc[i], 'type': 'bullish'
                })
        if df['close'].iloc[i] > df['open'].iloc[i]:
            if all(df['close'].iloc[i+j] < df['open'].iloc[i+j] for j in range(1, consecutive_candles+1)):
                bearish.append({
                    'timestamp': df['timestamp'].iloc[i],
                    'ob_high': df['high'].iloc[i], 'ob_low': df['low'].iloc[i], 'type': 'bearish'
                })
    return pd.DataFrame(bullish), pd.DataFrame(bearish)


# ────────────────────────────────────────────
#  LIQUIDITY POOLS
# ────────────────────────────────────────────
def identify_liquidity_pools(df, tolerance_pct=0.001, min_touches=2, lookback=200):
    """Equal Highs y Equal Lows — zonas de stops acumulados."""
    recent = df.tail(lookback).reset_index(drop=True)
    equal_highs = _find_equal_levels(recent, 'high', tolerance_pct, min_touches)
    equal_lows  = _find_equal_levels(recent, 'low', tolerance_pct, min_touches)
    return equal_highs, equal_lows


def _find_equal_levels(df, col, tolerance_pct, min_touches):
    values = df[col].values
    levels = []
    used = set()
    for i in range(len(values)):
        if i in used:
            continue
        cluster = [values[i]]
        indices = [i]
        tol = values[i] * tolerance_pct
        for j in range(i + 1, len(values)):
            if j in used:
                continue
            if abs(values[j] - values[i]) <= tol:
                cluster.append(values[j])
                indices.append(j)
                used.add(j)
        if len(cluster) >= min_touches:
            levels.append({
                'level': np.mean(cluster), 'touches': len(cluster),
                'first_touch': df['timestamp'].iloc[indices[0]],
                'last_touch': df['timestamp'].iloc[indices[-1]],
                'type': 'equal_highs' if col == 'high' else 'equal_lows'
            })
        used.add(i)
    return pd.DataFrame(levels)


# ────────────────────────────────────────────
#  LIQUIDITY SWEEPS
# ────────────────────────────────────────────
def detect_liquidity_sweeps(df, equal_highs, equal_lows, lookback=30):
    """Detecta barridos de liquidez (mecha rompe nivel, cierre dentro)."""
    sweeps = []
    recent = df.tail(lookback)

    if not equal_highs.empty:
        for _, pool in equal_highs.iterrows():
            level = pool['level']
            for idx, candle in recent.iterrows():
                if candle['high'] > level and candle['close'] < level:
                    sweeps.append({
                        'timestamp': candle['timestamp'], 'level': level,
                        'sweep_type': 'bearish_sweep', 'wick': candle['high'], 'close': candle['close']
                    })

    if not equal_lows.empty:
        for _, pool in equal_lows.iterrows():
            level = pool['level']
            for idx, candle in recent.iterrows():
                if candle['low'] < level and candle['close'] > level:
                    sweeps.append({
                        'timestamp': candle['timestamp'], 'level': level,
                        'sweep_type': 'bullish_sweep', 'wick': candle['low'], 'close': candle['close']
                    })

    return pd.DataFrame(sweeps)


# ────────────────────────────────────────────
#  LIQUIDATION MAP ESTIMATION
# ────────────────────────────────────────────
def estimate_liquidation_zones(current_price, leverages=None):
    """Estima zonas de liquidación en múltiples niveles de apalancamiento."""
    if leverages is None:
        leverages = [5, 10, 15, 20, 25, 35, 50, 75, 100]
    zones = []
    for lev in leverages:
        zones.append({
            'leverage': lev,
            'liq_long': round(current_price * (1 - 1/lev), 2),
            'liq_short': round(current_price * (1 + 1/lev), 2),
            'distance_pct': round(100 / lev, 2),
        })
    return pd.DataFrame(zones)


def find_nearest_liquidation_magnets(entry_price, liq_zones, direction):
    """Zonas de liquidación que actúan como imanes del precio."""
    magnets = []
    if direction == 'LONG':
        for _, z in liq_zones.iterrows():
            if z['liq_short'] > entry_price:
                magnets.append({
                    'level': z['liq_short'], 'leverage': z['leverage'],
                    'distance_pct': (z['liq_short'] - entry_price) / entry_price * 100,
                    'type': 'short_liquidation'
                })
    else:
        for _, z in liq_zones.iterrows():
            if z['liq_long'] < entry_price:
                magnets.append({
                    'level': z['liq_long'], 'leverage': z['leverage'],
                    'distance_pct': (entry_price - z['liq_long']) / entry_price * 100,
                    'type': 'long_liquidation'
                })
    magnets.sort(key=lambda x: x['distance_pct'])
    return magnets[:5]
