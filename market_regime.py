"""
market_regime.py (V2) — Detección del régimen de mercado y sesgo direccional.
"""

def determine_directional_bias(df, bos_choch_trend, rsi_value):
    """
    Determina un sesgo direccional (-1.0 a +1.0) combinando múltiples factores:
    - 40%: Estructura de mercado (BOS/CHoCH)
    - 30%: EMA Stack (tendencia a mediano plazo)
    - 15%: Posición del precio vs EMA 200 (tendencia largo plazo)
    - 15%: RSI promedio (momentum)

    Retorna: bias_score, detalles
    """
    bias = 0.0
    details = []
    last = df.iloc[-1]

    # ── 1. Estructura BOS/CHoCH (40%) ──
    if bos_choch_trend == 'bullish':
        bias += 0.40
        details.append("  ✅ Estructura BOS/CHoCH alcista (+40%)")
    elif bos_choch_trend == 'bearish':
        bias -= 0.40
        details.append("  🔴 Estructura BOS/CHoCH bajista (-40%)")
    else:
        details.append("  ⚪ Estructura neutral (0%)")

    # ── 2. EMA Stack (30%) ──
    if all(f'ema_{p}' in df.columns for p in [9, 21, 50]):
        e9, e21, e50 = last['ema_9'], last['ema_21'], last['ema_50']
        if e9 > e21 > e50:
            bias += 0.30
            details.append("  ✅ EMA 9 > 21 > 50 alcista (+30%)")
        elif e9 < e21 < e50:
            bias -= 0.30
            details.append("  🔴 EMA 9 < 21 < 50 bajista (-30%)")
        else:
            details.append("  ⚪ EMA Stack cruzadas (0%)")
    else:
        details.append("  ⚪ Sin datos de EMA Stack (0%)")

    # ── 3. Precio vs EMA 200 (15%) ──
    if 'ema_200' in df.columns:
        close = last['close']
        ema200 = last['ema_200']
        if close > ema200:
            bias += 0.15
            details.append(f"  ✅ Precio > EMA 200 alcista (+15%)")
        else:
            bias -= 0.15
            details.append(f"  🔴 Precio < EMA 200 bajista (-15%)")
    else:
        details.append("  ⚪ Sin datos de EMA 200 (0%)")

    # ── 4. Momentum RSI Histórico (15%) ──
    if 'rsi' in df.columns:
        # Promedio del RSI en las últimas 14 velas para ver quién domina
        rsi_mean = df['rsi'].tail(14).mean()
        if rsi_mean > 55:
            bias += 0.15
            details.append(f"  ✅ RSI Promedio ({rsi_mean:.1f}) alcista (+15%)")
        elif rsi_mean < 45:
            bias -= 0.15
            details.append(f"  🔴 RSI Promedio ({rsi_mean:.1f}) bajista (-15%)")
        else:
            details.append(f"  ⚪ RSI Promedio ({rsi_mean:.1f}) neutral (0%)")
    else:
        details.append("  ⚪ Sin datos históricos de RSI (0%)")

    return max(-1.0, min(1.0, bias)), details
