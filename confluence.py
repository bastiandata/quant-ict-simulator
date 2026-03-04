"""
confluence.py (V2) — Scoring unificado ponderando fuertemente BOS/CHoCH.
"""

def calculate_confluence_score(entry_price, tp, sl, direction,
                               bullish_fvgs, bearish_fvgs,
                               bullish_obs, bearish_obs,
                               indicator_signals, fib_tp_level,
                               sweeps, liq_magnets, bos_choch_trend,
                               **kwargs):
    """
    Score unificado (0-100) basado en parámetros ajustables.
    """
    weights = kwargs.get('weights', {
        'bos': 20,
        'zones': 15,
        'fib': 10,
        'sweeps': 15,
        'rsi': 10,
        'trend': 10,
        'rr': 10,
        'magnet': 10
    })

    score = 0
    details = []
    tolerance = entry_price * 0.005

    if direction == 'LONG':
        rel_fvgs, rel_obs = bullish_fvgs, bullish_obs
        fvg_label, ob_label = 'Bullish FVG', 'Bullish OB'
        fav_sig, sweep_type = 'bullish', 'bullish_sweep'
    else:
        rel_fvgs, rel_obs = bearish_fvgs, bearish_obs
        fvg_label, ob_label = 'Bearish FVG', 'Bearish OB'
        fav_sig, sweep_type = 'bearish', 'bearish_sweep'

    # 1. Estructura de Mercado
    if bos_choch_trend == direction.lower():
        score += weights['bos']
        details.append(f"  ✅ Estructura BOS alineada ({direction}) (+{weights['bos']})")
    elif bos_choch_trend == 'neutral':
        score += int(weights['bos'] / 2)
        details.append(f"  🔶 Estructura BOS neutral (+{int(weights['bos'] / 2)})")
    else:
        details.append(f"  ❌ Estructura BOS en contra (-0)")

    # 2. Zonas FVG y OB
    in_fvg = _check_zone(entry_price, rel_fvgs, 'fvg_bottom', 'fvg_top', tolerance)
    in_ob  = _check_zone(entry_price, rel_obs, 'ob_low', 'ob_high', tolerance)
    
    if in_fvg and in_ob:
        score += weights['zones']
        details.append(f"  ✅ Golden Zone ({fvg_label} + {ob_label}) (+{weights['zones']})")
    elif in_fvg or in_ob:
        score += int(weights['zones'] / 2)
        details.append(f"  🔶 Precio en Zona ICT (+{int(weights['zones'] / 2)})")
    else:
        details.append(f"  ❌ Precio fuera de zonas ICT (-0)")

    # 3. Fibonacci
    if fib_tp_level is not None:
        dist = abs(fib_tp_level - entry_price) / entry_price * 100
        if dist < 5.0:
            score += weights['fib']
            details.append(f"  ✅ TP Fibonacci realista (+{weights['fib']})")
        else:
            score += int(weights['fib'] / 2)
            details.append(f"  🔶 TP Fibonacci lejano (+{int(weights['fib'] / 2)})")
    else:
        details.append("  ❌ Sin TP Fibonacci válido (-0)")

    # 4. Liquidity Sweeps
    if sweeps is not None and not sweeps.empty:
        fav_sweeps = sweeps[sweeps['sweep_type'] == sweep_type]
        if not fav_sweeps.empty:
            score += weights['sweeps']
            details.append(f"  ✅ Sweep de liquidez a favor (+{weights['sweeps']})")
        else:
            details.append("  ❌ No hay sweeps a favor recientes (-0)")
    else:
        details.append("  ❌ Sin sweeps recientes (-0)")

    # 5. Indicadores
    rsi_sig = next((s for s in indicator_signals if s[0] == 'RSI'), None)
    if rsi_sig and rsi_sig[1] == fav_sig:
        score += weights['rsi']
        details.append(f"  ✅ RSI a favor (+{weights['rsi']})")
    elif rsi_sig and rsi_sig[1] == 'neutral':
        score += int(weights['rsi'] / 2)
        details.append(f"  🔶 RSI neutral (+{int(weights['rsi'] / 2)})")
    else:
        details.append(f"  ❌ RSI en contra (-0)")

    trend_sig = next((s for s in indicator_signals if s[0] == 'EMA Stack'), None)
    if trend_sig and trend_sig[1] == fav_sig:
        score += weights['trend']
        details.append(f"  ✅ Tendencia a favor (+{weights['trend']})")
    elif trend_sig and trend_sig[1] == 'neutral':
        score += int(weights['trend'] / 2)
        details.append(f"  🔶 Tendencia neutral (+{int(weights['trend'] / 2)})")
    else:
        details.append(f"  ❌ Tendencia en contra (-0)")

    # 6. Risk/Reward
    risk = abs(entry_price - sl)
    reward = abs(tp - entry_price)
    rr = reward / risk if risk > 0 else 0
    if rr >= 1.5:
        score += weights['rr']
        details.append(f"  ✅ R:R excelente ({rr:.2f}:1) (+{weights['rr']})")
    elif rr >= 1.0:
        score += int(weights['rr'] / 2)
        details.append(f"  🔶 R:R aceptable ({rr:.2f}:1) (+{int(weights['rr'] / 2)})")
    else:
        details.append(f"  ❌ R:R pobre ({rr:.2f}:1) (-0)")

    # 7. Liquidation Magnets
    if liq_magnets and len(liq_magnets) > 0:
        nearest = liq_magnets[0]
        if nearest['distance_pct'] < 4.0:
            score += weights['magnet']
            details.append(f"  ✅ Imán de liquidez (+{nearest['distance_pct']:.1f}%) (+{weights['magnet']})")
        else:
            score += int(weights['magnet'] / 2)
            details.append(f"  🔶 Imán lejano (+{int(weights['magnet'] / 2)})")
    else:
        details.append("  ❌ Sin imanes direccionales a favor (-0)")

    return min(100, score), details


def _check_zone(price, df, col_low, col_high, tol):
    if df.empty: return False
    for _, z in df.iterrows():
        if z[col_low] - tol <= price <= z[col_high] + tol:
            return True
    return False
